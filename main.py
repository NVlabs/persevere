import datetime
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import hydra
import msgpack
import numpy as np
from loguru import logger
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import (
    EgoLaneChangeStatistics,
)
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.callback.serialization_callback import (
    SerializationCallback,
)
from nuplan.planning.simulation.callback.simulation_log_callback import (
    SimulationLogCallback,
)
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.perfect_tracking import (
    PerfectTrackingController,
)

# from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.runner.simulations_runner import SimulationsRunner
from nuplan.planning.simulation.simulation import SimulationSetup
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from omegaconf import DictConfig, ListConfig, OmegaConf

from postprocessor import postprocess
from tools import validate_scenario_config
from severity_estimation.fault_injection.common_failures import *
from severity_estimation.fault_injection.failures import FaultInjectionManager
from severity_estimation.fault_injection.simulator import Simulator
from severity_estimation.fault_injection.datatypes import (  # necessary for configuration
    Angle,
    Constant,
    Gaussian,
    Offset,
    Position,
    Size,
    Uniform,
)
from severity_estimation.planner.idm_agents import IDMAgents
from severity_estimation.planner.metrics.collision_metric import (
    FaultAwareCollisionStatistics,
)
from severity_estimation.planner.prediction_risk_planner import PredictionRiskPlanner
from severity_estimation.planner.risk_planner_results import RiskPlannerResults
from severity_estimation.utils.scenario_stepper import ScenarioStepper

fmt = "<green>{time:MM.DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
logger.remove()  # All configured handlers are removed
logger.add(sys.stderr, format=fmt)

pi = np.pi  # used in the scenario configuration


def simulate(
    cfg,
    scenario: NuPlanScenario,
    failures: FaultInjectionManager,
    output_folder: Path,
    callbacks=MultiCallback([]),
) -> List[float]:
    # Agents and observations
    if cfg.simulation.reactive_agents:
        observations = IDMAgents(
            target_velocity=10,
            min_gap_to_lead_agent=0.5,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=2.0,
            scenario=scenario,
            open_loop_detections_types=[
                "PEDESTRIAN",
                "BICYCLE",
                "GENERIC_OBJECT",
                "TRAFFIC_CONE",
                "BARRIER",
            ],
        )
        ego_controller = PerfectTrackingController(scenario)
    else:
        observations = TracksObservation(scenario=scenario)
        ego_controller = LogPlaybackController(scenario=scenario)
    # Simulation
    sim_manager = ScenarioStepper(
        scenario=scenario, max_duration=cfg.simulation.max_duration
    )
    simulation_setup = SimulationSetup(
        time_controller=sim_manager,
        observations=observations,
        ego_controller=ego_controller,
        scenario=scenario,
    )
    simulation = Simulator(
        simulation_setup=simulation_setup,
        callback=callbacks,
        simulation_history_buffer_duration=cfg.simulation.simulation_history_buffer_duration,
        faults=failures,
    )
    # Planner
    plot_dir = (
        output_folder / cfg.results.plots / "trajectron" / scenario.token
        if cfg.planner.PredictionRiskPlanner.verbose
        else None
    )
    planner = PredictionRiskPlanner(cfg, plot_folder=plot_dir)
    runner = SimulationsRunner([simulation], planner)
    report = runner.run()[0]
    return RiskPlannerResults(
        start_time=report.start_time,
        end_time=report.end_time,
        compute_plan_runtimes=report.planner_report.compute_plan_runtimes,
        compute_trajectory_runtimes=report.planner_report.compute_trajectory_runtimes,
        prediction_runtimes=report.planner_report.prediction_runtimes,
        compute_cost_runtimes=report.planner_report.compute_cost_runtimes,
        risk_costs=report.planner_report.risk_costs,
        succeeded=report.succeeded,
        timesteps_us=report.planner_report.timesteps_us,
    )


@hydra.main(config_path="configs", config_name="general.yaml")
def main_app(cfg: DictConfig) -> None:
    experiment_time = time.strftime("%Y.%m.%d_%H.%M.%S")
    output_dir = Path(cfg.results.outdir) / experiment_time
    logger.info(f"Saving results to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Backup configuration for traceability
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    # Scenario Selection
    assert isinstance(cfg.scenarios, ListConfig), "scenarios must be a list"
    validate_scenario_config(cfg.scenarios)
    # Load scenarios
    filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=[s.name for s in cfg.scenarios],
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        timestamp_threshold_s=None,
    )
    scenario_builder = NuPlanScenarioBuilder(
        data_root=cfg.nuplan.DATA_ROOT,
        map_root=cfg.nuplan.MAPS_ROOT,
        db_files=cfg.nuplan.DB_FILES,
        map_version=cfg.nuplan.MAP_VERSION,
    )
    scenarios = scenario_builder.get_scenarios(filter, Sequential())

    # Callbacks
    ego_lane_change_metric = EgoLaneChangeStatistics(
        "ego_lane_change_statistics", "Planning", max_fail_rate=0.3
    )
    collision_metric = FaultAwareCollisionStatistics(
        "collisions_statistics", "Planning", ego_lane_change_metric
    )
    metric_engine = MetricsEngine(
        metrics=[ego_lane_change_metric, collision_metric],
        main_save_path=output_dir / cfg.results.metrics_dir,
        timestamp=0,
    )
    metrics = MetricCallback(metric_engine=metric_engine)
    sim_callbacks = MultiCallback(
        [
            SimulationLogCallback(
                output_directory=str(output_dir),
                simulation_log_dir=cfg.results.sim_logs_dir,
                serialization_type="msgpack",
            ),
            SerializationCallback(
                output_directory=str(output_dir),
                folder_name=cfg.results.sim_dir,
                serialize_into_single_file=True,
                serialization_type="msgpack",
            ),
            metrics,
        ]
    )

    # Setup NuBoard
    nuboard_file = NuBoardFile(
        simulation_main_path=str(output_dir),
        metric_main_path=str(output_dir),
        metric_folder=cfg.results.metrics_dir,
        simulation_folder=cfg.results.sim_logs_dir,
        aggregator_metric_folder=cfg.results.aggregator_metric_dir,
    )
    nuboard_file_name = output_dir / ("nuboard_file" + nuboard_file.extension())
    nuboard_file.save_nuboard_file(nuboard_file_name)

    # Run simulations...
    num_scenarios = len(scenarios)
    if num_scenarios != len(cfg.scenarios):
        logger.error(
            f"Error while loading scenarios, could load {num_scenarios} out of {len(cfg.scenarios)} scenarios."
        )
        exit(1)
    results = defaultdict(lambda: defaultdict(dict))
    runtimes = []
    avg_runtime = None
    for idx, scenario in enumerate(scenarios):
        start_time = time.perf_counter()
        if avg_runtime is not None:
            avg = f"{datetime.timedelta(seconds=int(avg_runtime))} /it"
            etc = f"{datetime.timedelta(seconds=int(avg_runtime * (num_scenarios - idx)))} ETC"
        else:
            avg, etc = "N/A", "N/A"
        logger.info(
            f"{idx+1}/{num_scenarios} ({idx/num_scenarios*100:.0f}%): Processing scenario {scenario.scenario_name} [ {avg} | {etc} ]"
        )
        # Find scenario config
        scenario_cfg = None
        for c in cfg.scenarios:
            if c.log == scenario.log_name and c.name == scenario.scenario_name:
                scenario_cfg = c
                break
        # Simulate scenario (with failures)
        failure_generators = (
            [eval(gen) for gen in scenario_cfg.failures]
            if scenario_cfg.failures
            else []
        )
        failure_injector = FaultInjectionManager(failure_generators)
        report = simulate(
            cfg,
            scenario,
            callbacks=sim_callbacks,
            failures=failure_injector,
            output_folder=output_dir,
        )
        results[scenario.log_name][scenario.scenario_name] = report.__dict__
        runtimes.append(time.perf_counter() - start_time)
        avg_runtime = sum(runtimes) / len(runtimes)

    # Save results
    with open(output_dir / cfg.results.result_file, "wb") as handle:
        msgpack.dump(dict(results), handle, use_bin_type=True)

    logger.info(f"Simulation completed!")

    # Postprocess results...
    postprocessed_results = postprocess(cfg, output_dir)
    with open(output_dir / cfg.results.postprocessed_file, "wb") as handle:
        pickle.dump(postprocessed_results, handle)

    logger.info(
        f"Done!\nYou can visualize results with:\n > poetry run viz {output_dir} --nuboard"
    )


if __name__ == "__main__":
    main_app()
