import datetime
import sys
import time
from pathlib import Path

import hydra
from loguru import logger
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.callback.serialization_callback import (
    SerializationCallback,
)
from nuplan.planning.simulation.callback.simulation_log_callback import (
    SimulationLogCallback,
)
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationsRunner
from nuplan.planning.simulation.simulation import Simulation, SimulationSetup
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from omegaconf import DictConfig, ListConfig, OmegaConf

from severity_estimation.fault_injection.simulator import Simulator
from severity_estimation.utils.scenario_stepper import ScenarioStepper

fmt = "<green>{time:MM.DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
logger.remove()  # All configured handlers are removed
logger.add(sys.stderr, format=fmt)


def expert_trajectory(cfg, scenario, callbacks=MultiCallback([])) -> None:
    observations = TracksObservation(scenario=scenario)
    ego_controller = LogPlaybackController(scenario=scenario)
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
    )
    planner = SimplePlanner(2, 0.5, [0, 0])
    runner = SimulationsRunner([simulation], planner)
    runner.run()


@hydra.main(config_path="configs", config_name="general.yaml")
def main_app(cfg: DictConfig) -> None:
    experiment_time = time.strftime("%Y.%m.%d_%H.%M.%S")
    output_dir = Path(cfg.results.outdir) / experiment_time
    # Scenario Selection
    if isinstance(cfg.scenarios, ListConfig):
        filter = ScenarioFilter(
            scenario_types=[s.type for s in cfg.scenarios],
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
    elif isinstance(cfg.scenarios, DictConfig):
        filter = ScenarioFilter(
            scenario_types=cfg.scenarios.scenario_types,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=cfg.scenarios.limit_scenarios_per_type,
            limit_total_scenarios=cfg.scenarios.limit_total_scenarios,
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=False,
            timestamp_threshold_s=None,
        )
    else:
        raise ValueError(
            "Scenario selection must be either a list of scenarios or a filter configuration"
        )

    scenario_builder = NuPlanScenarioBuilder(
        data_root=cfg.nuplan.DATA_ROOT,
        map_root=cfg.nuplan.MAPS_ROOT,
        db_files=cfg.nuplan.DB_FILES,
        map_version=cfg.nuplan.MAP_VERSION,
    )
    scenarios = scenario_builder.get_scenarios(filter, Sequential())

    # Callbacks
    callbacks = MultiCallback(
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
        ]
    )

    # File setup
    output_dir.mkdir(parents=True, exist_ok=True)
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

    runtimes = []
    avg_runtime = None
    num_scenarios = len(scenarios)
    if num_scenarios != len(cfg.scenarios):
        logger.error(
            f"Error while loading scenarios, could load {num_scenarios} out of {len(cfg.scenarios)} scenarios."
        )
        exit(1)
    for idx, scenario in enumerate(scenarios):
        start_time = time.perf_counter()
        if avg_runtime is not None:
            avg = f"{datetime.timedelta(seconds=avg_runtime)} /it"
            etc = (
                f"{datetime.timedelta(seconds=avg_runtime * (num_scenarios - idx))} ETC"
            )
        else:
            avg, etc = "N/A", "N/A"
        logger.info(
            f"{idx}/{num_scenarios} ({idx/num_scenarios*100:.0f}%): Processing scenario {scenario.scenario_name} [ {avg} | {etc} ]"
        )
        expert_trajectory(cfg, scenario, callbacks)
        runtimes.append(time.perf_counter() - start_time)
        avg_runtime = sum(runtimes) / len(runtimes)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    logger.info(
        f"Done!\nYou can visualize results with:\n > poetry run viz {output_dir} --nuboard"
    )


if __name__ == "__main__":
    main_app()
