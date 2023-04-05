import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from severity_estimation.cost.cost_util import CostUtil
from severity_estimation.hamilton_jacobi.hj_severity import HJSeverity
from severity_estimation.planner.hypothesis.generator import (
    HypothesisGenerator,
    TraffcRuleViolation,
)
from severity_estimation.planner.risk_planner_report import RiskPlannerReport
from severity_estimation.planner.trajectron_manager import TrajectronManager
# from severity_estimation.planner.trajectron_manager_online import TrajectronManager


class PredictionRiskPlanner(IDMPlanner):
    def __init__(self, cfg, plot_folder: Optional[Path] = None):
        super().__init__(
            target_velocity=cfg.planner.IDMPlanner.target_velocity,
            min_gap_to_lead_agent=cfg.planner.IDMPlanner.min_gap_to_lead_agent,
            headway_time=cfg.planner.IDMPlanner.headway_time,
            accel_max=cfg.planner.IDMPlanner.accel_max,
            decel_max=cfg.planner.IDMPlanner.decel_max,
            planned_trajectory_samples=cfg.planner.IDMPlanner.planned_trajectory_samples,
            planned_trajectory_sample_interval=cfg.planner.IDMPlanner.planned_trajectory_sample_interval,
            occupancy_map_radius=cfg.planner.IDMPlanner.occupancy_map_radius,
        )

        self._cfg = cfg
        self._plot_dir = plot_folder
        if self._plot_dir is not None:
            self._plot_dir.mkdir(parents=True, exist_ok=True)

        self.hyp_generator = HypothesisGenerator(cfg)
        self._trajectron = TrajectronManager(cfg, self._planned_horizon)

        # Load HJ LUT table
        self._hj = HJSeverity.load(cfg.hj_reachability.lut_table)
        self._hj_scene_radius = cfg.hj_reachability.scene_radius

        self.risk_costs = dict()
        self._timestemps_us = dict()
        self._compute_plan_runtimes = list()
        self._compute_cost_runtimes = dict()
        self._compute_trajectory_runtimes = list()

    def __getstate__(self):
        d = {"cfg": self._cfg}
        return d

    def __setstate__(self, state):
        self.__init__(state["cfg"])

    def _violations_cost(self, hyp_predictions, violations, prediction_timesteps):
        cost = 0.0
        if violations:
            cost = 1.0
        return cost

    def _compute_costs(
        self, cost_fcn, predictions, hyp_predictions, violations, prediction_timesteps
    ):
        start_t = time.perf_counter()
        base = cost_fcn(
            self._trajectron.base_scene.scene,
            self._trajectron.ego_node,
            prediction_timesteps,
            prediction_dict=predictions,
            split_agents=False,
        )
        hyp = cost_fcn(
            self._trajectron.hyp_scene.scene,
            self._trajectron.ego_node,
            prediction_timesteps,
            prediction_dict=hyp_predictions,
            split_agents=False,
        )
        base_costs, hyp_costs = np.amax(base, axis=1), np.amax(hyp, axis=1)
        violation_cost = self._violations_cost(
            hyp_predictions, violations, prediction_timesteps
        )
        cost = {
            "base": base_costs.tolist(),
            "hyp": (hyp_costs + violation_cost).tolist(),
        }
        runtime = time.perf_counter() - start_t
        return cost, runtime

    def _compute_risk_costs(
        self, predictions, hyp_predictions, violations: List[TraffcRuleViolation]
    ):
        prediction_timesteps = np.array(
            [
                self._trajectron.last_prediction_time + 1,
                self._trajectron.last_prediction_time
                + self._trajectron.prediction_horizon,
            ]
        )
        runtimes = dict()
        costs = dict()
        # Momentum Shape Distance
        costs["msd"], runtimes["msd"] = self._compute_costs(
            CostUtil.compute_momentum_shape_distance,
            predictions,
            hyp_predictions,
            violations,
            prediction_timesteps,
        )
        # Time To Collision
        costs["ttc"], runtimes["ttc"] = self._compute_costs(
            CostUtil.compute_time_to_collision_cost,
            predictions,
            hyp_predictions,
            violations,
            prediction_timesteps,
        )
        return costs, runtimes

    def compute_hj_reachability_risk(self, ego_state, observations, hypothesis):
        base = CostUtil.compute_hj_reachability_cost(
            ego_state,
            observations.tracked_objects,
            self._hj_scene_radius,
            self._hj,
        )
        start_t = time.perf_counter()
        hyp = CostUtil.compute_hj_reachability_cost(
            ego_state,
            hypothesis.observations[-1],
            self._hj_scene_radius,
            self._hj,
        )
        runtime = time.perf_counter() - start_t
        cost = {"base": base, "hyp": hyp}
        return cost, runtime

    def compute_collision_probability(self, predictions, hyp_predictions):
        prediction_timesteps = np.array(
            [
                self._trajectron.last_prediction_time + 1,
                self._trajectron.last_prediction_time
                + self._trajectron.prediction_horizon,
            ]
        )
        start_t = time.perf_counter()
        base_probs = CostUtil.compute_collision_probability(
            scene=self._trajectron.base_scene.scene,
            ego_node=self._trajectron.ego_node,
            timesteps=prediction_timesteps,
            prediction_dict=predictions,
        )
        hyp_probs = CostUtil.compute_collision_probability(
            scene=self._trajectron.hyp_scene.scene,
            ego_node=self._trajectron.ego_node,
            timesteps=prediction_timesteps,
            prediction_dict=hyp_predictions,
        )
        cost = {"base": base_probs.tolist(), "hyp": hyp_probs.tolist()}
        runtime = time.perf_counter() - start_t
        return cost, runtime

    def compute_planner_trajectory(
        self, current_input: List[PlannerInput]
    ) -> List[AbstractTrajectory]:
        # Ego current state
        history = current_input[0].history
        ego_state, observations = history.current_state
        traffic_light_data = history.current_traffic_light
        if not self._initialized:
            self._initialize_ego_path(ego_state)
            self._initialized = True

        compute_plan_start_t = time.perf_counter()
        # Compute Trajectory
        start_t = time.perf_counter()
        occupancy_map, unique_observations = self._construct_occupancy_map(
            ego_state, observations
        )
        self._annotate_occupancy_map(traffic_light_data, occupancy_map)
        trajectory = self._get_planned_trajectory(
            ego_state, occupancy_map, unique_observations
        )
        self._compute_trajectory_runtimes.append(time.perf_counter() - start_t)

        # Generate hypotheses
        actual_observations = [
            history.observations[i].tracked_objects for i in range(history.size)
        ]
        hyp = self.hyp_generator(
            history.ego_states, history.observations, history.current_traffic_light
        )

        # Predict trajectories
        self._trajectron.update(
            history.ego_states, actual_observations, hyp.observations, trajectory
        )
        predictions = self._trajectron.predict()
        if predictions is not None:
            # Plot
            if self._plot_dir is not None:
                fname = self._plot_dir / f"{ego_state.time_us}.png"
                self._trajectron.plot(predictions.hypothesis_predictions, fname)
            # Compute risk cost
            costs, runtimes = self._compute_risk_costs(
                predictions.perceived,
                predictions.hypothesis,
                hyp.violations,
            )
            costs["hj"], runtimes["hj"] = self.compute_hj_reachability_risk(
                *history.current_state, hyp
            )
            costs["cp"], runtimes["cp"] = self.compute_collision_probability(
                predictions.perceived, predictions.hypothesis
            )
            self.risk_costs[self._trajectron.step] = costs
            self._compute_cost_runtimes[self._trajectron.step] = runtimes
            self._timestemps_us[self._trajectron.step] = ego_state.time_point.time_us

        self._compute_plan_runtimes.append(time.perf_counter() - compute_plan_start_t)
        return [trajectory]

    def generate_planner_report(self, clear_stats: bool = True) -> RiskPlannerReport:
        report = RiskPlannerReport(
            risk_costs=deepcopy(self.risk_costs),
            compute_plan_runtimes=self._compute_plan_runtimes.copy(),
            compute_trajectory_runtimes=self._compute_trajectory_runtimes.copy(),
            prediction_runtimes=self._trajectron.prediction_runtimes.copy(),
            compute_cost_runtimes=deepcopy(self._compute_cost_runtimes),
            timesteps_us=self._timestemps_us.copy(),
        )
        if clear_stats:
            self._trajectron.reset()
            self._compute_plan_runtimes = list()
            self._compute_trajectory_runtimes = list()
            self.risk_costs = dict()
            self._compute_cost_runtimes = dict()
            self._timestemps_us = dict()
        return report
