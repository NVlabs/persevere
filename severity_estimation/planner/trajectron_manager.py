import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from severity_estimation.planner.scene_manager import Prediction, SceneManager
from severity_estimation.planner.utils import (
    context,
    convert_trajectory_to_node,
    load_trajectron_model,
)
from severity_estimation.trajectron.visualization import vis
import torch.multiprocessing as mp


class TrajectronManager:
    def __init__(self, cfg, planned_horizon):
        self._planned_horizon = planned_horizon
        self._scene_radius = cfg.trajectron.scene_radius
        self._subsample_ratio = cfg.trajectron.subsample_ratio
        self._samples = cfg.trajectron.num_samples
        self._prediction_runtimes = dict()

        self.ctx = context(cfg)
        eval_env, self.model = load_trajectron_model(self.ctx)
        self.base_scene = SceneManager(
            self.ctx, eval_env.scenes[0], self._scene_radius, self._subsample_ratio
        )
        self.hyp_scene = SceneManager(
            self.ctx,
            deepcopy(self.base_scene.scene),
            self._scene_radius,
            self._subsample_ratio,
        )
        self._ego_node = None

        self._ph = round(self._planned_horizon / self.base_scene.scene.dt)
        self._tau = -1
        self._sim_step = -1
        self._last_prediction_time = -1

        self.__plot_subsample = 0

        mp.set_start_method("spawn", force=True)

    @property
    def num_scenes(self) -> int:
        return 2

    @property
    def dt(self):
        return self.cscene.scene.dt

    @property
    def last_prediction_time(self):
        return self._last_prediction_time

    @property
    def prediction_horizon(self):
        return self._ph

    @property
    def ego_node(self):
        return self._ego_node

    @property
    def step(self):
        return self._sim_step

    @property
    def prediction_runtimes(self):
        return self._prediction_runtimes

    def reset(self):
        self._prediction_runtimes = dict()
        self._tau = -1
        self._sim_step = -1
        self._last_prediction_time = -1

    def update(self, ego_states, observations, hyp_observations, trajectory):
        self.base_scene.update(
            ego_states=ego_states,
            observations=observations,
            steps=self._sim_step,
        )
        self._sim_step = self.hyp_scene.update(
            ego_states=ego_states,
            observations=hyp_observations,
            steps=self._sim_step,
        )
        self._ego_node = convert_trajectory_to_node(
            self.ctx,
            trajectory,
            self.base_scene.scene,
            first_timestep=self._sim_step + 1,
        )
        self._ego_node.width = ego_states[-1].agent._box.width
        self._ego_node.length = ego_states[-1].agent._box.length
        self._tau = (
            (self._sim_step - self._last_prediction_time)
            if self._last_prediction_time > 0
            else -1
        )

    def predict(self):
        out_predictions = None
        if self._tau == self._ph or self._last_prediction_time < 0:
            self._last_prediction_time = self._sim_step
            self._tau = 0
            timestep = np.array([self._sim_step])
            start_t = time.perf_counter()
            with torch.no_grad():
                predictions = self.model.predict(
                    self.base_scene.scene,
                    timestep,
                    self._ph,
                    num_samples=self._samples,
                    full_dist=False,
                )
            with torch.no_grad():
                hyp_predictions = self.model.predict(
                    self.hyp_scene.scene,
                    timestep,
                    self._ph,
                    num_samples=self._samples,
                    full_dist=False,
                )
            self._prediction_runtimes[self._sim_step] = time.perf_counter() - start_t
            logger.trace(
                f"Prediction runtime: {self._prediction_runtimes[self._sim_step]}"
            )
            out_predictions = Prediction(
                step=self._sim_step,
                perceived_predictions=predictions,
                hypothesis_predictions=hyp_predictions,
                runtime=self._prediction_runtimes[self._sim_step],
            )
        return out_predictions

    def plot(self, predictions, fname):
        if self.__plot_subsample % 5 == 0:
            start_t = time.perf_counter()
            fig, ax = plt.subplots()
            vis.visualize_prediction(
                ax=ax,
                prediction_output_dict=predictions,
                dt=self.hyp_scene.dt,
                max_hl=2,
                ph=self._ph,
                robot_node=self.hyp_scene.scene.robot,
            )
            if self.hyp_scene.scene.robot is not None:
                prediction_timesteps = np.array(
                    [
                        self._last_prediction_time - self._ph + 1,
                        self._last_prediction_time,
                    ]
                )
                state_dict = {
                    "position": ["x", "y"],
                    "velocity": ["x", "y"],
                    "acceleration": ["x", "y"],
                    "heading": ["°", "d°"],
                }
                robot_for_plotting = self.hyp_scene.scene.robot.get(
                    prediction_timesteps, state_dict
                )
                # Trajectory
                ax.plot(
                    self._ego_node.data.data[:, 0],
                    self._ego_node.data.data[:, 1],
                    color="r",
                    linewidth=0.5,
                    alpha=1.0,
                )
                # Current Node Position
                circle = plt.Circle(
                    (robot_for_plotting[0, 0], robot_for_plotting[0, 1]),
                    0.3,
                    facecolor="r",
                    edgecolor="r",
                    lw=0.5,
                    zorder=3,
                )
                ax.add_artist(circle)
            plt.savefig(fname, dpi=300)
            plt.close(fig)
            logger.trace(f"Saved plot to {fname} ({time.perf_counter() - start_t:.3f}s")
        self.__plot_subsample += 1
