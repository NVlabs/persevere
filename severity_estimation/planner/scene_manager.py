from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from severity_estimation.planner.utils import add_observations_to_scene
from severity_estimation.trajectron.datatypes import Node

PredictionType = Dict[int, Dict[Node, np.ndarray]]


@dataclass(frozen=True)
class Prediction:
    step: int
    perceived_predictions: Optional[PredictionType]
    hypothesis_predictions: PredictionType
    runtime: float

    @property
    def perceived(self):
        return self.perceived_predictions[self.step]
        # return self.perceived_predictions

    @property
    def hypothesis(self):
        return self.hypothesis_predictions[self.step]
        # return self.hypothesis_predictions


class SceneManager:
    def __init__(self, ctx, scene, scene_radius, subsample_ratio):
        self.scene = scene
        self._ctx = ctx
        self._scene_radius = scene_radius
        self._subsample_ratio = subsample_ratio

    def update(self, ego_states, observations, steps):
        # Setup scene for Trajectron
        if steps == -1:
            buffer_ind = list(range(0, len(ego_states), round(1 / self._subsample_ratio)))
            steps += len(buffer_ind)
            ego_state = ego_states[0]
            self.scene.x_min = ego_state.center.x - self._scene_radius
            self.scene.y_min = ego_state.center.y - self._scene_radius
            self.scene.x_max = ego_state.center.x + self._scene_radius
            self.scene.y_max = ego_state.center.y + self._scene_radius
        else:
            buffer_ind = [-1]
            steps += 1
        # Add current ego, vehicles, and pedestrians to the scene
        for i in buffer_ind:
            add_observations_to_scene(self._ctx, self.scene, ego_states[i], observations[i])
        return steps

    @property
    def dt(self):
        return self.scene.dt
