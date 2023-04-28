from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects

from severity_estimation.fault_injection.failures import (
    FaultyDetectionsTracks,
    FaultyEgoState,
    FaultyTrafficLightStatusData,
)
from severity_estimation.fault_injection.utils import clear_agent_cache


class TraffcRuleViolation(Enum):
    RED_LIGHT = 1


@dataclass
class TrajectronTrackedObject:
    tracked_object: TrackedObject
    skip: bool = False


@dataclass
class Hypothesis:
    observations: List[TrackedObject]
    violations: List[TraffcRuleViolation]


class VelocityGaussian:
    def __init__(self, direction_std, magnitude_std):
        assert direction_std >= 0, "Direction standard deviation must be non-negative"
        assert magnitude_std >= 0, "Magnitude standard deviation must be non-negative"
        self._direction_std = direction_std
        self._magnitude_std = magnitude_std

    def sample(self):
        return (
            np.random.normal(0, self._direction_std),
            np.random.normal(1, self._magnitude_std),
        )


class PositionGaussian:
    def __init__(self, x_std, y_std, heading_std):
        assert x_std >= 0, "X standard deviation must be non-negative"
        assert y_std >= 0, "Y standard deviation must be non-negative"
        assert heading_std >= 0, "Heading standard deviation must be non-negative"
        self._x_std = x_std
        self._y_std = y_std
        self._heading_std = heading_std

    def sample(self):
        return (
            np.random.normal(0, self._x_std),
            np.random.normal(0, self._y_std),
            np.random.normal(0, self._heading_std),
        )


class HypothesisGenerator:
    def __init__(self, cfg):
        if hasattr(cfg.hypothesis_generator, "use_ground_truth"):
            self._use_ground_truth = cfg.hypothesis_generator.use_ground_truth
        else:
            self._use_ground_truth = False
        if hasattr(cfg.hypothesis_generator, "velocity_magnitude"):
            assert ~self._use_ground_truth, "Cannot fix velocity magnitude if using ground truth"
            self._velocity_magnitude = cfg.hypothesis_generator.velocity_magnitude
        else:
            self._velocity_magnitude = set()
        if hasattr(cfg.hypothesis_generator, "velocity_noise"):
            assert ~self._use_ground_truth, "Cannot add velocity noise if using ground truth"
            self._velocity_noise = dict()
            for cat, inst in cfg.hypothesis_generator.velocity_noise.items():
                if inst is not None:
                    self._velocity_noise[cat] = eval(inst)
        else:
            self._velocity_noise = set()
        if hasattr(cfg.hypothesis_generator, "position_noise"):
            assert ~self._use_ground_truth, "Cannot add position noise if using ground truth"
            self._position_noise = dict()
            for cat, inst in cfg.hypothesis_generator.position_noise.items():
                if inst is not None:
                    self._position_noise[cat] = eval(inst)
        else:
            self._position_noise = set()

    def _fix_velocity(self, agent: Agent, magnitude: float):
        agent.velocity.x = magnitude * np.cos(agent.center.heading)
        agent.velocity.y = magnitude * np.sin(agent.center.heading)
        clear_agent_cache(agent)

    def _add_velocity_noise(self, agent: Agent, noise):
        theta, magnitude = noise.sample()
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        v = np.dot(rot, agent.velocity.array)
        agent.velocity.x, agent.velocity.y = v[0], v[1]
        orientation = np.arctan2(agent.velocity.y, agent.velocity.x)
        agent.velocity.x *= magnitude * np.cos(orientation)
        agent.velocity.y *= magnitude * np.sin(orientation)
        clear_agent_cache(agent)

    def _add_position_noise(self, agent: Agent, noise):
        x, y, h = noise.sample()
        agent.center.x += x
        agent.center.y += y
        agent.center.heading += h
        agent.center.heading = np.mod(agent.center.heading, 2 * np.pi)
        clear_agent_cache(agent)

    def __call__(
        self,
        ego_states: List[FaultyEgoState],
        observations: List[FaultyDetectionsTracks],
        traffic_lights: Optional[List[FaultyTrafficLightStatusData]] = None,
    ):
        if self._use_ground_truth:
            hyp_observations = [observations[i].gt_tracked_objects for i in range(len(observations))]
        else:
            hyp_observations = []
            for obs in observations:
                tracked_objects = []
                for obj in obs.gt_tracked_objects:
                    if obj.track_token in obs.failures_by_token:
                        hobj: Agent = deepcopy(obj)
                        cat = hobj.metadata.category_name
                        if cat in self._velocity_magnitude:
                            self._fix_velocity(hobj, self._velocity_magnitude[cat])
                        if cat in self._velocity_noise:
                            self._add_velocity_noise(hobj, self._velocity_noise[cat])
                        if cat in self._position_noise:
                            self._add_position_noise(hobj, self._position_noise[cat])
                    else:
                        hobj = obj
                    tracked_objects.append(hobj)
                hyp_observations.append(TrackedObjects(tracked_objects))

        violations = []
        if traffic_lights is not None:
            traffic_light_failures = frozenset().union(*[tl.active_failures for tl in traffic_lights])
            if "TrafficLightMisdetection" in traffic_light_failures:
                # TODO[antonap]: should activate only if the ego is actually crossing the intersection
                violations.append(TraffcRuleViolation.RED_LIGHT)

        return Hypothesis(hyp_observations, violations)
