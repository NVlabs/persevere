from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from shapely.geometry import Point
from shapely.ops import nearest_points
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from severity_estimation.fault_injection.datatypes import (
    Number,
    Angle,
    Constant,
    Offset,
    Position,
    Size,
)
from severity_estimation.fault_injection.failures import FailureGenerator, FaultyState
from severity_estimation.fault_injection.utils import (
    clear_agent_cache,
    clear_ego_cache,
    find_agent,
    mutate_agent,
    random_token,
    remove_agent_from,
)


class MissedObstacle(FailureGenerator):
    def __init__(self, token: str):
        self._target_token = token
        assert token is not None, "Token must be specified"

    def reset(self):
        pass

    def __call__(self, state: FaultyState):
        tracked_objects = state.detections_tracks.tracked_objects.tracked_objects
        removed = remove_agent_from(tracked_objects, self._target_token)
        if removed:
            state.detections_tracks.failures_by_token[self._target_token].add(
                self.__class__.__name__
            )


class Misdetection(FailureGenerator):
    def __init__(
        self,
        token: str,
        offset: Offset = Offset(0, 0),
        shape_ratio: Size = Size(1, 1),
        rotation: Angle = Angle(Constant(0)),
        velocity_ratio: Number = Constant(1),
        object_type: Optional[str] = None,
    ):
        self._target_token = token
        self._shape_ratio = shape_ratio
        self._offset = offset
        self._rotation = rotation
        self._velocity_ratio = velocity_ratio
        if object_type is not None:
            self._category = object_type.lower()
            assert (
                self._category in FailureGenerator.OBJECT_TYPES
            ), f"Invalid object type {object_type}"
        else:
            self._category = None

    def reset(self):
        pass

    def __call__(self, state: FaultyState):
        detection_tracks = state.detections_tracks.tracked_objects
        offset = self._offset.get()
        shape_ratio = self._shape_ratio.get()
        rotation = self._rotation.get()
        vel_magnitude = self._velocity_ratio.get()
        agent = find_agent(detection_tracks.tracked_objects, self._target_token)
        if agent is not None:
            mutate_agent(
                agent, offset, shape_ratio, rotation, vel_magnitude, self._category
            )
            state.detections_tracks.failures_by_token[self._target_token].add(
                self.__class__.__name__
            )


class GhostObstacle(FailureGenerator):
    def __init__(
        self,
        offset: Union[Offset, Position],
        rotation: Angle = Angle(Constant(0)),
        size: Optional[Size] = None,
        velocity_ratio: Number = Constant(1),
        object_type: str = "vehicle",
    ):
        self._token = random_token()
        self._track_token = random_token()
        self._size = size
        self._rotation = rotation
        self._velocity_ratio = velocity_ratio
        self._offset = offset
        self._category = object_type.lower()
        assert (
            self._category in FailureGenerator.OBJECT_TYPES
        ), f"Invalid object type {object_type}"

    def reset(self):
        self._token = random_token()
        self._track_token = random_token()

    def _validate(self, detection_tracks: List[Agent]) -> bool:
        for obj in detection_tracks.tracked_objects:
            if obj.track_token == self._track_token:
                self.reset()
                return self._validate(detection_tracks)
        return True

    def __call__(self, state: FaultyState):
        detection_tracks = state.detections_tracks.tracked_objects
        self._validate(detection_tracks)
        offset = self._offset.get()
        size = self._size.get() if self._size is not None else None
        rotation = self._rotation.get()
        vel_ratio = self._velocity_ratio.get()
        box = deepcopy(state.ego_state.agent._box)
        if isinstance(offset, Position):
            box._center.x = offset.x
            box._center.y = offset.y
            box._center.heading = offset.heading
        else:
            box._center.x += np.cos(box.center.heading + offset.angle) * offset.distance
            box._center.y += np.sin(box.center.heading + offset.angle) * offset.distance
            box._center.heading += rotation
        if self._size is not None:
            box._width = size.width
            box._length = size.length
        metadata = SceneObjectMetadata(
            state.ego_state.agent.metadata.timestamp_us,
            self._token,
            None,
            self._track_token,
            self._category,
        )
        velocity = deepcopy(state.ego_state.agent._velocity)
        velocity.x *= vel_ratio
        velocity.y *= vel_ratio
        agent = Agent(
            tracked_object_type=FailureGenerator.OBJECT_TYPES[self._category],
            oriented_box=box,
            metadata=metadata,
            velocity=velocity,
            angular_velocity=deepcopy(state.ego_state.agent._angular_velocity),
            predictions=[],
            past_trajectory=None,
        )
        detection_tracks.tracked_objects.append(agent)
        state.detections_tracks.failures_by_token[self._track_token].add(
            self.__class__.__name__
        )
        # Detection tracks need to be sorted and _ranges_per_type deleted
        detection_tracks.tracked_objects = sorted(
            detection_tracks.tracked_objects, key=lambda agent: agent.tracked_object_type.value  # type: ignore
        )
        if hasattr(detection_tracks, "_ranges_per_type"):
            del detection_tracks._ranges_per_type


class TrafficLightMisdetection(FailureGenerator):
    def __init__(
        self,
        selector: Union[List[int], str],
        state: str,
    ):
        self._radius = 2
        if isinstance(selector, str):
            self._selector = selector.lower()
        else:
            self._selector = selector if isinstance(selector, list) else [selector]
        self._state = TrafficLightStatusType.deserialize(state.upper())

    def reset(self):
        pass

    def _get_lane_connector_ids(self, state: FaultyState):
        if isinstance(self._selector, str):
            if self._selector == "all":
                return [light.lane_connector_id for light in state.traffic_lights]
            elif self._selector == "proximal":
                pt = state.ego_state.gt_center
                return [
                    int(l.id)
                    for l in state.map_api.get_proximal_map_objects(
                        pt, self._radius, [SemanticMapLayer.LANE_CONNECTOR]
                    )[SemanticMapLayer.LANE_CONNECTOR]
                ]
            else:
                raise ValueError(f"Invalid selector {self._selector}")
        else:
            return self._selector

    def __call__(self, state: FaultyState):
        if state.traffic_lights is None:
            return
        lane_ids = self._get_lane_connector_ids(state)
        for light in state.traffic_lights:
            if light.lane_connector_id in lane_ids:
                light.status = self._state
                light.active_failures.add(self.__class__.__name__)


class Mislocalization(FailureGenerator):
    def __init__(
        self, offset: Union[Offset, Position], rotation: Angle = Angle(Constant(0))
    ):
        self._offset = offset
        self._rotation = rotation
        self._radius = 2
        self._reproject_on_lane = True
        self._need_init = True

    def reset(self):
        self._need_init = True

    def __call__(self, state: FaultyState):
        offset = self._offset.get()
        rotation = self._rotation.get()
        box = state.ego_state.car_footprint.oriented_box
        # Find good offset by reprojecting on the nearest lane line
        if isinstance(offset, Position):
            dx = offset.x - box._center.x
            dy = offset.y - box._center.y
        else:
            dx = np.cos(box.center.heading + offset.angle) * offset.distance
            dy = np.sin(box.center.heading + offset.angle) * offset.distance
        if self._need_init and self._reproject_on_lane:
            query_pt = Point2D(
                box._center.x + dx,
                box._center.y + dy,
            )
            lanes = state.map_api.get_proximal_map_objects(
                query_pt, self._radius, [SemanticMapLayer.LANE]
            )[SemanticMapLayer.LANE]
            if lanes:
                pt = Point(query_pt.x, query_pt.y)
                distances = [l.baseline_path.linestring.distance(pt) for l in lanes]
                idx = distances.index(min(distances))
                nearest_pt = nearest_points(lanes[idx].baseline_path.linestring, pt)[0]
                dx = nearest_pt.x - box._center.x
                dy = nearest_pt.y - box._center.y
        self._need_init = False
        # update ego state
        box._center.x += dx
        box._center.y += dy
        box._center.heading += rotation
        clear_ego_cache(state.ego_state)
        state.ego_state.active_failures.add(self.__class__.__name__)
        # update other agents
        detection_tracks = state.detections_tracks.tracked_objects
        for agent in detection_tracks.tracked_objects:
            agent._box._center.x += dx
            agent._box._center.y += dy
            agent._box._center.heading += rotation
            clear_agent_cache(agent)
            state.detections_tracks.failures_by_token[agent.track_token].add(
                self.__class__.__name__
            )


class Flickering(FailureGenerator):
    def __init__(
        self,
        failure: FailureGenerator,
        probability: float = 0.25,
        duration: float = 1.0,
    ):
        assert isinstance(
            failure, FailureGenerator
        ), "Failure must be a FailureGenerator"
        assert probability > 0 and probability < 1, "Probability must be in (0, 1)"
        assert duration > 0, "Duration must be positive"
        self._failure = failure
        self._probability = probability
        self._duration = duration  # in seconds
        self._last_activation = 0
        self._is_active = False

    def reset(self):
        self._failure.reset()
        self._last_activation = 0
        self._is_active = False

    def __call__(self, state: FaultyState):
        ts = state.ego_state.time_seconds
        if self._is_active:
            if ts > self._last_activation + self._duration:
                # Deactivate failure for at least one step
                self._is_active = False
            else:
                return self._failure(state)
        else:
            u = np.random.uniform()
            if u <= self._probability:
                self._is_active = True
                self._last_activation = ts
                return self._failure(state)


class AtTimestep(FailureGenerator):
    def __init__(
        self, failure: FailureGenerator, time_us: int = 0, stop_at: int = float("inf")
    ):
        assert isinstance(
            failure, FailureGenerator
        ), "Failure must be a FailureGenerator"
        assert time_us >= 0, "Timestep must be positive"
        self._failure = failure
        self._timestep = time_us
        self._stop_at = stop_at

    def reset(self):
        self._failure.reset()

    def __call__(self, state: FaultyState):
        if (
            state.ego_state.time_us >= self._timestep
            and state.ego_state.time_us < self._stop_at
        ):
            return self._failure(state)


COMMON_FAILURES = [
    TrafficLightMisdetection,
    MissedObstacle,
    GhostObstacle,
    Misdetection,
    Mislocalization,
]
TEMPORAL_FAILURES = [Flickering, AtTimestep]
