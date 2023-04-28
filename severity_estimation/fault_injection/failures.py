import abc
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Set

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.common.utils.split_state import SplitState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


@dataclass
class FaultyDetectionsTracks(DetectionsTracks):
    """
    Output of the perception system, i.e. tracks.
    """

    gt_tracked_objects: TrackedObjects
    failures_by_token: Dict[str, Set[str]]

    @staticmethod
    def from_detections_tracks(detections_tracks: DetectionsTracks):
        return FaultyDetectionsTracks(
            tracked_objects=detections_tracks.tracked_objects,
            gt_tracked_objects=deepcopy(detections_tracks.tracked_objects),
            failures_by_token=defaultdict(set),
        )

    @property
    def has_failures(self) -> bool:
        return bool(self.failures_by_token)

    @property
    def active_failures(self):
        return set.union(*self.failures_by_token.values()) if self.has_failures else set()


class FaultyEgoState(EgoState):
    def __init__(
        self,
        car_footprint: CarFootprint,
        gt_car_footprint: CarFootprint,
        dynamic_car_state: DynamicCarState,
        tire_steering_angle: float,
        is_in_auto_mode: bool,
        time_point: TimePoint,
        active_failures: Optional[Set[str]] = set(),
    ):
        super().__init__(
            car_footprint,
            dynamic_car_state,
            tire_steering_angle,
            is_in_auto_mode,
            time_point,
        )
        self._gt_car_footprint = gt_car_footprint
        self.active_failures = active_failures

    @staticmethod
    def from_ego_state(ego_state: EgoState):
        return FaultyEgoState(
            car_footprint=ego_state.car_footprint,
            gt_car_footprint=deepcopy(ego_state.car_footprint),
            dynamic_car_state=ego_state.dynamic_car_state,
            tire_steering_angle=ego_state.tire_steering_angle,
            is_in_auto_mode=ego_state.is_in_auto_mode,
            time_point=ego_state.time_point,
            active_failures=set(),
        )

    def gt_ego_state(self):
        return EgoState(
            self.gt_car_footprint,
            self.dynamic_car_state,
            self.tire_steering_angle,
            self.is_in_auto_mode,
            self.time_point,
        )

    @property
    def gt_car_footprint(self) -> CarFootprint:
        return self._gt_car_footprint

    @property
    def gt_center(self) -> StateSE2:
        """
        Getter for Ego's center pose (center of mass)
        :return: Ego's center pose
        """
        return self._gt_car_footprint.oriented_box.center

    @cached_property
    def gt_agent(self) -> AgentState:
        """
        Casts the EgoState to an Agent object.
        :return: An Agent object with the parameters of EgoState
        """
        return AgentState(
            metadata=self.scene_object_metadata,
            tracked_object_type=TrackedObjectType.EGO,
            oriented_box=self.gt_car_footprint.oriented_box,
            velocity=self.dynamic_car_state.center_velocity_2d,
        )

    @cached_property
    def gt_waypoint(self) -> Waypoint:
        """
        :return: waypoint corresponding to this ego state
        """
        return Waypoint(
            time_point=self.time_point,
            oriented_box=self.gt_car_footprint,
            velocity=self.dynamic_car_state.rear_axle_velocity_2d,
        )

    def to_gt_split_state(self) -> SplitState:
        """Inherited, see superclass."""
        linear_states = [
            self.time_us,
            self.rear_axle.x,
            self.rear_axle.y,
            self.dynamic_car_state.rear_axle_velocity_2d.x,
            self.dynamic_car_state.rear_axle_velocity_2d.y,
            self.dynamic_car_state.rear_axle_acceleration_2d.x,
            self.dynamic_car_state.rear_axle_acceleration_2d.y,
            self.tire_steering_angle,
        ]
        angular_states = [self.rear_axle.heading]
        fixed_state = [self.gt_car_footprint.vehicle_parameters]

        return SplitState(linear_states, angular_states, fixed_state)

    @property
    def has_failures(self) -> bool:
        return bool(self.active_failures)


@dataclass
class FaultyTrafficLightStatusData(TrafficLightStatusData):
    gt_status: TrafficLightStatusType
    active_failures: Set[str]

    @staticmethod
    def from_traffic_light_status_data(traffic_light: TrafficLightStatusData):
        return FaultyTrafficLightStatusData(
            traffic_light.status,
            traffic_light.lane_connector_id,
            traffic_light.timestamp,
            traffic_light.status,
            set(),
        )

    @property
    def has_failures(self) -> bool:
        return bool(self.active_failures)

    def serialize(self) -> Dict[str, Any]:
        """Serialize traffic light status."""
        return {
            "status": self.status.serialize(),
            "gt_status": self.gt_status.serialize(),
            "lane_connector_id": self.lane_connector_id,
            "timestamp": self.timestamp,
            "active_failures": list(self.active_failures),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """Deserialize a dict of data to this class."""
        return TrafficLightStatusData(
            status=TrafficLightStatusType.deserialize(data["status"]),
            lane_connector_id=data["lane_connector_id"],
            timestamp=data["timestamp"],
            gt_status=data["gt_status"],
            active_failures=set(data["active_failures"]),
        )


@dataclass
class FaultyState:
    ego_state: FaultyEgoState
    detections_tracks: FaultyDetectionsTracks
    traffic_lights: Optional[List[FaultyTrafficLightStatusData]] = None
    map_api: Optional[NuPlanMap] = None

    @property
    def has_failures(self) -> bool:
        return (
            self.detections_tracks.has_failures
            or self.ego_state.has_failures
            or any(l.has_failures for l in self.traffic_lights)
        )

    def active_failures(self):
        return set.union(
            self.detections_tracks.active_failures,
            self.ego_state.active_failures,
            *[l.active_failures for l in self.traffic_lights]
        )


class FailureGenerator(abc.ABC):
    """Class to generate a perception failures in a nuPlan scenario."""

    OBJECT_TYPES = {
        "vehicle": TrackedObjectType.VEHICLE,
        "pedestrian": TrackedObjectType.PEDESTRIAN,
        "bicycle": TrackedObjectType.BICYCLE,
        "genericobject": TrackedObjectType.GENERIC_OBJECT,
        "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
        "barrier": TrackedObjectType.BARRIER,
    }

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """Generate a new failure"""

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, state: FaultyState):
        """Directly modifies the observations injecting the failure.
        Returns the set of token to wich the failure is active."""
        pass


class FaultInjectionManager:
    def __init__(self, generators: List[FailureGenerator] = []):
        self._generators = generators

    @property
    def failures(self) -> Set[str]:
        return {type(g).__name__ for g in self._generators}

    def reset(self):
        for g in self._generators:
            g.reset()

    def clear(self):
        self._generators = []

    def apply(self, state: FaultyState) -> FaultyState:
        for g in self._generators:
            g(state)
        return state
