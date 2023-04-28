from random import choices
from typing import Optional

import numpy as np
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def random_token():
    charset = [str(i) for i in range(10)] + ["a", "b", "c", "d", "e", "f"]
    return "".join(choices(charset, k=16))


def find_agent_index(tracked_objects: TrackedObjects, track_token: str) -> Optional[int]:
    for idx, obj in enumerate(tracked_objects):
        if obj.track_token == track_token:
            return idx
    return None


def find_agent(tracked_objects: TrackedObjects, track_token: str) -> Optional[TrackedObject]:
    idx = find_agent_index(tracked_objects, track_token)
    if idx is not None:
        return tracked_objects[idx]
    return None


def remove_agent_from(
    tracked_objects: TrackedObjects,
    track_token: str,
) -> bool:
    idx = find_agent_index(tracked_objects, track_token)
    if idx is not None:
        del tracked_objects[idx]
        return True
    return False


def mutate_agent(agent: TrackedObjects, offset, shape_ratio, rotation, velocity_magnitude, category):
    cos_rot, sin_rot = np.cos(rotation), np.sin(rotation)
    agent._box._center.x += np.cos(agent._box.center.heading + offset.angle) * offset.distance
    agent._box._center.y += np.sin(agent._box.center.heading + offset.angle) * offset.distance
    agent._box._length *= shape_ratio.length
    agent._box._width *= shape_ratio.width
    agent._box._center.heading = wrap_pi(agent._box._center.heading + rotation)
    agent._velocity.x = velocity_magnitude * (agent._velocity.x * cos_rot - agent._velocity.y * sin_rot)
    agent._velocity.y = velocity_magnitude * (agent._velocity.x * sin_rot + agent._velocity.y * cos_rot)
    metadata = SceneObjectMetadata(
        timestamp_us=agent.metadata.timestamp_us,
        token=agent.token,
        track_id=None,
        track_token=agent.track_token,
        category_name=agent.metadata.category_name if category is None else category,
    )
    agent._metadata = metadata
    clear_agent_cache(agent)


def clear_agent_cache(agent):
    # invalidate cached property
    if hasattr(agent._box, "geometry"):
        del agent._box.geometry
    if hasattr(agent._box, "rear_axle"):
        del agent._box.car_footprint.rear_axle
    agent._box.corner.cache_clear()


def clear_ego_cache(ego_state):
    # Car Footprint
    if hasattr(ego_state.car_footprint, "geometry"):
        del ego_state.car_footprint.geometry
    if hasattr(ego_state.car_footprint, "rear_axle"):
        del ego_state.car_footprint.rear_axle
    ego_state.car_footprint.corner.cache_clear()
    # Dynamic Car State
    if hasattr(ego_state.dynamic_car_state, "center_velocity_2d"):
        del ego_state.dynamic_car_state.center_velocity_2d
    if hasattr(ego_state.dynamic_car_state, "center_acceleration_2d"):
        del ego_state.dynamic_car_state.center_acceleration_2d
    if hasattr(ego_state.dynamic_car_state, "speed"):
        del ego_state.dynamic_car_state.speed
    if hasattr(ego_state.dynamic_car_state, "acceleration"):
        del ego_state.dynamic_car_state.acceleration
    # Ego State
    if hasattr(ego_state, "waypoint"):
        del ego_state.waypoint
    if hasattr(ego_state, "agent"):
        del ego_state.agent
