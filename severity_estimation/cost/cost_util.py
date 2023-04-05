import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from severity_estimation.cost.cost_functions import (
    collision_probability,
    max_distance_to_agent_cost,
    max_time_to_collision_cost,
)
from severity_estimation.planner.query import Query
from severity_estimation.planner.query_utils import query_node


def wrap(angle):
    # wrap to [-pi, pi]
    return (angle + np.pi) % (2 * np.pi) - np.pi


def magnitude(x, y):
    return np.sqrt(x**2 + y**2)


class CostUtil:
    @staticmethod
    def compute_momentum_shape_distance(
        scene, ego_node, timesteps, prediction_dict=None, split_agents=True
    ):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)
        costs = max_distance_to_agent_cost(
            ego_node,
            scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=split_agents,
        )
        return costs

    @staticmethod
    def compute_collision_probability(scene, ego_node, timesteps, prediction_dict=None):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)
        costs = collision_probability(
            ego_node=ego_node,
            scene=scene,
            timesteps=timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
        )
        return costs

    @staticmethod
    def compute_time_to_collision_cost(
        scene, ego_node, timesteps, prediction_dict=None, split_agents=True
    ):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)
        costs = max_time_to_collision_cost(
            ego_node,
            scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=split_agents,
        )
        return costs

    @staticmethod
    def compute_hj_reachability_cost(ego_state, tracked_objects, scene_radius, hj):
        # tracked_obj_types = set(tracked_object_types.values())
        tracked_object_types = {
            "vehicles": TrackedObjectType.VEHICLE,
            # 'pedestrians': TrackedObjectType.PEDESTRIAN,
            # 'bicycles': TrackedObjectType.BICYCLE,
            # 'genericobjects': TrackedObjectType.GENERIC_OBJECT,
            # 'traffic_cone': TrackedObjectType.TRAFFIC_CONE,
            # 'barrier': TrackedObjectType.BARRIER,
            # 'czone_sign': TrackedObjectType.CZONE_SIGN,
        }
        values = list()
        for (
            tracked_object_type_name,
            tracked_object_type,
        ) in tracked_object_types.items():
            for obj in tracked_objects.get_tracked_objects_of_type(tracked_object_type):
                ego_velocity = ego_state.dynamic_car_state.center_velocity_2d
                ego_vel = magnitude(ego_velocity.x, ego_velocity.y)
                x = obj.box.center.x - ego_state.center.x
                y = obj.box.center.y - ego_state.center.y
                if magnitude(x, y) > scene_radius:
                    continue
                rel_heading = wrap(obj.box.center.heading - ego_state.center.heading)
                obj_vel = magnitude(obj.velocity.x, obj.velocity.y)
                rel_state = [x, y, rel_heading, ego_vel, obj_vel]
                values.append(hj(rel_state))
        return values
