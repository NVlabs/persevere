import numpy as np

from severity_estimation.cost.cost_functions import max_distance_to_agent_cost
from severity_estimation.planner.query import Query
from severity_estimation.planner.query_utils import query_node, query_prediction


class CostManager:

    # momentum shaped distance
    def compute_noplan_distance_to_agent_cost(self, ego_node, timesteps, predictions=None, split_agents=False):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)

        cost = {}
        agents = predictions.keys()
        for agent in agents:
            if agent.id == "ego":
                continue
            # agent_states = query_prediction(predictions[agent], queries, timesteps)
            # cost[agent] = max_distance_to_agent_cost(ego_states, agent_states)
            cost[agent] = 0
        return cost

    def _distance_to_agent(ego, agent, timesteps):
        if agent.type.name == "VEHICLE":
            # lower means we care more about it
            vi_epsilon, vo_epsilon = 0.5, 0.5
        else:
            vi_epsilon, vo_epsilon = 1, 1
        queries = [Query.rotated_relative_position, Query.rotated_relative_velocity]

        returns = _distance_to_agent_cost(rotated_relative_position, rotated_relative_velocity, vi_epsilon, vo_epsilon)
