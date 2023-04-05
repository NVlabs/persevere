from typing import Dict, List

import numpy as np
import torch

from severity_estimation.planner.query import Query
from severity_estimation.trajectron.datatypes import Node


def single_query(
    node: Node, query: Query.q, timesteps: np.ndarray, predictions=None, states=None
):
    if predictions is None:
        return query_node(node, query, timesteps, states)[str(query)]
    else:
        return query_prediction(node, query, timesteps, predictions, states)[str(query)]


def query_node(node: Node, queries: List[Query.q], timesteps: np.ndarray, states=None):
    queries, timesteps, states = _initialize(node, queries, timesteps, None, states)

    for query in queries:
        if str(query) in states.keys():
            continue

        query_name = str(query)
        sub_queries = query(1)

        if query in [Query.position, Query.velocity, Query.acceleration]:
            temp = node.get(timesteps, sub_queries)
            states[query_name] = temp

        elif query in [Query.heading, Query.velocity_norm, Query.acceleration_norm]:
            states[query_name] = node.get(timesteps, sub_queries)[:, 0]

        elif query in [Query.jerk_norm]:
            states = query_node(node, sub_queries, timesteps, states)
            states[query_name] = np.linalg.norm(states[str(sub_queries[0])], axis=-1)

        elif query == Query.rotation_matrix:
            states = query_node(node, sub_queries, timesteps, states)
            heading = states[str(sub_queries[0])]
            cs = np.cos(heading)
            ss = np.sin(heading)
            R = np.array([[cs, ss], [-ss, cs]])
            states[query_name] = R

        elif query in [Query.jerk, Query.heading_rate, Query.heading_acceleration]:
            timestep_request = np.array([timesteps[0] - 1, timesteps[1]])
            temp_states = query_node(node, sub_queries, timestep_request)
            state = temp_states[str(sub_queries[0])]
            dstate = (state[1:] - state[:-1]) / Query.dt
            states[query_name] = dstate

        elif query in [
            Query.rotated_position,
            Query.rotated_velocity,
            Query.rotated_acceleration,
            Query.rotated_jerk,
        ]:
            states = query_node(node, sub_queries, timesteps, states)
            R = states[str(sub_queries[0])]
            state = states[str(sub_queries[1])]
            rotated_state = np.einsum("ijt,tj->ti", R, state)
            states[query_name] = rotated_state

        elif query in [
            Query.lon_velocity,
            Query.lat_velocity,
            Query.lon_acceleration,
            Query.lat_acceleration,
            Query.lon_jerk,
            Query.lat_jerk,
        ]:
            states = query_node(node, sub_queries, timesteps, states)
            states[query_name] = states[str(sub_queries[0])][:, query(2)]

        else:
            raise NotImplementedError

    return states


def query_prediction(
    node: Node,
    queries: List[Query.q],
    timesteps: np.ndarray,
    predictions: Dict,
    states=None,
):
    queries, timesteps, states = _initialize(
        node, queries, timesteps, predictions, states
    )

    for query in queries:
        if str(query) in states.keys():
            continue

        query_name = str(query)
        sub_queries = query(1)

        if query in [Query.true_position, Query.true_velocity]:
            true_states = query_node(node, sub_queries, timesteps)
            states[query_name] = true_states[str(sub_queries[0])]

        elif query == Query.position:
            agent_pos = states["predictions"][
                0, :, 0 : timesteps[1] - timesteps[0] + 1, :
            ]
            states[query_name] = agent_pos

        elif query == Query.velocity:
            states = query_prediction(
                node, (Query.position,), timesteps, predictions, states
            )
            agent_pos = states["position"]
            n_repeat = agent_pos.shape[0]
            prev_timestep = np.array([timesteps[0] - 1])
            agent_posm1 = node.get(prev_timestep, sub_queries)
            agent_posm1 = np.tile(agent_posm1, (n_repeat, 1, 1))
            agent_pos1 = np.concatenate((agent_posm1, agent_pos[:, :-1]), axis=1)
            agent_pos2 = agent_pos
            agent_vel = (agent_pos2 - agent_pos1) / Query.dt
            states[query_name] = agent_vel

        # elif query == Query.velocity:
        #     prev_timestep = np.array([timesteps[0]-1])
        #     agent_posm1 = node.get(prev_timestep, sub_queries)[0]
        #     states = query_prediction(node, (Query.position,), timesteps, predictions, states)
        #     agent_pos = states['position']
        #     avg_agent_pos = np.mean(agent_pos, axis=0)
        #     agent_possm1 = np.array([agent_posm1, *avg_agent_pos[:-1,:]])
        #     agent_vel = (avg_agent_pos - agent_possm1)/Query.dt
        #     states[query_name] = agent_vel

        elif query == Query.weighted_mean:
            gmm = states["predictions"]
            weighted_x = np.sum(
                np.exp(gmm.log_pis.numpy())[0][0] * gmm.mus.numpy()[0, 0, :, :, 0],
                axis=-1,
            )
            weighted_y = np.sum(
                np.exp(gmm.log_pis.numpy())[0][0] * gmm.mus.numpy()[0, 0, :, :, 1],
                axis=-1,
            )
            weighted_mean = np.array([weighted_x, weighted_y]).T
            states[query_name] = weighted_mean

        elif query == Query.mean_distance_error:
            states = query_prediction(node, sub_queries, timesteps, predictions, states)
            true_pos = states[str(sub_queries[0])]
            weighted_mean = states[str(sub_queries[1])]
            states[query_name] = np.linalg.norm(true_pos - weighted_mean, axis=-1)

        elif query == Query.position_likelihood:
            states = query_prediction(node, sub_queries, timesteps, predictions, states)
            true_pos = states[str(sub_queries[0])]
            gmm = states["predictions"]
            weighted_mean = states[str(sub_queries[1])]
            likelihood = np.exp(
                gmm.log_prob(value=torch.tensor(true_pos))[0, 0].numpy()
            )
            likelihood[np.isnan(likelihood)] = np.inf
            states[query_name] = likelihood

        else:
            raise NotImplementedError

    return states


def query_node_and_node(
    ego_node: Node,
    agent_node: Node,
    queries: List[Query.q],
    timesteps: np.ndarray,
    ego_states=None,
    agent_states=None,
    states=None,
):
    queries, timesteps, ego_states, agent_states, states = _initialize_two(
        ego_node, agent_node, queries, timesteps, None, ego_states, agent_states, states
    )

    for query in queries:
        if str(query) in states.keys():
            continue
        query_name = str(query)
        sub_queries = query(1)
        ego_queries = query(2)
        agent_queries = query(3)

        ego_states = query_node(ego_node, ego_queries, timesteps, ego_states)
        agent_states = query_node(agent_node, agent_queries, timesteps, agent_states)
        states = query_node_and_node(
            ego_node,
            agent_node,
            sub_queries,
            timesteps,
            ego_states,
            agent_states,
            states,
        )

        if query in [
            Query.relative_position,
            Query.relative_velocity,
            Query.relative_acceleration,
            Query.relative_heading,
        ]:
            states[query_name] = (
                agent_states[str(agent_queries[0])] - ego_states[str(ego_queries[0])]
            )
        elif query in [
            Query.rotated_relative_position,
            Query.rotated_relative_velocity,
            Query.rotated_relative_acceleration,
        ]:
            r = ego_states[str(ego_queries[0])]
            rel_state = states[str(sub_queries[0])]
            rotated_state = np.einsum("ijt,tj->ti", r, rel_state)
            states[query_name] = rotated_state
        else:
            raise NotImplementedError

    return states


def query_node_and_prediction(
    ego_node: Node,
    agent_node: Node,
    queries: List[Query.q],
    timesteps: np.ndarray,
    predictions: Dict,
    ego_states=None,
    agent_states=None,
    states=None,
):
    queries, timesteps, ego_states, agent_states, states = _initialize_two(
        ego_node,
        agent_node,
        queries,
        timesteps,
        predictions,
        ego_states,
        agent_states,
        states,
    )

    for query in queries:
        if str(query) in states.keys():
            continue
        query_name = str(query)
        sub_queries = query(1)
        ego_queries = query(2)
        agent_queries = query(3)

        ego_states = query_node(ego_node, ego_queries, timesteps, ego_states)
        agent_states = query_prediction(
            agent_node, agent_queries, timesteps, predictions, agent_states
        )
        states = query_node_and_prediction(
            ego_node,
            agent_node,
            sub_queries,
            timesteps,
            predictions,
            ego_states,
            agent_states,
            states,
        )

        if query in [
            Query.relative_position,
            Query.relative_velocity,
            Query.relative_acceleration,
            Query.true_relative_position,
            Query.true_relative_velocity,
        ]:
            states[query_name] = (
                agent_states[str(agent_queries[0])] - ego_states[str(ego_queries[0])]
            )
        elif query in [
            Query.rotated_relative_position,
            Query.rotated_relative_velocity,
            Query.rotated_relative_acceleration,
        ]:
            r = ego_states[str(ego_queries[0])]
            rel_state = states[str(sub_queries[0])]
            rotated_state = np.einsum("ijt,atj->ati", r, rel_state)
            states[query_name] = rotated_state
        else:
            raise NotImplementedError

    return states


def _initialize_two(
    ego_node: Node,
    agent_node: Node,
    queries: List[Query.q],
    timesteps: np.ndarray,
    predictions=None,
    ego_states=None,
    agent_states=None,
    states=None,
):
    queries = _initialize_queries(queries)
    timesteps = _initialize_timesteps(timesteps)
    ego_states = _initialize_states(ego_node, timesteps, None, ego_states)
    agent_states = _initialize_states(agent_node, timesteps, predictions, agent_states)

    if states is None:  # if not cached state is provided, initialize one
        states = {}
        states["ego_node"] = ego_node
        states["agent_node"] = agent_node
        states["timesteps"] = timesteps
        if predictions is not None:
            states["predictions"] = predictions[agent_node]
        assert states["ego_node"] == ego_node, "Node in cache doesn't match request"
        assert states["agent_node"] == agent_node, "Node in cache doesn't match request"
        assert (
            states["timesteps"] == timesteps
        ).all(), "Timesteps in cache doesn't match request"

    return queries, timesteps, ego_states, agent_states, states


def _initialize(
    node: Node,
    queries: List[Query.q],
    timesteps: np.ndarray,
    predictions=None,
    states=None,
):
    queries = _initialize_queries(queries)
    timesteps = _initialize_timesteps(timesteps)
    states = _initialize_states(node, timesteps, predictions, states)
    return queries, timesteps, states


def _initialize_timesteps(timesteps: np.ndarray):
    if len(timesteps) == 1:
        timesteps = np.array([timesteps[0], timesteps[0]])
    return timesteps


def _initialize_queries(queries: List[Query.q]):
    if type(queries) == Query.q:  # hasattr(queries, '__len__'):
        queries = [queries]
    return queries


def _initialize_states(
    node: Node, timesteps: np.ndarray, predictions=None, states=None
):
    if states is None:  # if not cached state is provided, initialize one
        states = {}
        states["node"] = node
        states["timesteps"] = timesteps
        if predictions is not None:
            states["predictions"] = predictions[node]
    assert states["node"] == node, "Node in cache doesn't match request"
    assert (
        states["timesteps"] == timesteps
    ).all(), "Timesteps in cache doesn't match request"
    return states
