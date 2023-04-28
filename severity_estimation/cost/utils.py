import numpy as np

from severity_estimation.trajectron.datatypes import Node, Scene


def probability_over_scene(
    function,
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
    **kwargs
):
    if split_agents:
        results = {}
    else:
        first_dim = timesteps[1] - timesteps[0] + 1
        results = np.zeros((first_dim,))

    if agent_caches is None:
        agent_caches = {}
    if caches is None:
        caches = {}

    pred_nodes = None
    if predictions is not None:
        pred_nodes = predictions.keys()

    for agent_node in scene.nodes:
        if predictions is not None and agent_node not in pred_nodes:
            continue
        if agent_node.id == "ego":
            continue
        if agent_node not in agent_caches:
            agent_caches[agent_node] = None
        if (ego_node, agent_node) not in caches:
            caches[(ego_node, agent_node)] = None
        result = function(
            ego_node,
            agent_node,
            timesteps,
            predictions=predictions,
            ego_cache=ego_cache,
            agent_cache=agent_caches[agent_node],
            cache=caches[(ego_node, agent_node)],
            **kwargs
        )
        probs = np.count_nonzero(result, axis=0) / result.shape[0]
        if split_agents:
            # result[~valid_ind] = initial_extrema
            results[agent_node] = result
        else:
            valid_ind = ~np.isnan(probs)
            results = np.maximum(results[valid_ind], probs[valid_ind])
    return results


def extrema_over_scene(
    function,
    operation,
    initial_extrema,
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
    **kwargs
):
    if split_agents:
        extrema_result = {}
    else:
        first_dim = timesteps[1] - timesteps[0] + 1
        try:
            second_dim = predictions[list(predictions.keys())[0]].shape[1]
            extrema_result = np.zeros((second_dim, first_dim)) + initial_extrema
        except:
            extrema_result = np.zeros((1, first_dim)) + initial_extrema

    if agent_caches is None:
        agent_caches = {}
    if caches is None:
        caches = {}

    pred_nodes = None
    if predictions is not None:
        pred_nodes = predictions.keys()

    for agent_node in scene.nodes:
        if predictions is not None and agent_node not in pred_nodes:
            continue
        if agent_node.id == "ego":
            continue

        if agent_node not in agent_caches:
            agent_caches[agent_node] = None
        if (ego_node, agent_node) not in caches:
            caches[(ego_node, agent_node)] = None

        result = function(
            ego_node,
            agent_node,
            timesteps,
            predictions=predictions,
            ego_cache=ego_cache,
            agent_cache=agent_caches[agent_node],
            cache=caches[(ego_node, agent_node)],
            max_value=initial_extrema,
            **kwargs
        )

        if split_agents:
            # result[~valid_ind] = initial_extrema
            extrema_result[agent_node] = result
        else:
            valid_ind = ~np.isnan(result)
            extrema_result[valid_ind] = operation(extrema_result[valid_ind], result[valid_ind])

    return extrema_result
