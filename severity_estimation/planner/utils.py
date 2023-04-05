import json
import random
import string
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.label.utils import global2local
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from severity_estimation.trajectron.datatypes import (
    DoubleHeaderNumpyArray,
    Environment,
    ModelRegistrar,
    Node,
    NodeType,
    Scene,
    Trajectron,
)
from severity_estimation.utils.containers import DotDict
from severity_estimation.trajectron.online import OnlineTrajectron


def _x_radius(scene: Scene):
    return (scene.x_max - scene.x_min) / 2


def _y_radius(scene: Scene):
    return (scene.y_max - scene.y_min) / 2


def position_shift(scene: Scene):
    return [scene.x_min + _x_radius(scene), scene.y_min + _y_radius(scene)]


def generate_token(n=32):
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(n)
    )


def context(cfg):
    vehicle = NodeType(name="VEHICLE", value=1)
    pedestrian = NodeType(name="PEDESTRIAN", value=2)
    model_path = Path(cfg.trajectron.model).resolve()
    labelmap = global2local.copy()
    labelmap["vehicle"] = vehicle
    labelmap["pedestrian"] = pedestrian
    return DotDict(
        {
            "NodeTypes": [
                "VEHICLE",
                "PEDESTRIAN",
            ],
            "Pedestrian": pedestrian,
            "Vehicle": vehicle,
            "AttentionRadius": {
                (pedestrian, pedestrian): 10.0,
                (pedestrian, vehicle): 20.0,
                (vehicle, pedestrian): 20.0,
                (vehicle, vehicle): 30.0,
            },
            "Header": [
                ("position", "x"),
                ("position", "y"),
                ("velocity", "x"),
                ("velocity", "y"),
                ("acceleration", "x"),
                ("acceleration", "y"),
                ("heading", "x"),
                ("heading", "y"),
                ("heading", "°"),
                ("heading", "d°"),
                ("velocity", "norm"),
                ("acceleration", "norm"),
            ],
            "Standardization": {
                pedestrian: {
                    "position": {
                        "x": {"mean": 0, "std": 1},
                        "y": {"mean": 0, "std": 1},
                    },
                    "velocity": {
                        "x": {"mean": 0, "std": 2},
                        "y": {"mean": 0, "std": 2},
                    },
                    "acceleration": {
                        "x": {"mean": 0, "std": 1},
                        "y": {"mean": 0, "std": 1},
                    },
                },
                vehicle: {
                    "position": {
                        "x": {"mean": 0, "std": 80},
                        "y": {"mean": 0, "std": 80},
                    },
                    "velocity": {
                        "x": {"mean": 0, "std": 15},
                        "y": {"mean": 0, "std": 15},
                        "norm": {"mean": 0, "std": 15},
                    },
                    "acceleration": {
                        "x": {"mean": 0, "std": 4},
                        "y": {"mean": 0, "std": 4},
                        "norm": {"mean": 0, "std": 4},
                    },
                    "heading": {
                        "x": {"mean": 0, "std": 1},
                        "y": {"mean": 0, "std": 1},
                        "°": {"mean": 0, "std": 3.141592653589793},
                        "d°": {"mean": 0, "std": 1},
                    },
                },
            },
            "dt": cfg.trajectron.dt,
            "ModelPath": model_path,
            "Labelmap": labelmap,
        }
    )


def build_trajectron_env(ctx):
    timesteps = 0
    placeholder_scene = [Scene(timesteps=timesteps, dt=ctx.dt)]
    env = Environment(
        scenes=placeholder_scene,
        node_type_list=ctx.NodeTypes,
        standardization=ctx.Standardization,
        attention_radius=ctx.AttentionRadius,
        robot_type=ctx.Vehicle,
        dt=ctx.dt,
    )
    return env


def load_trajectron_model(ctx, ts=12, device="cuda"):
    eval_env = build_trajectron_env(ctx)
    with open(ctx.ModelPath / "config.json", "r") as config_json:
        trajectron_hyperparams = json.load(config_json)
    model_registrar = ModelRegistrar(ctx.ModelPath, device)
    model_registrar.load_models(ts)
    trajectron_hyperparams["map_enc_dropout"] = 0.0
    if "incl_robot_node" not in trajectron_hyperparams:
        trajectron_hyperparams["incl_robot_node"] = False
    stg = Trajectron(model_registrar, trajectron_hyperparams, None, device)
    stg.set_environment(eval_env)
    stg.set_annealing_params()
    return eval_env, stg


def create_online_env(env, hyperparams, scene_idx, init_timestep):
    test_scene = env.scenes[scene_idx]

    online_scene = Scene(
        timesteps=init_timestep + 1, map=test_scene.map, dt=test_scene.dt
    )
    online_scene.nodes = test_scene.get_nodes_clipped_at_time(
        timesteps=np.arange(
            init_timestep - hyperparams["maximum_history_length"], init_timestep + 1
        ),
        state=hyperparams["state"],
    )
    online_scene.robot = test_scene.robot
    online_scene.calculate_scene_graph(
        attention_radius=env.attention_radius,
        edge_addition_filter=hyperparams["edge_addition_filter"],
        edge_removal_filter=hyperparams["edge_removal_filter"],
    )
    return Environment(
        node_type_list=env.node_type_list,
        standardization=env.standardization,
        scenes=[online_scene],
        attention_radius=env.attention_radius,
        robot_type=env.robot_type,
    )


def load_online_trajectron_model(ctx, ts=12, device="cuda"):
    scene_idx, init_timestep = 0, 1
    eval_env = build_trajectron_env(ctx)
    with open(ctx.ModelPath / "config.json", "r") as config_json:
        hyperparams = json.load(config_json)
    model_registrar = ModelRegistrar(ctx.ModelPath, device)
    model_registrar.load_models(ts)
    hyperparams["map_enc_dropout"] = 0.0
    if "incl_robot_node" not in hyperparams:
        hyperparams["incl_robot_node"] = False
    online_env: Environment = create_online_env(
        eval_env, hyperparams, scene_idx, init_timestep
    )
    stg = OnlineTrajectron(model_registrar, hyperparams, device)
    stg.set_environment(online_env, init_timestep)
    return eval_env, hyperparams, stg


def add_observations_to_scene(
    ctx: DotDict, scene: Scene, ego_state: EgoState, tracked_objects
):
    # scene_radius = (scene.y_max - scene.y_min)/2
    add_agent_to_scene(ctx, scene, ego_state, is_ego=True)
    for obj in tracked_objects:
        if obj.metadata.category_name in {"vehicle", "pedestrian"}:
            # d = np.sqrt(
            #     (obj.center.x - ego_state.gt_center.x) ** 2
            #     + (obj.center.y - ego_state.gt_center.y) ** 2
            # )
            # if d < scene_radius:
            add_agent_to_scene(ctx, scene, obj, is_ego=False)
            # else:
                # remove_object_from_scene(scene, obj)
    scene.timesteps += 1


def remove_object_from_scene(scene: Scene, agent_data):
    node = find_node(scene, agent_data.track_token, agent_data)
    if node is not None:
        scene.nodes.remove(node)


def add_agent_to_scene(ctx: DotDict, scene: Scene, agent_data: Box3D, is_ego=False):
    node = find_node(scene, "ego" if is_ego else agent_data.track_token, agent_data)

    if node is None or node == -1:
        if node == -1:
            # print("NEW TOKEN IS GENERATED")
            token = generate_token()
        else:
            token = "ego" if is_ego else agent_data.track_token
        data = DoubleHeaderNumpyArray(np.array([]), ctx.Header)
        node_type = (
            ctx.Vehicle if is_ego else ctx.Labelmap[agent_data.metadata.category_name]
        )
        node = Node(
            node_type=node_type,
            node_id=token,
            data=data,
            first_timestep=scene.timesteps,
            is_robot=is_ego,
        )
        scene.nodes.append(node)
    data = (
        convert_ego_state_to_node(scene, agent_data, node)
        if is_ego
        else convert_box_to_node(scene, agent_data, node)
    )
    # TODO[antonap]: at the moment if a detection is intermittent, the old history is disregarded
    # lag = scene.timesteps - (node.first_timestep + len(node.data.data))
    # if lag == 0 it is intermittent detection
    if len(node.data.data.shape) == 1:
        # New or intermittent detection
        node.data.data = np.array([data])
    else:
        node.data.data = np.append(node.data.data, [data], axis=0)
    node._last_timestep = node.first_timestep + node.timesteps - 1
    # if node.data.data.shape[0] > 4:
    #     node.data.data = node.data.data[-4:, :]
    #     node.first_timestep = node._last_timestep - 3
    if is_ego:
        node.length = agent_data.agent.box.length
        node.width = agent_data.agent.box.width
        scene.robot = node
    else:
        node.length = agent_data.box.length
        node.width = agent_data.box.width


def find_node(scene: Scene, token: str, agent_data=None):
    nodes = scene.nodes
    if token is None and agent_data is not None:  # simple tracking for reactive agents
        assert (
            token is not None
        ), "I added tokens to reactive agents, this should not be accessed."
        closest_feasible_node = -1
        min_track_metric = np.inf
        for node in nodes:
            lag = scene.timesteps - (node.first_timestep + len(node.data.data))
            dist = np.linalg.norm(
                agent_data.center[:-1]
                - (node.data.data[-1, :2] + position_shift(scene))
            )
            orientation_diff = abs(
                agent_data.orientation.yaw_pitch_roll[0] - node.data.data[-1, 8]
            )
            if lag == 0 and dist < 10 and orientation_diff < np.pi / 6:
                track_metric = dist / 5 + orientation_diff
                if track_metric < min_track_metric:
                    min_track_metric = track_metric
                    closest_feasible_node = node
        return closest_feasible_node

    for node in nodes:
        if token == node.id:
            return node
    return None


def convert_box_to_node(scene: Scene, box: Box3D, node: Node):
    dt = scene.dt
    x_position = box.center.x - scene.x_min - _x_radius(scene)
    y_position = box.center.y - scene.y_min - _y_radius(scene)

    heading = box.center.heading
    if heading > np.pi:
        heading -= np.pi

    if len(node.data.data) == 0:
        x_velocity = 0
        y_velocity = 0
        d_heading = 0
        norm_velocity = 0
        x_heading = 0
        y_heading = 0
    else:
        x_velocity = (x_position - node.data.data[-1, 0]) / dt
        y_velocity = (y_position - node.data.data[-1, 1]) / dt
        d_heading = (heading - node.data.data[-1, 8]) / dt
        norm_velocity = np.sqrt(x_velocity**2 + y_velocity**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_heading = x_velocity / norm_velocity
            y_heading = y_velocity / norm_velocity
    if len(node.data.data) <= 1:
        x_acceleration = 0
        y_acceleration = 0
        norm_acceleration = 0
    else:
        x_acceleration = (x_velocity - node.data.data[-1, 2]) / dt
        y_acceleration = (y_velocity - node.data.data[-1, 3]) / dt
        norm_acceleration = np.sqrt(x_acceleration**2 + y_acceleration**2)

    return np.array(
        [
            x_position,
            y_position,
            x_velocity,
            y_velocity,
            x_acceleration,
            y_acceleration,
            x_heading,
            y_heading,
            heading,
            d_heading,
            norm_velocity,
            norm_acceleration,
        ]
    )


def convert_ego_state_to_node(
    scene: Scene, ego_state: EgoState, ego_node: Optional[Node] = None
):
    x_position = ego_state.center.x - scene.x_min - _x_radius(scene)
    y_position = ego_state.center.y - scene.y_min - _y_radius(scene)
    heading = ego_state.center.heading
    cc = np.cos(heading)
    ss = np.sin(heading)
    dt = scene.dt

    lon_vel = ego_state.dynamic_car_state.center_velocity_2d.x
    lat_vel = ego_state.dynamic_car_state.center_velocity_2d.y
    x_velocity = lon_vel * cc - lat_vel * ss
    y_velocity = lon_vel * ss + lat_vel * cc

    if abs(x_velocity) < 1e-6 and abs(y_velocity) < 1e-6 and ego_node is not None:
        x_velocity = (x_position - ego_node.data.data[-1, 0]) / dt
        y_velocity = (y_position - ego_node.data.data[-1, 1]) / dt
        x_acceleration = (x_velocity - ego_node.data.data[-1, 2]) / dt
        y_acceleration = (y_velocity - ego_node.data.data[-1, 3]) / dt
        d_heading = (heading - ego_node.data.data[-1, 8]) / dt
        norm_velocity = np.sqrt(x_velocity**2 + y_velocity**2)
        norm_acceleration = np.sqrt(x_acceleration**2 + y_acceleration**2)
    else:
        lon_acc = ego_state.dynamic_car_state.center_acceleration_2d.x
        lat_acc = ego_state.dynamic_car_state.center_acceleration_2d.y
        x_acceleration = lon_acc * cc - lat_acc * ss
        y_acceleration = lon_acc * ss + lat_acc * cc
        d_heading = ego_state.dynamic_car_state.angular_velocity
        norm_velocity = ego_state.dynamic_car_state.speed
        norm_acceleration = ego_state.dynamic_car_state.acceleration

    x_heading = x_velocity / norm_velocity
    y_heading = y_velocity / norm_velocity

    return np.array(
        [
            x_position,
            y_position,
            x_velocity,
            y_velocity,
            x_acceleration,
            y_acceleration,
            x_heading,
            y_heading,
            heading,
            d_heading,
            norm_velocity,
            norm_acceleration,
        ]
    )


def convert_trajectory_to_node(
    ctx,
    trajectory: AbstractTrajectory,
    scene: Scene,
    first_timestep: int = 0,
) -> Node:
    ts = [
        TimePoint(t)
        for t in range(
            trajectory.start_time.time_us,
            trajectory.end_time.time_us,
            int(scene.dt * 1e6),
        )
    ]
    node_data = [
        convert_ego_state_to_node(scene, trajectory.get_state_at_time(t)) for t in ts
    ]
    node = Node(
        node_type=ctx.Vehicle,
        node_id="ego",
        data=DoubleHeaderNumpyArray(np.array(node_data), ctx.Header),
        first_timestep=first_timestep,
        is_robot=True,
    )
    return node


def serialize_ego_trajectory(scenario: AbstractScenario, subsample_ratio=0.1):
    scenario_length = scenario.get_number_of_iterations()
    num_time_steps = round(scenario_length * subsample_ratio)
    ego_traj = np.zeros((num_time_steps, 3))
    for i in range(num_time_steps):
        center = scenario.get_ego_state_at_iteration(i).center
        ego_traj[i] = [center.x, center.y, center.heading]
    return ego_traj
