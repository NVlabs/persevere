from copy import deepcopy
from typing import Any, Generator, Optional, Tuple, Type

from loguru import logger
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation, Sensors
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from severity_estimation.fault_injection.failures import (
    FaultInjectionManager,
    FaultyDetectionsTracks,
    FaultyEgoState,
    FaultyState,
    FaultyTrafficLightStatusData,
)
from severity_estimation.fault_injection.simulation_history_buffer import SimulationHistoryBuffer


class Simulator(Simulation):
    def __init__(
        self,
        simulation_setup: SimulationSetup,
        callback: Optional[AbstractCallback] = None,
        simulation_history_buffer_duration: float = 2,
        faults: FaultInjectionManager = FaultInjectionManager([]),
    ):
        super().__init__(simulation_setup, callback, simulation_history_buffer_duration)
        self._faults = faults

    def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (
            self._setup,
            self._callback,
            self._simulation_history_buffer_duration,
            self._faults,
        )

    def reset(self) -> None:
        """
        Reset all internal states of simulation.
        """
        # Clear created log
        self._history.reset()
        # Reset all simulation internal members
        self._setup.reset()
        # Clear history buffer
        self._history_buffer = None
        # Restart simulation
        self._is_simulation_running = True
        # Restart failure injector
        self._faults.reset()

    def _preprocess_observation(
        self,
        gt_ego_states: EgoState,
        gt_tracks: DetectionsTracks,
        traffic_lights: Optional[TrafficLightStatusData] = None,
    ) -> FaultyState:
        if traffic_lights is not None:
            faulty_traffic_lights = [
                FaultyTrafficLightStatusData.from_traffic_light_status_data(tl) for tl in traffic_lights
            ]
        else:
            faulty_traffic_lights = None
        state = FaultyState(
            ego_state=FaultyEgoState.from_ego_state(gt_ego_states),
            detections_tracks=FaultyDetectionsTracks.from_detections_tracks(gt_tracks),
            traffic_lights=faulty_traffic_lights,
            map_api=self.scenario.map_api,
        )
        return self._faults.apply(state)

    def _initialize_history_buffer(
        self,
        buffer_size: int,
        scenario: AbstractScenario,
        observation_type: Type[Observation],
    ) -> None:
        buffer_duration = buffer_size * scenario.database_interval

        if observation_type == DetectionsTracks:
            observation_getter = scenario.get_past_tracked_objects
        elif observation_type == Sensors:
            observation_getter = scenario.get_past_sensors
        else:
            raise ValueError(f"No matching observation type for {observation_type} for history!")
        past_observation = list(observation_getter(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size))
        past_ego_states = list(
            scenario.get_ego_past_trajectory(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size)
        )
        traffic_lights = list(
            scenario.get_past_traffic_light_status(iteration=0, time_horizon=buffer_duration, num_samples=buffer_size)
        )
        faulty_states = [
            self._preprocess_observation(ego, obs, tl)
            for ego, obs, tl in zip(past_ego_states, past_observation, traffic_lights)
        ]

        return SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=[s.ego_state for s in faulty_states],
            observations=[s.detections_tracks for s in faulty_states],
            traffic_lights=[s.traffic_lights for s in faulty_states],
            # sample_interval=scenario.database_interval,
        )

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario
        self._history_buffer = self._initialize_history_buffer(
            self._history_buffer_size,
            self._scenario,
            self._observations.observation_type(),
        )

        # Initialize observations
        self._observations.initialize()

        # Initialize controller
        # self._ego_controller.initialize()

        # Extract traffic light status data
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(
            self._time_controller.get_iteration().index
        )

        # Add the current state into the history buffer
        faulty_state = self._preprocess_observation(
            self._ego_controller.get_state(),
            self._observations.get_observation(),
            traffic_light_data,
        )
        self._history_buffer.append(
            faulty_state.ego_state,
            faulty_state.detections_tracks,
            faulty_state.traffic_lights,
        )

        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            mission_goal=self._scenario.get_mission_goal(),
            map_api=self._scenario.map_api,
        )

    def _get_traffic_light_status_data(self, state) -> Generator[TrafficLightStatusData, None, None]:
        for light in state.traffic_lights:
            yield TrafficLightStatusData(
                status=light.status,
                lane_connector_id=light.lane_connector_id,
                timestamp=light.timestamp,
            )

    def get_planner_input(self) -> PlannerInput:
        """
        Construct inputs to the planner for the current iteration step
        :return Inputs to the planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, stepping can not be performed!")

        # Extract current state
        iteration = self._time_controller.get_iteration()

        # Extract traffic light status data
        traffic_light_data_ = self._scenario.get_traffic_light_status_at_iteration(iteration.index)
        # traffic_light_data = self._get_traffic_light_status_data()
        logger.trace(f"Executing {iteration.index}!")
        return PlannerInput(
            iteration=iteration,
            history=self._history_buffer,
            traffic_light_data=traffic_light_data_,
        )

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))
        faulty_state = self._preprocess_observation(
            self._ego_controller.get_state(),
            self._observations.get_observation(),
            traffic_light_status,
        )

        # Add new sample to history
        logger.trace(f"Adding to history: {iteration.index}")
        self._history.add_sample(
            SimulationHistorySample(
                iteration,
                faulty_state.ego_state,
                trajectory,
                faulty_state.detections_tracks,
                faulty_state.traffic_lights,
            )
        )

        # Propagate state to next iteration
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, faulty_state.ego_state, trajectory)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        self._history_buffer.append(
            faulty_state.ego_state,
            faulty_state.detections_tracks,
            faulty_state.traffic_lights,
        )
