from collections import defaultdict
from typing import Dict, List

from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.idm_agents import (
    IDMAgents as NuPlanIDMAgents,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)


class IDMAgents(NuPlanIDMAgents):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_observation(
        self,
        iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
        tspan = next_iteration.time_s - iteration.time_s
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(
            self.current_iteration
        )

        # Extract traffic light data into Dict[traffic_light_status, lane_connector_ids]
        traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(
            list
        )

        for data in traffic_light_data:
            traffic_light_status[data.status].append(str(data.lane_connector_id))

        ego_state, _ = history.current_state
        self._get_idm_agent_manager().propagate_agents(
            ego_state.gt_ego_state(),
            tspan,
            self.current_iteration,
            traffic_light_status,
            self._get_open_loop_track_objects(self.current_iteration),
            self._radius,
        )
