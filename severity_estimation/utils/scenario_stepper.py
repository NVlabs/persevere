from typing import Optional, cast

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)


class ScenarioStepper(StepSimulationTimeController):
    """
    Class handling simulation time and completion.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        max_duration: Optional[int] = None,
        start_time: int = 0,
    ):
        """
        Initialize simulation control.
        """
        self.scenario = scenario
        scenario_nr_iterations = cast(int, self.scenario.get_number_of_iterations())
        if max_duration is not None:
            self._max_nr_iterations = round(max_duration / scenario.database_interval)
        else:
            self._max_nr_iterations = scenario_nr_iterations
        self._start_iteration = round(start_time * scenario.database_interval)
        self._end_iteration = min(
            [
                scenario_nr_iterations,
                self._start_iteration + self._max_nr_iterations - 1,
            ]
        )
        self.current_iteration_index = self._start_iteration

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration_index = self._start_iteration

    def get_iteration(self) -> SimulationIteration:
        """Inherited, see superclass."""
        scenario_time = self.scenario.get_time_point(self.current_iteration_index)
        return SimulationIteration(time_point=scenario_time, index=self.current_iteration_index)

    def next_iteration(self) -> Optional[SimulationIteration]:
        """Inherited, see superclass."""
        self.current_iteration_index += 1
        return None if self.reached_end() else self.get_iteration()

    def reached_end(self) -> bool:
        """Inherited, see superclass."""
        return self.current_iteration_index >= self._end_iteration - 1

    def number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return self._end_iteration - self._start_iteration + 1
