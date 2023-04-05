import pickle

import hj_reachability as hj
import numpy as np
from hj_reachability import dynamics

from severity_estimation.hamilton_jacobi.system_dynamics import DynUnicycleCAvoid
from severity_estimation.hamilton_jacobi.utils import (
    convert_relative_state_to_signed_distance,
)

DEFAULT_CONTROL_BOUNDS = {
    "evader_accel_bounds": [
        -1.0,
        1,
    ],  # change this to get braking only, or constant motion
    "pursuer_accel_bounds": [
        -1.0,
        1.0,
    ],  # change this to get braking only, or constant motion
    "evader_max_steering": 0.5,  # omega max/min [rads/sec]
    "pursuer_max_steering": 0.5,  # omega max/min [rads/sec]
    "evader_min_speed": 0.0,
    "pursuer_min_speed": 0.0,
    "evader_max_speed": 5.0,
    "pursuer_max_speed": 5.0,
    "control_mode": "min",
    "disturbance_mode": "min",
}
# State: x, y, relative heading, v_ego, v_other
DEFAULT_GRID = {
    "lower_bound": [-22.0, -22.0, -np.pi, 0, 0],
    "upper_bound": [22.0, 22.0, np.pi, 16, 20],
    "num_points": (50, 50, 30, 30, 40),
    "periodic_dims": 2,
}
DEFAULT_CAR_PARAMS = {"width": 2.297, "length": 5.176, "L": 0.0}
SIGNED_DISTANCE = lambda state: convert_relative_state_to_signed_distance(
    state,
    width=DEFAULT_CAR_PARAMS["width"],
    length=DEFAULT_CAR_PARAMS["length"],
    L=DEFAULT_CAR_PARAMS["L"],
)


class HJSeverity:
    def __init__(
        self,
        sys_dynamics=DynUnicycleCAvoid(**DEFAULT_CONTROL_BOUNDS),
        value_function=SIGNED_DISTANCE,
        grid=DEFAULT_GRID,
    ) -> None:
        assert isinstance(sys_dynamics, dynamics.ControlAndDisturbanceAffineDynamics)
        assert "lower_bound" in grid and "upper_bound" in grid and "num_points" in grid
        assert len(grid["lower_bound"]) == len(grid["upper_bound"]) and len(
            grid["num_points"]
        ) == len(grid["lower_bound"])

        self._sys_dynamics = sys_dynamics
        self._lattice = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj.sets.Box(
                lo=np.array(grid["lower_bound"]),
                hi=np.array(grid["upper_bound"]),
            ),
            shape=grid["num_points"],  # number of grid points
            periodic_dims=grid["periodic_dims"],
        )
        self._init_values = hj.utils.multivmap(
            value_function, np.arange(self._lattice.ndim)
        )(self._lattice.states)
        self._target_values = None
        self._time_horizon = None

    def precompute(self, time_horizon=2.0, accuracy="high", progress_bar=True):
        self._time_horizon = time_horizon
        # accuracy can be "medium", "high", or "low" if you want. e.g., can choose "medium" but with a denser grid. Might run faster / improve quality.
        solver_settings = hj.SolverSettings.with_accuracy(
            accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )
        initial_time = 0.0
        target_time = -self._time_horizon  # backward in time
        self._target_values = hj.step(
            solver_settings,
            self._sys_dynamics,
            self._lattice,
            initial_time,
            self._init_values,
            target_time,
            progress_bar=progress_bar,
        ).block_until_ready()

    def __call__(self, state):
        assert self._target_values is not None, "Must call precompute() first"
        return self._lattice.interpolate(self._target_values, state).item()

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
