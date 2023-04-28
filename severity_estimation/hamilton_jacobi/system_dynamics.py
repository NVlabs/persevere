import jax.numpy as jnp
from hj_reachability import dynamics, sets


class DynUnicycleCAvoid(dynamics.ControlAndDisturbanceAffineDynamics):
    """Relative dynamics of a two-agent system where each is modelled as a dynamically extended simple car model.
    Dynamically extended simple car model:

    dx   = v * cos(phi)
    dy   = v * sin(phi)
    dphi = w
    dv   = a

    A is the ego, B is the other agent

    Relative (R) dynamics:
        zRdot = [-vA + vB * cos(phiR) + yR * wA,
                  vB * sin(phiR) - xR * wA,
                  wB - wA,
                  aA,
                  aB],

        where:
            xR = (xB - xA) * cos(phiA)  + (yB - yA) * sin(phiA)
            yR = -(xB - xA) * sin(phiA)  + (yB - yA) * cos(phiA)
            phiR = phiB - phiA
            vA
            vB

    Control of agent (A):
        uA = [wA, aA]

        where:
            wA in [-wmaxA, wmaxA]
            aA in [aminA, amaxA]

    Control of agent (B):
        uB = [wB, aB]

        where:
            dB in [-wmaxB, wmaxB]
            aB in [aminB, amaxB]
    """

    def __init__(
        self,
        evader_accel_bounds=[-2.0, 1.0],
        pursuer_accel_bounds=[-2.0, 1.0],
        evader_max_steering=0.5,
        pursuer_max_steering=0.5,
        control_mode="max",
        disturbance_mode="min",
        control_space=None,
        disturbance_space=None,
        evader_min_speed=0.0,
        pursuer_min_speed=0.0,
        evader_max_speed=10.0,
        pursuer_max_speed=10.0,
    ):

        self.evader_min_max_speed = [evader_min_speed, evader_max_speed]
        self.pursuer_min_max_speed = [pursuer_min_speed, pursuer_max_speed]

        if control_space is None:
            control_space = sets.Box(
                lo=jnp.array([-evader_max_steering, evader_accel_bounds[0]]),
                hi=jnp.array([evader_max_steering, evader_accel_bounds[1]]),
            )

        if disturbance_space is None:
            disturbance_space = sets.Box(
                lo=jnp.array([-pursuer_max_steering, pursuer_accel_bounds[0]]),
                hi=jnp.array([pursuer_max_steering, pursuer_accel_bounds[1]]),
            )

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, psiR, vA, vB = state
        vA = jnp.clip(vA, *self.evader_min_max_speed)
        vB = jnp.clip(vB, *self.pursuer_min_max_speed)
        return jnp.array([-vA + vB * jnp.cos(psiR), vB * jnp.sin(psiR), 0.0, 0.0, 0.0])

    def control_jacobian(self, state, time):
        """
        uA = [wA, aA]
        """
        xR, yR, _, vA, _ = state
        return jnp.array([[yR, 0.0], [-xR, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def disturbance_jacobian(self, state, time):
        """
        uB = [wB, aB]
        """
        _, _, _, _, vB = state
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
