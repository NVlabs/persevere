# `NamedTuple`s are used (more accurately, abused) in this notebook to minimize dependencies;
# better choices would be `flax.struct.dataclass` or `equinox.Module`.
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def interval_interval_separation_distance(interval_0, interval_1):
    return jnp.maximum(
        interval_0[0] - interval_1[1],
        interval_1[0] - interval_0[1],
    )


def rotate_points(points, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return points @ jnp.array([[c, s], [-s, c]])


class Rectangle(NamedTuple):
    center: jnp.array
    orientation: jnp.array
    half_dimensions: jnp.array  # (half_width, half_height)

    @property
    def corners(self):
        half_width, half_height = self.half_dimensions
        untransformed_corners = jnp.array(
            [
                [half_width, half_height],
                [-half_width, half_height],
                [-half_width, -half_height],
                [half_width, -half_height],
            ]
        )
        return self.center + rotate_points(untransformed_corners, self.orientation)

    def distance_to_point(self, point):
        point_in_body_frame = rotate_points(point - self.center, -self.orientation)
        return jnp.linalg.norm(
            point_in_body_frame
            - jnp.clip(point_in_body_frame, -self.half_dimensions, self.half_dimensions)
        )


@jax.jit
def rectangle_rectangle_separation_distance(
    rectangle_0: Rectangle, rectangle_1: Rectangle
):
    def _separation_distance(rectangle_0, rectangle_1):
        points_1 = rotate_points(
            rectangle_1.corners - rectangle_0.center, -rectangle_0.orientation
        )
        return jnp.max(
            jax.vmap(interval_interval_separation_distance, 1)(
                jnp.array([-rectangle_0.half_dimensions, rectangle_0.half_dimensions]),
                jnp.array([jnp.min(points_1, 0), jnp.max(points_1, 0)]),
            )
        )

    return jnp.maximum(
        _separation_distance(rectangle_0, rectangle_1),
        _separation_distance(rectangle_1, rectangle_0),
    )


@jax.jit
def rectangle_rectangle_signed_distance(rectangle_0: Rectangle, rectangle_1: Rectangle):
    separation_distance = rectangle_rectangle_separation_distance(
        rectangle_0, rectangle_1
    )
    return jnp.where(
        separation_distance < 0,
        separation_distance,
        jnp.minimum(
            jnp.min(jax.vmap(rectangle_0.distance_to_point)(rectangle_1.corners)),
            jnp.min(jax.vmap(rectangle_1.distance_to_point)(rectangle_0.corners)),
        ),
    )


def plot_rectangle(r0):
    plt.scatter(*r0.corners.T, s=10)
    plt.plot(*np.concatenate([r0.corners, r0.corners[:1, :]], 0).T)


def convert_relative_state_to_signed_distance(
    relative_state, width=0.5, length=1.1, L=1.0, plot=False
):
    r0 = Rectangle(jnp.array([L / 2, 0.0]), 0.0, jnp.array([length / 2, width / 2]))
    center = (
        relative_state[:2]
        + jnp.array([jnp.cos(relative_state[2]), jnp.sin(relative_state[2])]) * L / 2
    )
    r1 = Rectangle(center, relative_state[2], jnp.array([length / 2, width / 2]))

    if plot:
        plt.title(rectangle_rectangle_signed_distance(r0, r1))
        plot_rectangle(r0)
        plot_rectangle(r1)
        plt.scatter([0], [0], s=10, c="tab:blue")
        plt.scatter(*relative_state[:2], s=10, c="tab:orange")

        plt.axis("equal")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.grid()
    return rectangle_rectangle_signed_distance(r0, r1)
