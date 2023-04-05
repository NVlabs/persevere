import abc
from typing import Union

import numpy as np

from severity_estimation.fault_injection.utils import wrap_pi


class Parameter(abc.ABC):
    @abc.abstractmethod
    def get(self):
        pass


class Size(Parameter):
    width: float
    length: float
    noise_std: float = 0.0
    MIN_SIZE = 0.2

    def __init__(self, width, length, noise_std=0.0):
        assert width > 0.0, "Width must be positive"
        assert length > 0.0, "Length must be positive"
        assert noise_std >= 0.0, "Noise STD must be non-negative"
        self.width = width
        self.length = length
        self.noise_std = noise_std

    def get(self):
        return Size(
            width=max(self.MIN_SIZE, self.width + np.random.normal(0, self.noise_std)),
            length=max(
                self.MIN_SIZE, self.length + np.random.normal(0, self.noise_std)
            ),
        )


class Offset(Parameter):
    angle: float
    distance: float
    noise_std: float = 0.0

    def __init__(self, angle, distance, noise_std=0.0):
        assert noise_std >= 0.0, "Noise STD must be non-negative"
        self.angle = wrap_pi(angle)
        self.distance = distance
        self.noise_std = noise_std

    def get(self):
        return Offset(
            angle=self.angle + np.random.normal(0, self.noise_std),
            distance=self.distance + np.random.normal(0, self.noise_std),
        )


class Position(Parameter):
    x: float
    y: float
    heading: float
    noise_std: float = 0.0

    def __init__(self, x, y, heading, noise_std=0.0):
        assert noise_std >= 0.0, "Noise STD must be non-negative"
        self.x = x
        self.y = y
        self.heading = heading
        self.noise_std = noise_std

    def get(self):
        return Position(
            x=self.x + np.random.normal(0, self.noise_std),
            y=self.y + np.random.normal(0, self.noise_std),
            heading=self.heading,
        )


class Gaussian(Parameter):
    mean: float
    std: float

    def __init__(self, mean, std):
        assert std >= 0.0, "STD must be non-negative"
        self.mean = mean
        self.std = std

    def get(self):
        return np.random.normal(self.mean, self.std)


class Uniform(Parameter):
    min: float
    max: float

    def __init__(self, min, max):
        assert min <= max, "Min must be <= max"
        self.min = min
        self.max = max

    def get(self):
        return np.random.uniform(self.min, self.max)


class Constant(Parameter):
    value: float

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


Number = Union[Gaussian, Uniform, Constant]


class Angle(Parameter):
    def __init__(self, num_generator: Number) -> None:
        self._num_generator = num_generator

    def get(self):
        return wrap_pi(self._num_generator.get())
