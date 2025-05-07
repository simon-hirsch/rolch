from typing import Tuple

import numpy as np

from ..base import LinkFunction
from .robust_math import robust_log, zero_safe_division


def sigmoid(y):
    return 1 / (1 + np.fmax(1e-6, np.exp(-np.fmin(y, 25))))


class ScaledInverseSigmoidLink(LinkFunction):

    def __init__(
        self,
        lower: float = 0,
        upper: float = 1,
    ):
        self.lower = lower
        self.upper = upper

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.lower + np.nextafter(0, 1), self.upper - np.nextafter(0, 1))

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x - self.lower) - robust_log(self.upper - x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * sigmoid(x)

    def inverse_derivative(self, x):
        return (
            (self.upper - self.lower)
            * sigmoid(x)
            * (1 - sigmoid(x))
            / (self.upper - self.lower)
        )

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, x - self.lower) - zero_safe_division(
            1, x - self.upper
        )

    def link_second_derivative(self, x) -> np.ndarray:
        return super().link_second_derivative(x)


class ScaledSigmoidLink(LinkFunction):

    def __init__(
        self,
        lower: float = 0,
        upper: float = 1,
    ):
        self.lower = lower
        self.upper = upper

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.lower + np.nextafter(0, 1), self.upper - np.nextafter(0, 1))

    def link(self, x: np.ndarray) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * sigmoid(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.fmax(1e-10, x - self.lower)) - np.log(
            np.fmax(1e-10, self.upper - x)
        )

    def inverse_derivative(self, x):
        return 1 / (x - self.lower) - 1 / (x - self.upper)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return (
            (self.upper - self.lower)
            * sigmoid(x)
            * (1 - sigmoid(x))
            / (self.upper - self.lower)
        )

    def link_second_derivative(self, x) -> np.ndarray:
        return super().link_second_derivative(x)
