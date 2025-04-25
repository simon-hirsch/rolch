from typing import Tuple

import numba as nb
import numpy as np

from .base import LinkFunction

SMALL_NUMBER = 1e-10
LARGE_NUMBER = 1e10
LOG_LARGE_NUMBER = np.log(LARGE_NUMBER)


@nb.vectorize(["float64(float64)", "float32(float32)"])
def robust_log(x: np.ndarray) -> np.ndarray:
    """
    A robust log function that handles negative and zero values.

    This function returns the logarithm of the input array, replacing
    negative and zero values with a small positive number to avoid
    undefined logarithm values.
    """
    if x > SMALL_NUMBER:
        return np.log(x)
    else:
        return np.log(x)


@nb.vectorize(["float64(float64)", "float32(float32)"])
def robust_exp(x: np.ndarray) -> np.ndarray:
    """
    A robust exponential function that handles large values.
    """

    if x > LOG_LARGE_NUMBER:
        return LARGE_NUMBER
    else:
        return np.exp(x)


class LogLink(LinkFunction):
    """
    The log-link function.

    The log-link function is defined as \(g(x) = \log(x)\).
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / x

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / x**2


class IdentityLink(LinkFunction):
    """
    The identity link function.

    The identity link is defined as \(g(x) = x\).
    """

    link_support = (-np.inf, np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class LogShiftValueLink(LinkFunction):
    """
    The Log-Link function shifted to a value \(v\).

    This link function is defined as \(g(x) = \log(x - v)\). It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.value + np.nextafter(0, 1), np.inf)

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x - self.value + SMALL_NUMBER)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + robust_exp(x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return robust_exp(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (x - self.value + SMALL_NUMBER)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / (x - self.value + SMALL_NUMBER) ** 2


class LogShiftTwoLink(LogShiftValueLink):
    """
    The Log-Link function shifted to 2.

    This link function is defined as \(g(x) = \log(x - 2)\). It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self):
        super().__init__(2)
        pass


class SqrtLink(LinkFunction):
    """The square root Link function.

    The square root link function is defined as $$g(x) = \sqrt(x)$$.
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.power(x, 2)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (2 * np.sqrt(x))

    def link_second_derivative(self, x) -> np.ndarray:
        return -1 / (4 * x ** (3 / 2))


class SqrtShiftValueLink(LinkFunction):
    """
    The Sqrt-Link function shifted to a value \(v\).

    This link function is defined as $$g(x) = \sqrt(x - v)$$. It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.value + np.nextafter(0, 1), np.inf)

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x - self.value + SMALL_NUMBER)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.power(x, 2)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (2 * np.sqrt(x - self.value + SMALL_NUMBER))

    def link_second_derivative(self, x) -> np.ndarray:
        return -1 / (4 * (x - self.value + SMALL_NUMBER) ** (3 / 2))


class SqrtShiftTwoLink(SqrtShiftValueLink):
    """
    The Sqrt-Link function shifted to 2.

    This link function is defined as $$g(x) = \sqrt(x - 2)$$. It can be used
    to ensure that certain distribution paramters don't fall below lower
    bounds, e.g. ensuring that the degrees of freedom of a Student's t distribtuion
    don't fall below 2, hence ensuring that the variance exists.
    """

    def __init__(self):
        super().__init__(2)
        pass


class LogIdentLink(LinkFunction):
    """The Logident Link function.

    The LogIdent Link function has been introduced by [Narajewski & Ziel 2020](https://arxiv.org/pdf/2005.01365) and can be
    used to avoid the exponential inverse for large values while keeping the log-behaviour in small ranges. This can stabilize
    the estimation procedure.
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 1, robust_log(x), x - 1)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, robust_exp(x), x + 1)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, robust_exp(x), 1)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return super().link_derivative(x)

    def link_second_derivative(self, x) -> np.ndarray:
        return super().link_second_derivative(x)


__all__ = [
    "LogLink",
    "IdentityLink",
    "LogShiftValueLink",
    "LogShiftTwoLink",
    "LogIdentLink",
    "SqrtLink",
    "SqrtShiftValueLink",
    "SqrtShiftTwoLink",
    "SqrtShiftTwoLink",
]
