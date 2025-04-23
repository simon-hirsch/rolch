from typing import Tuple

import numba as nb
import numpy as np

from .base import LinkFunction

LOG_LOWER_BOUND = 1e-25
EXP_UPPER_BOUND = 25
SMALL_NUMBER = 1e-10


@nb.vectorize(["float64(float64)", "float32(float32)"])
def robust_log(x: float) -> float:
    if x < SMALL_NUMBER:
        return SMALL_NUMBER
    else:
        return np.log(x)


@nb.vectorize(["float64(float64)", "float32(float32)"])
def robust_exp(x: float) -> float:
    """Avoid overflow in exp by clipping the input.

    We calculate $\exp(x) = \exp(\operatorname{clip}(x, -5, 100))$.
    This gives a range between rougly 0.0004 and 2.6881e+43.

    Args:
        x (float): input value

    Returns:
        float: exponential of the input value
    """
    return np.exp(min(max(x, -5.0), 100.0))


@nb.vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def robust_sigmoid(value, k):
    if value / k > 50:
        return 1
    elif value / k < -50:
        return 0
    else:
        return 1 / (1 + np.exp(-k * (value - 1)))


@nb.vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def zero_safe_division(a, b):
    """Return the 0 at the a / b if b becomes 0.

    Args:
        a (float): Nominator.
        b (float): Denominator.

    Returns:
        float: Result of the division a / b.
    """
    if np.isclose(b, 0):
        return 0
    elif np.isclose(a, 0):
        return 0
    else:
        return a / b


class LogLink(LinkFunction):
    """
    The log-link function.

    The log-link function is defined as \(g(x) = \log(x)\).
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.fmax(x, LOG_LOWER_BOUND))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.fmax(
            np.exp(np.fmin(x, EXP_UPPER_BOUND)),
            LOG_LOWER_BOUND,
        )

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.fmin(x, EXP_UPPER_BOUND))

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
        return np.log(x - self.value + LOG_LOWER_BOUND)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.fmax(
            np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND
        )

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.fmax(np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (x - self.value)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / (x - self.value) ** 2


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
        return np.sqrt(x - self.value + LOG_LOWER_BOUND)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.power(np.fmin(x, EXP_UPPER_BOUND), 2)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (2 * np.sqrt(x - self.value))

    def link_second_derivative(self, x) -> np.ndarray:
        return -1 / (4 * (x - self.value) ** (3 / 2))


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


class InverseSoftPlusLink(LinkFunction):
    """
    The softplus is defined as $$
        \operatorname{SoftPlus(x)} = \log(1 + \exp(x))
    $$ and hence the inverse is defined as $$
        \log(\exp(x) - 1)
    $$ which can be used as link function for the parameters on the
    positive real line. The behavior of the inverse softplus is more
    graceful on large values as it avoids the exp of the log-link and
    converges towards a linear behaviour.

    The softplus is the smooth approximation of $\max(x, 0)$.
    """

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self):
        pass

    def link(self, x: np.ndarray):
        return np.log(np.clip(np.exp(x) - 1, 1e-10, np.inf))

    def inverse(self, x: np.ndarray):
        return np.log(1 + np.exp(x))

    def link_derivative(self, x: np.ndarray):
        return 1 / (np.exp(x) - 1) + 1

    def link_second_derivative(self, x: np.ndarray):
        # return 1 / (2 - 2 * np.cosh(x)) # equivalent
        return -1 / (np.exp(x) - 1) - 1 / (np.exp(x) - 1) ** 2

    def inverse_derivative(self, x: np.ndarray):
        return 1 / (np.exp(-x) + 1)


class InverseSoftPlusShiftValueLink(LinkFunction):
    """
    The Inverse SoftPlus function shifted to a value \(v\).
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.value + np.nextafter(0, 1), np.inf)

    def link(self, x: np.ndarray):
        return np.log(np.exp(x - self.value) - 1)

    def inverse(self, x: np.ndarray):
        return np.log(1 + np.exp(x)) + self.value

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (np.exp(-x) + 1))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (1 - np.exp(self.value - x)))

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return zero_safe_division(1, (2 - 2 * np.cosh(self.value - x)))


class InverseSoftPlusShiftTwoLink(InverseSoftPlusShiftValueLink):
    """
    The Inverse SoftPlus function shifted to 2.
    """

    def __init__(self):
        super().__init__(2)
        pass


class TwiceDifferentiableLogIdentLink(LogIdentLink):

    link_support = (np.nextafter(0, 1), np.inf)

    def __init__(self, k=1):
        self.k = k

    def link(self, x: np.ndarray):
        return np.where(
            x <= 1,
            np.log(x),
            (1 - robust_sigmoid(x, self.k)) * np.log(x)
            + robust_sigmoid(x, self.k) * (x - 1),
        )

    def link_derivative(self, x):
        sigm = robust_sigmoid(x, self.k)
        deriv = (
            (1 - sigm) * (1 / x)
            + self.k * sigm * (1 - sigm) * (np.log(x) - (x - 1))
            + sigm
        )
        return deriv

    def link_second_derivative(self, x):
        sigm = robust_sigmoid(x, self.k)
        sigm_d1 = self.k * sigm * (1 - sigm)
        sigm_d2 = self.k**2 * sigm * (1 - sigm) * (1 - 2 * sigm)
        deriv = (
            -sigm_d1 / x
            - (1 - sigm) / x**2
            + sigm_d2 * (np.log(x) - (x - 1))
            + sigm_d1 * (1 / x)
        )
        return deriv


class LogIdentShiftValueLink(LinkFunction):

    def __init__(self, value):
        self.value = value

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.value + np.nextafter(0, 1), np.inf)

    def link(self, x: np.ndarray):
        return np.where(
            x < (1 + self.value),
            np.log(
                np.fmax(LOG_LOWER_BOUND, x - self.value)
            ),  # Ensure that everything is robust
            x - 1 - self.value,
        )

    def inverse(self, x):
        return np.where(
            x <= 0,
            self.value + np.exp(np.fmin(x, EXP_UPPER_BOUND)),
            x + 1 + self.value,
        )

    def link_derivative(self, x):
        # return np.where(x < (1 + self.value), 1 / (x - self.value), 1)
        raise ValueError("Not  continuous differentiable.")

    def link_second_derivative(self, x):
        raise ValueError("Not twice continuous differentiable.")

    def inverse_derivative(self, x):
        return np.where(x <= 0, np.exp(np.fmin(x - 1, EXP_UPPER_BOUND)), 1)


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
    "InverseSoftPlusLink",
    "InverseSoftPlusShiftValueLink",
    "InverseSoftPlusShiftTwoLink",
    "TwiceDifferentiableLogIdentLink",
    "LogIdentShiftValueLink",
]
