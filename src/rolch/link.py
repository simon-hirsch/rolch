import numpy as np

from rolch.abc import LinkFunction

LOG_LOWER_BOUND = 1e-25
EXP_UPPER_BOUND = 25
SMALL_NUMBER = 1e-10


class LogLink(LinkFunction):
    """
    The log-link function.

    The log-link function is defined as \(g(x) = \log(x)\).
    """

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


class IdentityLink(LinkFunction):
    """
    The identity link function.

    The identity link is defined as \(g(x) = x\).
    """

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

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(x - self.value + LOG_LOWER_BOUND)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.fmax(
            np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND
        )

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.fmax(np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return super().link_derivative(x)


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

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x - self.value + LOG_LOWER_BOUND)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.value + np.power(np.fmin(x, EXP_UPPER_BOUND), 2)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return super().link_derivative(x)


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

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 1, np.log(x), x - 1)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, np.exp(x), x + 1)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, np.exp(x), 1)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return super().link_derivative(x)


__all__ = [
    "LogLink",
    "IdentityLink",
    "LogShiftValueLink",
    "LogShiftTwoLink",
    "LogIdentLink",
    "SqrtLink",
    "SqrtShiftValueLink",
    "SqrtShiftTwoLink",
]
