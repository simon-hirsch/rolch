import numpy as np

from rolch.abc import LinkFunction

LOG_LOWER_BOUND = 1e-25
EXP_UPPER_BOUND = 25


class LogLink(LinkFunction):
    def __init__(self):
        pass

    def link(self, x):
        return np.log(np.fmax(x, LOG_LOWER_BOUND))

    def inverse(self, x):
        return np.exp(np.fmin(x, EXP_UPPER_BOUND))

    def derivative(self, x):
        return np.exp(np.fmin(x, EXP_UPPER_BOUND))


class IdentityLink(LinkFunction):
    def __init__(self):
        pass

    def link(self, x):
        return x

    def inverse(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class LogShiftValueLink(LinkFunction):
    def __init__(self, value):
        self.value = value

    def link(self, x):
        return np.log(x - self.value + LOG_LOWER_BOUND)

    def inverse(self, x):
        return self.value + np.fmax(
            np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND
        )

    def derivative(self, x):
        return np.fmax(np.exp(np.fmin(x, EXP_UPPER_BOUND)), LOG_LOWER_BOUND)


class LogShiftTwoLink(LogShiftValueLink):
    def __init__(self):
        super().__init__(2)
        pass


__all__ = [LogLink, IdentityLink, LogShiftValueLink, LogShiftTwoLink]
