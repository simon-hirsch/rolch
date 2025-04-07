import numpy as np

from ..base import TransformationCallback
from ..utils import parse_to_array_for_lags


class LaggedValue(TransformationCallback):
    def __init__(self, lags: np.ndarray | int):
        self.lags = parse_to_array_for_lags(lags)
        super().__init__(np.max(self.lags), np.min(self.lags), len(self.lags))

    def __call__(self, residuals):
        return np.vstack([np.roll(residuals, i) for i in self.lags]).T


class LaggedAbsoluteValue(TransformationCallback):
    def __init__(self, lags: np.ndarray | int):
        self.lags = parse_to_array_for_lags(lags)
        super().__init__(np.max(self.lags), np.min(self.lags), len(self.lags))

    def __call__(self, residuals):
        return np.vstack([np.roll(np.abs(residuals), i) for i in self.lags]).T


class LaggedSquaredValue(TransformationCallback):
    def __init__(self, lags: np.ndarray | int):
        self.lags = parse_to_array_for_lags(lags)
        super().__init__(np.max(self.lags), np.min(self.lags), len(self.lags))

    def __call__(self, residuals):
        return np.vstack([np.roll(np.power(residuals, 2), i) for i in self.lags]).T


class LaggedLeverageEffect(TransformationCallback):
    """The leverage effect is the phenomenon where negative returns are associated with higher volatility than positive returns.
    This residual transformation will return the absolute value of the negative residuals lagged by the specified lags.
    """

    def __init__(self, lags: np.ndarray | int):
        self.lags = parse_to_array_for_lags(lags)
        super().__init__(np.max(self.lags), np.min(self.lags), len(self.lags))

    def __call__(self, residuals):
        neg_returns = np.where(residuals < 0, residuals, 0)
        return np.vstack([np.roll(np.abs(neg_returns), i) for i in self.lags]).T
