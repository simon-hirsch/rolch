import numpy as np

from ..utils import parse_to_array_for_lags
from .autoregressive_effects import (
    LaggedAbsoluteValue,
    LaggedLeverageEffect,
    LaggedSquaredValue,
    LaggedValue,
    add_lagged_effects,
    make_lagged_effects,
)

__all__ = [
    "LaggedValue",
    "LaggedAbsoluteValue",
    "LaggedSquaredValue",
    "LaggedLeverageEffect",
    "add_lagged_effects",
    "make_lagged_effects",
]


def make_intercept(n_observations: int) -> np.ndarray:
    return np.ones((n_observations, 1))


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack((make_intercept(X.shape[0]), X))


def make_lags(y: np.ndarray, lags: int | np.ndarray) -> np.ndarray:
    return np.vstack([np.roll(y, i, axis=0) for i in parse_to_array_for_lags(lags)]).T


def add_lags(X: np.ndarray, y: np.ndarray, lags: int | np.ndarray) -> np.ndarray:
    """Add lags of $y$ to the design matrix $X$.

    !!! note
        The lags are added to the end of the design matrix and we do not remove the first rows of the design matrix.

    Args:
        X (np.ndarray): Design matrix.
        y (np.ndarray): Variable to lag.
        lags (int | np.ndarray): Number / array of lags to add.

    Returns:
        np.ndarray: Returns the design matrix with the lags of $y$.
    """

    if np.max(lags) >= y.shape[0]:
        error = f"The maximum lag ({max(lags)}) should be less than the number of observations: {y.shape[0]}"
        raise ValueError(error)
    if np.min(lags) < 1:
        out = X
    else:
        out = np.hstack((X, make_lags(y, lags)))

    return out
