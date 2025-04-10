from abc import ABC, abstractmethod


import numpy as np


class TransformationCallback(ABC):

    max_lags: int
    J: int

    def __init__(self, max_lag: int, min_lag: int, J: int):
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.J = J

    @abstractmethod
    def __call__(self, residuals: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented.")
