from abc import ABC

import numpy as np


class Estimator(ABC):

    def __init__(self):
        self.n_observations: int
        self.n_training: float

    @property
    def is_fitted(self) -> bool:
        """Has the estimator been fitted."""
        return self.n_observations > 0

    @property
    def n_obs(self) -> int:
        return self.n_observations

    @property
    def n_train(self) -> float:
        return self.n_training

    @property
    def coef_(self):
        return self.beta

    @property
    def coef_path_(self):
        return self.beta_path

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError("Not Implemented")

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        raise NotImplementedError("Not Implemented")

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.update(X=X, y=y, sample_weights=sample_weights)
