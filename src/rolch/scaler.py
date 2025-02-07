import numpy as np

from rolch.utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)


class OnlineScaler:
    def __init__(
        self,
        forget: float = 0.0,
        to_scale: bool | np.ndarray = True,
    ):
        self.forget = forget
        self.to_scale = to_scale

    def _prepare_estimator(self, X: np.ndarray):
        """Add derived attributes to estimator"""
        # Slowly align with sklearn API of not saving more | less in the construction than user passed arguments
        # Prepare the scaling
        if isinstance(self.to_scale, np.ndarray):
            self._selection = self.to_scale
            self._do_scale = True
        elif isinstance(self.to_scale, bool):
            if self.to_scale:
                self._selection = np.arange(X.shape[1])
                self._do_scale = True
            else:
                self._selection = False
                self._do_scale = False

        # Variables
        self.m = 0
        self.M = 0
        self.v = 0
        self.n_observations = X.shape[0]
        self.n_asymmptotic = calculate_asymptotic_training_length(self.forget)

    def fit(self, X: np.ndarray):
        self._prepare_estimator(X)
        if self._do_scale:
            # Calculate the mean of each column of x_init and assing it to self.m
            self.m = np.mean(X[:, self._selection], axis=0)
            # Calculate the variance of each column of x_init and assing it to self.v
            self.v = np.var(X[:, self._selection], axis=0)
            self.M = self.v * self.n_observations
        else:
            pass

    def update(self, X: np.ndarray):
        """Wrapper for partial_fit to align API."""
        self.partial_fit(X)

    def partial_fit(self, X: np.ndarray):
        # Loop over all rows of new X
        if self._do_scale:
            for i in range(X.shape[0]):
                self.n_observations += 1
                n_seen = calculate_effective_training_length(
                    self.forget, self.n_observations
                )

                forget_scaled = self.forget * np.maximum(
                    self.n_asymmptotic / n_seen, 1.0
                )

                diff = X[i, self._selection] - self.m
                incr = forget_scaled * diff

                if forget_scaled > 0:
                    self.m += incr
                    self.v = (1 - forget_scaled) * (self.v + forget_scaled * diff**2)
                else:
                    self.m += diff / self.n_observations
                    self.M += diff * (X[i, self._selection] - self.m)
                    self.v = self.M / self.n_observations
        else:
            pass

    def transform(self, X: np.ndarray):
        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = (X[:, self._selection] - self.m) / np.sqrt(self.v)
            return out
        else:
            return X

    def inverse_transform(self, X: np.ndarray):
        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = X[:, self._selection] * np.sqrt(self.v) + self.m
            return out
        else:
            return X
