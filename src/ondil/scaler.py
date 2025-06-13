import numpy as np

from .gram import (
    init_forget_vector,
)


class OnlineScaler:
    def __init__(
        self,
        forget: float = 0.0,
        to_scale: bool | np.ndarray = True,
    ):
        """The online scaler allows for incremental updating and scaling of matrices.

        Args:
            forget (float, optional): The forget factor. Older observations will be exponentially discounted. Defaults to 0.0.
            to_scale (bool | np.ndarray, optional): The variables to scale.
                `True` implies all variables will be scaled.
                `False` implies no variables will be scaled.
                An `np.ndarray` of type `bool` or `int` implies that the columns `X[:, to_scale]` will be scaled, all other columns will not be scaled.
                Defaults to True.
        """
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
        self.w = 0  # Track cumulative weights for exponential forgetting

    def fit(self, X: np.ndarray, sample_weight: np.ndarray = None) -> None:
        """Fit the OnlineScaler() Object for the first time.

        Args:
            X (np.ndarray): Matrix of covariates X.
            sample_weight (np.ndarray, optional): Weights for each sample. Defaults to None (uniform weights).
        """
        self._prepare_estimator(X)
        if self._do_scale:
            if sample_weight is None:
                sample_weight = np.ones(X.shape[0])

            forget_vector = init_forget_vector(self.forget, X.shape[0])
            effective_weights = sample_weight * forget_vector

            self.w = np.sum(effective_weights)  # Initialize cumulative weight
            self.m = np.average(
                X[:, self._selection], weights=effective_weights, axis=0
            )

            # Calculate the variance of each column of x_init and assing it to self.v
            diff_sq = (X[:, self._selection] - self.m) ** 2
            self.v = np.average(diff_sq, weights=effective_weights, axis=0)
            self.M = self.v * self.w
        else:
            pass

    def update(self, X: np.ndarray, sample_weight: np.ndarray = None) -> None:
        """Wrapper for partial_fit to align API."""
        self.partial_fit(X, sample_weight)

    def partial_fit(self, X: np.ndarray, sample_weight: np.ndarray = None) -> None:
        """Update the `OnlineScaler()` for new rows of X.

        Args:
            X (np.ndarray): New data for X.
            sample_weight (np.ndarray, optional): Weights for each sample. Defaults to None (uniform weights).
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        # Loop over all rows of new X
        if self._do_scale:
            for i in range(X.shape[0]):
                # Effective weight for the old state
                eff_old_w = self.w * (1 - self.forget)
                self.w = eff_old_w + sample_weight[i]
                diff_old = X[i, self._selection] - self.m

                # Update mean
                self.m = (
                    self.m * eff_old_w + X[i, self._selection] * sample_weight[i]
                ) / self.w

                diff_new = X[i, self._selection] - self.m

                # Update M (sum of squared deviations)
                self.M = (
                    self.M * (1 - self.forget) + sample_weight[i] * diff_old * diff_new
                )

                # Update variance
                self.v = self.M / self.w

        else:
            pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a mean-std scaled matrix.

        Args:
            X (np.ndarray): X matrix for covariates.

        Returns:
            np.ndarray: Scaled X matrix.
        """
        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = (X[:, self._selection] - self.m) / np.sqrt(self.v)
            return out
        else:
            return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Back-transform a scaled X matrix to the original domain.

        Args:
            X (np.ndarray): Scaled X matrix.

        Returns:
            np.ndarray: Scaled back to the original scale.
        """
        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = X[:, self._selection] * np.sqrt(self.v) + self.m
            return out
        else:
            return X
