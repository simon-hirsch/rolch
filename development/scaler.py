# %%
import numpy as np
import sys


def calculate_asymptotic_training_length(forget: float):
    if forget == 0:
        # Return the highest possible value that is no infinity
        return sys.maxsize
    else:
        return 1 / forget


def calculate_effective_training_length(forget: float, n_obs: int):
    if forget == 0:
        return n_obs
    else:
        return (1 - (1 - forget) ** n_obs) / forget


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
        self.n_observations = X.shape[0]
        self.sum_weights = 0  # Track sum of weights for weighted updates
        self.n_asymmptotic = calculate_asymptotic_training_length(self.forget)

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

            # Calculate weighted mean and variance
            self.sum_weights = np.sum(sample_weight)
            self.w = self.sum_weights  # Initialize cumulative weight
            self.m = np.average(X[:, self._selection], weights=sample_weight, axis=0)

            # Calculate weighted variance
            diff_sq = (X[:, self._selection] - self.m) ** 2
            self.v = np.average(diff_sq, weights=sample_weight, axis=0)
            self.M = self.v * self.sum_weights
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
                self.n_observations += 1
                weight = sample_weight[i]

                # Exponential forgetting case
                old_m = self.m
                old_w = self.w

                # Update cumulative weight first
                self.w = old_w * (1 - self.forget) + weight

                # Update mean using exponential forgetting formula
                self.m = (
                    old_m * old_w * (1 - self.forget) + X[i, self._selection] * weight
                ) / self.w

                # Update variance using the correct formula for exponential forgetting
                # This maintains the interpretation of variance as E[X^2] - E[X]^2
                diff_old = X[i, self._selection] - old_m
                diff_new = X[i, self._selection] - self.m

                # Update M (sum of squared deviations)
                self.M = self.M * (1 - self.forget) + weight * diff_old * diff_new

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


# %%
np.random.seed(42)

N = 15  # Number of samples
N_init = 1
assert N > N_init, "N must be greater than N_init for this test."

X = np.random.uniform(
    0,
    10,
    N,
).reshape(-1, 1)

W = np.random.uniform(
    0.1,
    1.0,
    (N, 1),
).reshape(-1, 1)  # Uniform weights for each sample

# %%
forget = 0.000001
os = OnlineScaler(
    forget=forget,
)
os.fit(X[0:N_init,], sample_weight=W[0:N_init,].flatten())
print(os.m)
print(np.average(X[0:N_init,], weights=W[0:N_init,]))

# %% Test the update method
for i in range(N_init, N):
    os.partial_fit(X[i,].reshape(-1, 1), sample_weight=W[i,])

    # For exponential forgetting, we need to compute the effective weights
    if True:
        # Compute weights with exponential decay
        effective_weights = []
        for j in range(i + 1):
            decay_factor = (1 - forget) ** (i - j)
            effective_weights.append(W[j, 0] * decay_factor)
        effective_weights = np.array(effective_weights)

        # Normalize weights
        true_mean = np.average(
            X[0 : (i + 1),], weights=effective_weights.reshape(-1, 1)
        )
        true_var = np.average(
            (X[0 : (i + 1),] - true_mean) ** 2, weights=effective_weights.reshape(-1, 1)
        )

    true_std = np.sqrt(true_var)

    mean_diff = round(abs(true_mean - os.m[0]), 8)
    var_diff = round(abs(true_var - os.v[0]), 8)
    std_diff = round(abs(true_std - np.sqrt(os.v[0])), 8)

    print(
        f"{i - N_init + 1:03d}: Mean diff = {mean_diff}, Var diff = {var_diff}, Std diff = {std_diff}"
    )

# %%
