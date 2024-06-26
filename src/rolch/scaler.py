import numpy as np

from rolch.utils import (
    calculate_asymptotic_training_length,
    calculate_effective_training_length,
)


class OnlineScaler:
    def __init__(
        self, forget: float = 0.0, do_scale: bool = True, intercept: bool = True
    ):
        # member variables
        self.do_scale = do_scale
        self.intercept = intercept
        self.forget = forget
        self.to_scale = None
        self.k = 0
        self.m = 0
        self.M = 0
        self.v = 0
        self.n_asymmptotic = calculate_asymptotic_training_length(self.forget)

    def fit(self, x_init: np.ndarray):
        if self.do_scale:
            self.to_scale = np.arange(self.intercept, x_init.shape[1], 1, dtype=int)

            self.k = x_init.shape[0]
            # Calculate the mean of each column of x_init and assing it to self.m
            self.m = np.mean(x_init[:, self.to_scale], axis=0)
            # Calculate the variance of each column of x_init and assing it to self.v
            self.v = np.var(x_init[:, self.to_scale], axis=0)
            self.M = self.v * self.k

    def partial_fit(self, x_new: np.ndarray):
        # Loop over all rows of x_new
        if self.do_scale:
            for i in range(x_new.shape[0]):
                self.k += 1
                n_seen = calculate_effective_training_length(self.forget, self.k)

                forget_scaled = self.forget * np.maximum(
                    self.n_asymmptotic / n_seen, 1.0
                )

                diff = x_new[i, self.to_scale] - self.m
                incr = forget_scaled * diff

                if forget_scaled > 0:
                    self.m += incr
                    self.v = (1 - forget_scaled) * (self.v + forget_scaled * diff**2)
                else:
                    self.m += diff / self.k
                    self.M += diff * (x_new[i, self.to_scale] - self.m)
                    self.v = self.M / self.k
        else:
            pass

    def transform(self, x: np.ndarray):
        out = np.copy(x)
        if self.do_scale:
            out[:, self.to_scale] = (x[:, self.to_scale] - self.m) / np.sqrt(self.v)
            return out
        else:
            return x

    def inverse_transform(self, x: np.ndarray):
        out = np.copy(x)
        if self.do_scale:
            out[:, self.to_scale] = x[:, self.to_scale] * np.sqrt(self.v) + self.m
            return out
        else:
            return x
