from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy.stats as st


class ScipyMixin(ABC):

    @property
    @abstractmethod
    def scipy_dist(self) -> st.rv_continuous:
        """The names of the parameters in the scipy.stats distribution and the corresponding column in theta."""
        pass

    @property
    @abstractmethod
    def scipy_names(self) -> Tuple[str]:
        """The names of the parameters in the scipy.stats distribution and the corresponding column in theta."""
        pass

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        if not self.scipy_names:
            raise ValueError(
                f"{self.__class__.__name__} has no scipy_names defined. To use theta_to_scipy_params Please define them in the subclass. Or override this method in the subclass if there is no 1:1 mapping between theta columns and scipy params."
            )

        params = {}
        for idx, name in self.parameter_names.items():
            params[self.scipy_names[name]] = theta[:, idx]
        return params

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).cdf(y)

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).pdf(y)

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).ppf(q)

    def rvs(self, size: int, theta: np.ndarray) -> np.ndarray:
        return (
            self.scipy_dist(**self.theta_to_scipy_params(theta))
            .rvs((size, theta.shape[0]))
            .T
        )
