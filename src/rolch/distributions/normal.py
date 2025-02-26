from typing import Tuple, Union

import numpy as np
import scipy.stats as st

from rolch.base import Distribution, LinkFunction
from rolch.link import IdentityLink, LogLink


class DistributionNormal(Distribution):
    """Corresponds to GAMLSS NO() and scipy.stats.norm()"""

    def __init__(
        self,
        loc_link: LinkFunction = IdentityLink(),
        scale_link: LinkFunction = LogLink(),
    ) -> None:
        self.loc_link: LinkFunction = loc_link
        self.scale_link: LinkFunction = scale_link
        self.links: list[LinkFunction] = [self.loc_link, self.scale_link]

    n_params = 2

    distribution_support = (-np.inf, np.inf)
    parameter_support = {0: (-np.inf, np.inf), 1: (np.nextafter(0, 1), np.inf)}

    def theta_to_params(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return mu, sigma

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (1 / sigma**2) * (y - mu)

        if param == 1:
            return ((y - mu) ** 2 - sigma**2) / (sigma**3)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -(1 / sigma**2)

        if param == 1:
            # SIGMA
            return -(2 / (sigma**2))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def link_function(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link(y)

    def link_inverse(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse(y)

    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse_derivative(y)

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Union[int, None] = None
    ) -> np.ndarray:
        if param == 0:
            return np.repeat(np.mean(y, axis=axis), y.shape[0])
        if param == 1:
            return np.repeat(np.std(y, axis=axis), y.shape[0])

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).cdf(y)

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).pdf(y)

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).ppf(q)

    def rvs(self, size: int, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).rvs((size, theta.shape[0])).T
