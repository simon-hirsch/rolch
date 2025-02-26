from typing import List, Tuple

import numpy as np
import scipy.special as sp
import scipy.stats as st

from rolch.base import Distribution, LinkFunction
from rolch.link import IdentityLink, LogLink, LogShiftTwoLink


class DistributionT(Distribution):
    """Corresponds to GAMLSS TF() and scipy.stats.t()"""

    def __init__(
        self,
        loc_link: LinkFunction = IdentityLink(),
        scale_link: LinkFunction = LogLink(),
        tail_link: LinkFunction = LogShiftTwoLink(),
    ) -> None:
        self.loc_link: LinkFunction = loc_link
        self.scale_link: LinkFunction = scale_link
        self.tail_link: LinkFunction = tail_link
        self.links: List[LinkFunction] = [
            self.loc_link,
            self.scale_link,
            self.tail_link,
        ]
        self.scipy_dist: st.rv_continuous = st.t

    distribution_support = (-np.inf, np.inf)

    n_params: int = 3
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (np.nextafter(0, 1), np.inf),
    }
    scipy_parameters = {"loc": 0, "scale": 1, "df": 2}

    def theta_to_params(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        return mu, sigma, nu

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        if param == 0:
            # MU
            s2 = sigma**2
            dsq = (y - mu) ** 2 / s2
            omega = (nu + 1) / (nu + dsq)
            return (omega * (y - mu)) / s2

        if param == 1:
            # SIGMA
            dsq = (y - mu) ** 2 / sigma**2
            omega = (nu + 1) / (nu + dsq)
            return (omega * dsq - 1) / sigma

        if param == 2:
            # TAIL
            dsq = (y - mu) ** 2 / sigma**2
            omega = (nu + 1) / (nu + dsq)
            dsq3 = 1 + (dsq / nu)
            v2 = nu / 2
            v3 = (nu + 1) / 2
            return (
                -np.log(dsq3)
                + ((omega * dsq - 1) / nu)
                + sp.digamma(v3)
                - sp.digamma(v2)
            ) / 2

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma, nu = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -(nu + 1) / ((nu + 3) * sigma**2)

        if param == 1:
            # SIGMA
            return -(2 * nu) / ((nu + 3) * sigma**2)

        if param == 2:
            # TAIL
            nu = np.fmin(nu, 1e15)
            v2 = nu / 2
            v3 = (nu + 1) / 2
            out = (  ## Polygamma(1, x) is the same as trigamma(x) in R
                (sp.polygamma(1, v3) - sp.polygamma(1, v2))
                + (2 * (nu + 5)) / (nu * (nu + 1) * (nu + 3))
            ) / 4
            return np.clip(out, -np.inf, -1e-15)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            # d2l/(dm ds)
            return np.zeros_like(y)

        if sorted(params) == [0, 2]:
            # d2l/(dm dn)
            return np.zeros_like(y)

        if sorted(params) == [1, 2]:
            # d2l / (dm dn)
            _, sigma, nu = self.theta_to_params(theta)
            return 2 / (sigma * (nu + 3) * (nu + 1))

    def link_function(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link(y)

    def link_inverse(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse(y)

    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse_derivative(y)

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: int = None
    ) -> np.ndarray:
        if param == 0:
            return np.repeat(np.mean(y, axis=axis), y.shape[0])
        if param == 1:
            return np.repeat(np.std(y, axis=axis), y.shape[0])
        if param == 2:
            return np.full_like(y, 10)

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
