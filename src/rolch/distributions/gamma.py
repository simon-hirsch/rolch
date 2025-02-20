from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from rolch.base import Distribution, LinkFunction
from rolch.link import LogLink


class DistributionGamma(Distribution):
    """The Gamma Distribution for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$
    f(y|\mu,\sigma)=\\frac{y^{(1/\sigma^2-1)}\exp[-y/(\sigma^2 \mu)]}{(\sigma^2 \mu)^{(1/\sigma^2)} \Gamma(1/\sigma^2)}
    $$

    with the location and shape parameters $\mu, \sigma > 0$.

    !!! Note
        The function is parameterized as GAMLSS' GA() distribution.

        This parameterization is different to the `scipy.stats.gamma(alpha, loc, scale)` parameterization.

        We can use `DistributionGamma().gamlss_to_scipy(mu, sigma)` to map the distribution parameters to scipy.

    The `scipy.stats.gamma()` distribution is defined as:
    $$
    f(x, \\alpha, \\beta) = \\frac{\\beta^\\alpha x^{\\alpha - 1} \exp[-\\beta x]}{\Gamma(\\alpha)}
    $$
    with the paramters $\\alpha, \\beta >0$. The parameters can be mapped as follows:
    $$
    \\alpha = 1/\sigma^2 \Leftrightarrow \sigma = \sqrt{1 / \\alpha}
    $$
    and
    $$
    \\beta = 1/(\sigma^2\mu).
    $$

    Args:
        loc_link (LinkFunction, optional): The link function for $\mu$. Defaults to LogLink().
        scale_link (LinkFunction, optional): The link function for $\sigma$. Defaults to LogLink().
    """

    def __init__(
        self,
        loc_link: LinkFunction = LogLink(),
        scale_link: LinkFunction = LogLink(),
    ):
        self.loc_link: LinkFunction = loc_link
        self.scale_link: LinkFunction = scale_link
        self.links: dict[int, LinkFunction] = {0: self.loc_link, 1: self.scale_link}
        self.n_params: int = 2
        self.corresponding_gamlss: str = "GA"
        self.scipy_dist: st.rv_continuous = st.gamma

    def theta_to_params(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return mu, sigma

    @staticmethod
    def gamlss_to_scipy(
        mu: np.ndarray, sigma: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            mu (np.ndarray): mu parameter
            sigma (np.ndarray): sigma parameter

        Returns:
            tuple: Tuple of (alpha, loc, scale) for scipy.stats.gamma(alpha, loc, scale)
        """
        alpha = 1 / sigma**2
        beta = 1 / (sigma**2 * mu)
        loc = 0
        scale = 1 / beta
        return alpha, loc, scale

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (y - mu) / ((sigma**2) * (mu**2))

        if param == 1:
            return (2 / sigma**3) * (
                (y / mu)
                - np.log(y)
                + np.log(mu)
                + np.log(sigma**2)
                - 1
                + spc.digamma(1 / (sigma**2))
            )

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -1 / ((sigma**2) * (mu**2))

        if param == 1:
            # SIGMA
            return (4 / sigma**4) - (4 / sigma**6) * spc.polygamma(1, (1 / sigma**2))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
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
        self, y: np.ndarray, param: int = 0, axis: int = None
    ) -> np.ndarray:
        if param == 0:
            return np.repeat(np.mean(y, axis=None), y.shape[0])
        if param == 1:
            return np.ones_like(y)

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu, sigma)).cdf(y)

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu, sigma)).pdf(y)

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(*self.gamlss_to_scipy(mu, sigma)).ppf(q)

    def rvs(self, size: int, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        return (
            self.scipy_dist(*self.gamlss_to_scipy(mu, sigma))
            .rvs((size, theta.shape[0]))
            .T
        )
