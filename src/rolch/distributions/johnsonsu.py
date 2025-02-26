from typing import Tuple

import numpy as np
import scipy.stats as st

from rolch.base import Distribution, LinkFunction
from rolch.link import IdentityLink, LogLink


class DistributionJSU(Distribution):
    """
    Corresponds to GAMLSS JSUo() and scipy.stats.johnsonsu()

    Distribution parameters:
    0 : Location
    1 : Scale (close to standard deviation)
    2 : Skewness
    3 : Tail behaviour
    """

    def __init__(
        self,
        loc_link: LinkFunction = IdentityLink(),
        scale_link: LinkFunction = LogLink(),
        shape_link: LinkFunction = IdentityLink(),
        tail_link: LinkFunction = LogLink(),
    ) -> None:
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.shape_link = shape_link
        self.tail_link = tail_link
        self.links = [
            self.loc_link,
            self.scale_link,
            self.shape_link,  # skew
            self.tail_link,  # tail
        ]

    n_params = 4

    distribution_support = (-np.inf, np.inf)
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (-np.inf, np.inf),
        3: (np.nextafter(0, 1), np.inf),
    }

    def theta_to_params(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        tau = theta[:, 3]
        return mu, sigma, nu, tau

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)

        if param == 0:
            # MU
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            return dldm

        if param == 1:
            # SIGMA
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            return dldd

        if param == 2:
            # nu
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldv = -r
            return dldv

        if param == 3:
            # tau
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldt = (1 + r * nu - r * r) / tau
            return dldt

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if param == 0:
            # MU
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldm2 = -dldm * dldm
            return d2ldm2

        if param == 1:
            # SIGMA
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldd2 = -(dldd * dldd)
            return d2ldd2

        if param == 2:
            # TAIL
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            d2ldv2 = -(r * r)
            return d2ldv2

        if param == 3:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldt = (1 + r * nu - r * r) / tau
            d2ldt2 = -dldt * dldt
            return d2ldt2

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldmdd = -(dldm * dldd)
            return d2ldmdd

        if sorted(params) == [0, 2]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldv = -r
            d2ldmdv = -(dldm * dldv)
            return d2ldmdv

        if sorted(params) == [0, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldt = (1 + r * nu - r * r) / tau
            d2ldmdt = -(dldm * dldt)
            return d2ldmdt

        if sorted(params) == [1, 2]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            dldv = -r
            d2ldddv = -(dldd * dldv)
            return d2ldddv

        if sorted(params) == [1, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            dldt = (1 + r * nu - r * r) / tau
            d2ldddt = -(dldd * dldt)
            return d2ldddt

        if sorted(params) == [2, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldv = -r
            dldt = (1 + r * nu - r * r) / tau
            d2ldvdt = -(dldv * dldt)
            return d2ldvdt

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
            return np.full_like(y, 0)
        if param == 3:
            return np.full_like(y, 10)

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        return st.johnsonsu(
            loc=mu,
            scale=sigma,
            a=nu,
            b=tau,
        ).cdf(y)

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        return st.johnsonsu(
            loc=mu,
            scale=sigma,
            a=nu,
            b=tau,
        ).pdf(y)

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        return st.johnsonsu(
            loc=mu,
            scale=sigma,
            a=nu,
            b=tau,
        ).ppf(q)

    def rvs(self, size: int, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        return (
            st.johnsonsu(
                loc=mu,
                scale=sigma,
                a=nu,
                b=tau,
            )
            .rvs((size, theta.shape[0]))
            .T
        )
