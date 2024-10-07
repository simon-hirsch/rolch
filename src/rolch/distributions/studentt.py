import numpy as np
import scipy.special as sp
import scipy.stats as st

from rolch.abc import Distribution
from rolch.link import IdentityLink, LogLink, LogShiftTwoLink


class DistributionT(Distribution):
    """Corresponds to GAMLSS TF() and scipy.stats.t()"""

    def __init__(
        self, loc_link=IdentityLink(), scale_link=LogLink(), tail_link=LogShiftTwoLink()
    ):
        self.n_params = 3
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.tail_link = tail_link
        self.links = [self.loc_link, self.scale_link, self.tail_link]

    def theta_to_params(self, theta):
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        return mu, sigma, nu

    def dl1_dp1(self, y, theta, param=0):
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

    def dl2_dp2(self, y, theta, param=0):
        mu, sigma, nu = self.theta_to_params(theta)
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

    def dl2_dpp(self, y, theta, params=(0, 1)):
        mu, sigma, nu = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            # d2l/(dm ds)
            return np.zeros_like(y)

        if sorted(params) == [0, 2]:
            # d2l/(dm dn)
            return np.zeros_like(y)

        if sorted(params) == [1, 2]:
            # d2l / (dm dn)
            return 2 / (sigma * (nu + 3) * (nu + 1))

    def link_function(self, y, param=0):
        return self.links[param].link(y)

    def link_inverse(self, y, param=0):
        return self.links[param].inverse(y)

    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse_derivative(y)

    def initial_values(self, y, param=0, axis=None):
        if param == 0:
            return y  # (y + np.mean(y, axis=None)) / 2
        if param == 1:
            return (
                np.repeat(np.std(y, axis=None), y.shape[0]) + np.abs(y - np.mean(y))
            ) / 2  #  np.repeat(np.std(y, axis=None), y.shape[0])
        if param == 2:
            return np.full_like(y, 10)

    def cdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        return st.t(nu, mu, sigma).cdf(y)

    def pdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        return st.t(nu, mu, sigma).pdf(y)

    def ppf(self, q, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        return st.t(nu, mu, sigma).ppf(q)

    def rvs(self, size, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        return st.t(nu, mu, sigma).rvs((size, theta.shape[0])).T
