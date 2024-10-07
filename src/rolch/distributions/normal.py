import numpy as np
import scipy.stats as st

from rolch.abc import Distribution
from rolch.link import IdentityLink, LogLink


class DistributionNormal(Distribution):
    """Corresponds to GAMLSS NO() and scipy.stats.norm()"""

    def __init__(self, loc_link=IdentityLink(), scale_link=LogLink()):
        self.n_params = 2
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.links = [self.loc_link, self.scale_link]

    def theta_to_params(self, theta):
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return mu, sigma

    def dl1_dp1(self, y, theta, param=0):
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (1 / sigma**2) * (y - mu)

        if param == 1:
            return ((y - mu) ** 2 - sigma**2) / (sigma**3)

    def dl2_dp2(self, y, theta, param=0):
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -(1 / sigma**2)

        if param == 1:
            # SIGMA
            return -(2 / (sigma**2))

    def dl2_dpp(self, y, theta, params=(0, 1)):
        mu, sigma = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

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
            return (y + np.mean(y, axis=None)) / 2
        if param == 1:
            return (
                np.repeat(np.std(y, axis=None), y.shape[0]) + np.abs(y - np.mean(y))
            ) / 2

    def cdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).cdf(y)

    def pdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).pdf(y)

    def ppf(self, q, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).ppf(q)

    def rvs(self, size, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm(mu, sigma).rvs((size, theta.shape[0])).T
