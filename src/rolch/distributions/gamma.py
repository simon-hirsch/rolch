import numpy as np
import scipy.special as spc
import scipy.stats as st

from rolch.abc import Distribution
from rolch.link import LogLink


class DistributionGamma(Distribution):
    """Corresponds to GAMLSS GA() and scipy.stats.gamma()"""

    def __init__(self, loc_link=LogLink(), scale_link=LogLink()):
        self.n_params = 2
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.links = [self.loc_link, self.scale_link]
        self.corresponding_gamlss = "GA"
        self.scipy_dist = st.gamma

    def theta_to_params(self, theta):
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return mu, sigma

    def dl1_dp1(self, y, theta, param=0):
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return ((y - mu) / ((sigma ^ 2) * (mu ^ 2)),)

        if param == 1:
            return (2 / sigma ^ 3) * (
                (y / mu)
                - np.log(y)
                + np.log(mu)
                + np.log(sigma ^ 2)
                - 1
                + spc.digamma(1 / (sigma ^ 2))
            )

    def dl2_dp2(self, y, theta, param=0):
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            return -1 / ((sigma ^ 2) * (mu ^ 2))

        if param == 1:
            # SIGMA
            # R Code (4/sigma^4)-(4/sigma^6)*trigamma((1/sigma^2))
            return (4 / sigma ^ 4) - (4 / sigma ^ 6) * spc.polygama(2, (1 / sigma ^ 2))

    def dl2_dpp(self, y, theta, params=(0, 1)):
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def link_function(self, y, param=0):
        return self.links[param].link(y)

    def link_inverse(self, y, param=0):
        return self.links[param].inverse(y)

    def link_derivative(self, y, param=0):
        return self.links[param].derivative(y)

    def initial_values(self, y, param=0, axis=None):
        if param == 0:
            return (y + np.mean(y, axis=None)) / 2
        if param == 1:
            return np.ones_like(y)

    # TODO: This can be factored to the base class
    def cdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(mu, sigma).cdf(y)

    def pdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(mu, sigma).pdf(y)

    def ppf(self, q, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(mu, sigma).ppf(q)

    def rvs(self, size, theta):
        mu, sigma = self.theta_to_params(theta)
        return self.scipy_dist(mu, sigma).rvs((size, theta.shape[0])).T
