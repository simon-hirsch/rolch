from typing import Optional, Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..link import LogLink, LogitLink


class DistributionBeta(ScipyMixin, Distribution):
    """The Beta Distribution for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$
    f(y|\mu,\sigma)=\\frac{y^{(\mu\sigma-1)}(1-y)^{((1-\mu)\sigma-1)}}{B(\mu\sigma,(1-\mu)\sigma)}
    $$

    with the location and shape parameters $0 < \mu < 1$ and $\sigma > 0$.

    Args:
        loc_link (LinkFunction, optional): The link function for $\mu$. Defaults to LogitLink().
        scale_link (LinkFunction, optional): The link function for $\sigma$. Defaults to LogLink().
    """

    corresponding_gamlss: str = "BE"

    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (0, 1),
        1: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (0, 1)
    # Scipy equivalent and parameter mapping rolch -> scipy
    scipy_dist = st.beta
    scipy_names = {"mu": "a", "sigma": "b"}

    def __init__(
        self,
        loc_link: LinkFunction = LogitLink(),
        scale_link: LinkFunction = LogLink(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link})

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters

        Returns:
            dict: Dict of (a, b) for scipy.stats.beta(a, b)
        """
        mu = theta[:, 0]
        sigma = theta[:, 1]
        params = {"a": mu * sigma, "b": (1 - mu) * sigma}
        return params

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (y - mu) / (mu * (1 - mu))

        if param == 1:
            return np.log(y / (1 - y)) - spc.digamma(mu * sigma) + spc.digamma((1 - mu) * sigma)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            return -1 / (mu * (1 - mu))

        if param == 1:
            return -spc.polygamma(1, mu * sigma) - spc.polygamma(1, (1 - mu) * sigma)

    def dl3_dp3(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return 2 / (mu * (1 - mu))

        if param == 1:
            return spc.polygamma(2, mu * sigma) + spc.polygamma(2, (1 - mu) * sigma)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Optional[int | None] = None
    ) -> np.ndarray:
        if param == 0:
            return np.repeat(np.mean(y, axis=axis), y.shape[0])
        if param == 1:
            return np.ones_like(y)
