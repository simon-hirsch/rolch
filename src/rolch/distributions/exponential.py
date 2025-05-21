from typing import Optional, Tuple
import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..link import LogLink, IdentityLink


class DistributionExponential(ScipyMixin, Distribution):
    """
    The Exponential distribution parameterized by mean (mu).

    PDF:
    $$
    f(y|\mu) = \frac{1}{\mu} \exp\left(-\frac{y}{\mu}\right)
    $$

    This corresponds to EXP() in GAMLSS where:
    - mu > 0
    - y > 0

    Link: mu.link = "log" by default
    """

    corresponding_gamlss: str = "EXP"
    parameter_names = {0: "mu"}
    parameter_support = {0: (np.nextafter(0, 1), np.inf)}
    distribution_support = (np.nextafter(0, 1), np.inf)
    scipy_dist = st.expon
    scipy_names = {"mu": "scale"}  # scipy uses scale = mu

    def __init__(self, mu_link: LinkFunction = LogLink()) -> None:
        super().__init__(links={0: mu_link})

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        mu = theta[:, 0]
        return {"scale": mu}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu = theta[:, 0]
        return (y - mu) / mu**2

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu = theta[:, 0]
        return -1 / mu**2

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        return np.zeros_like(y)

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Optional[int] = None
    ) -> np.ndarray:
        return (y + np.mean(y, axis=axis)) / 2
