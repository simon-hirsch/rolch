from typing import Dict, Optional, Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..link import IdentityLink, LogLink


class DistributionNormal(ScipyMixin, Distribution):
    """Corresponds to GAMLSS NO() and scipy.stats.norm()"""

    corresponding_gamlss: str = "NO"

    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {0: (-np.inf, np.inf), 1: (np.nextafter(0, 1), np.inf)}
    distribution_support = (-np.inf, np.inf)

    # Scipy equivalent and parameter mapping rolch -> scipy
    scipy_dist = st.norm
    scipy_names = {"mu": "loc", "sigma": "scale"}

    def __init__(
        self,
        loc_link: LinkFunction = IdentityLink(),
        scale_link: LinkFunction = LogLink(),
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

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

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Optional[int | None] = None
    ) -> np.ndarray:
        if param == 0:
            return np.repeat(np.mean(y, axis=axis), y.shape[0])
        if param == 1:
            return np.repeat(np.std(y, axis=axis), y.shape[0])
