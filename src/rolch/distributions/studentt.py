from typing import Optional, Tuple

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..link import IdentityLink, LogLink, LogShiftTwoLink


class DistributionT(ScipyMixin, Distribution):
    """Corresponds to GAMLSS TF() and scipy.stats.t()"""

    corresponding_gamlss: str = "TF"

    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (-np.inf, np.inf)

    # Scipy distribution and parameter mapping rolch -> scipy
    scipy_dist = st.t
    scipy_names = {"mu": "loc", "sigma": "scale", "nu": "df"}

    def __init__(
        self,
        loc_link: LinkFunction = IdentityLink(),
        scale_link: LinkFunction = LogLink(),
        tail_link: LinkFunction = LogShiftTwoLink(),
        dof_guesstimate: float = 10,
    ) -> None:
        """The student-t distribution. The PDF of the student-t distribution is given by:
        $$
            f(y; \\mu, \\sigma, \\nu) = \\frac{\\Gamma((\\nu + 1) / 2)}{\\Gamma(\\nu / 2) \\sqrt{\\nu \\pi \\sigma^2}} \\left(1 + \\frac{(y - \\mu)^2}{\\nu \\sigma^2}\\right)^{-(\\nu + 1) / 2}
        $$
        where $\\mu$ is the location parameter, $\\sigma$ is the scale parameter and $\\nu$ is the shape parameter.

        Args:
            loc_link (LinkFunction, optional): Link for the location parameter. Defaults to IdentityLink().
            scale_link (LinkFunction, optional): Link for the scale parameter. Defaults to LogLink().
            tail_link (LinkFunction, optional): Link for the tail parameter. Defaults to LogShiftTwoLink().
            dof_guesstimate (float, optional): Initial guess for the (conditional) degrees of freedom. Defaults to 10.
        """
        self.dof_guesstimate = dof_guesstimate
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: tail_link,
            }
        )

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

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Optional[int | None] = None
    ) -> np.ndarray:
        if param == 0:
            return np.repeat(np.mean(y, axis=axis), y.shape[0])
        if param == 1:
            return np.repeat(np.std(y, axis=axis), y.shape[0])
        if param == 2:
            return np.full_like(y, self.dof_guesstimate)
