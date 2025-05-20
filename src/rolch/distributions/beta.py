
from typing import Optional, Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..link import LogitLink

SMALL_NUMBER = 1e-10
LARGE_NUMBER = 1e+10


class DistributionBeta(ScipyMixin, Distribution):
    """The Beta Distribution for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$
    f(y|\mu,\sigma)=\\frac{\Gamma(\frac{1 - \sigma^2}{\sigma^2})}
	{
	\Gamma(\\frac{\mu (1 - \sigma^2)}{\sigma^2})
	\Gamma(\\frac{(1 - \mu) (1 - \sigma^2)}{\sigma^2})}
	y^{\\frac{\mu (1 - \sigma^2)}{\sigma^2} - 1}
	(1-y)^{\\frac{(1 - \mu) (1 - \sigma^2)}{\sigma^2} - 1}
    $$



    with the location and shape parameters $\mu, \sigma > 0$.

    !!! Note
        The function is parameterized as GAMLSS' BE() distribution.

        This parameterization is different to the `scipy.stats.gamma(alpha, loc, scale)` parameterization.

        We can use `DistributionBeta().gamlss_to_scipy(mu, sigma)` to map the distribution parameters to scipy.

    The `scipy.stats.beta()` distribution is defined as:
    $$
    f(x, \\alpha, \\beta) = \\frac{\Gamma(\\alpha + \\beta) x^{\\alpha - 1} {(1 - x)}^{\beta - 1}}{\Gamma(\\alpha) \Gamma(\\beta)}
    $$

    with the paramters $\\alpha, \\beta >0$. The parameters can be mapped as follows:
    $$
    \\alpha = \mu (1 - \sigma^2) / \sigma^2 \Leftrightarrow \mu = \\alpha / (\\alpha + \\beta)
    $$
    and
    $$
    \\beta = (1 - \mu) (1 - \sigma^2)/ \sigma^2 \Leftrightarrow \sigma = \sqrt{((\\alpha + \\beta + 1) )}
    $$


    Args:
        loc_link (LinkFunction, optional): The link function for $\mu$. Defaults to  LogitLink()
        scale_link (LinkFunction, optional): The link function for $\sigma$. Defaults to LogitLink()
    """

    corresponding_gamlss: str = "BE"

    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: ( np.nextafter(0, 1), np.nextafter(1, 0)),
        1: (np.nextafter(0, 1), np.nextafter(1, 0) ),
    }
    distribution_support = (np.nextafter(0, 1), np.nextafter(1, 0) )
    # Scipy equivalent and parameter mapping rolch -> scipy
    scipy_dist = st.beta
    # Theta columns do not map 1:1 to scipy parameters for beta
    # So we have to overload theta_to_scipy_params
    scipy_names = {}

    def __init__(
        self,
        loc_link: LinkFunction = LogitLink(),   
        scale_link: LinkFunction = LogitLink(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link})

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters

        Returns:
            dict: Dict of (a, b, loc, scale) for scipy.stats.beta(a, b, loc, scale)
        """
        mu = theta[:, 0]
        sigma = theta[:, 1]
        alpha = mu * (1 - sigma**2) / sigma**2
        beta = (1 - mu) * (1 - sigma**2) / sigma**2
        params = {"a": alpha, "b": beta, "loc": 0, "scale": 1}
        return params

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            '''return ((1 - sigma**2) / sigma**2) * ( 
                -spc.digamma(alpha) + spc.digamma(beta) + 
                np.log(np.fmax(y, LOG_LOWER_BOUND)) - np.log(np.fmax(1-y, LOG_LOWER_BOUND))
                )'''
            return ((1 - sigma**2) / sigma**2) * ( 
                -spc.digamma(np.fmax(alpha, SMALL_NUMBER)) + spc.digamma(np.fmax(beta, SMALL_NUMBER)) + 
                np.log(np.fmax(y, LOG_LOWER_BOUND)) - np.log(np.fmax(1-y, LOG_LOWER_BOUND))
                )                                     ###beta dist with bounds on digamma 

        if param == 1:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            '''return -(2 / sigma**3) * ( 
                mu * ( -spc.digamma(alpha) + spc.digamma(alpha + beta) + np.log(y)) + (1 - mu) * ( 
                ( -spc.digamma(beta) + spc.digamma(alpha + beta) + np.log(1-y) ) ) 
                )'''                                    ##beta -- breaks without bounds
        
            return -(2 / sigma**3) * ( 
                mu * ( -spc.digamma(np.fmax(alpha, SMALL_NUMBER)) + spc.digamma(np.fmax(alpha + beta, SMALL_NUMBER)) + 
                np.log(np.fmax(y, LOG_LOWER_BOUND))) + (1 - mu) * ( 
                ( -spc.digamma(np.fmax(beta, SMALL_NUMBER)) + spc.digamma(np.fmax(alpha + beta, SMALL_NUMBER)) 
                + np.log(np.fmax(1-y, LOG_LOWER_BOUND)) ) ) 
                )
            

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # MU
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            #return - ( ( (1 - sigma**2)**2 ) / sigma**4 ) * ( 
                #spc.polygamma(1, alpha) + spc.polygamma(1, beta) )      ##breaks, needs bounds on polygamma
        
            return - ( ( (1 - sigma**2)**2 ) / sigma**4 ) * ( 
                spc.polygamma(1, np.fmax(alpha, SMALL_NUMBER)) + spc.polygamma(1, np.fmax(beta, SMALL_NUMBER)) ) 

        if param == 1:
            # SIGMA
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            safe_sigma = np.fmax(sigma, SMALL_NUMBER) 
            safe_alpha = np.fmax(alpha, SMALL_NUMBER)
            safe_beta = np.fmax(beta, SMALL_NUMBER)

            '''return - (4 / sigma**3) * ( mu**2 * spc.polygamma(1, alpha) + (1-mu)**2 * spc.polygamma(1, beta) -
                spc.polygamma(1, alpha + beta)
                )'''         ####breaks here for log link instead of logit 
        
            '''return - (4 / sigma**3) * ( mu**2 * spc.polygamma(1, np.fmax(alpha, SMALL_NUMBER)) + (1-mu)**2 * 
                spc.polygamma(1, np.fmax(beta, SMALL_NUMBER)) -
                spc.polygamma(1, np.fmax(alpha + beta, SMALL_NUMBER))
                )'''
            return - (4 / safe_sigma**3) * ( mu**2 * spc.polygamma(1, safe_alpha) + (1-mu)**2 * 
                spc.polygamma(1, safe_beta) -
                spc.polygamma(1, safe_alpha + safe_beta)
                )
            

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma = self.theta_to_params(theta)

        if sorted(params) == [0, 1]:
            alpha = mu * (1 - sigma**2) / sigma**2
            beta = (1 - mu) * (1 - sigma**2) / sigma**2

            safe_sigma = np.fmax(sigma, SMALL_NUMBER) 
            safe_alpha = np.fmax(alpha, SMALL_NUMBER)
            safe_beta = np.fmax(beta, SMALL_NUMBER)

            '''return ( 2*(1 - sigma**2) / sigma**5 ) * ( 
                mu*spc.polygamma(1, np.fmax(alpha, SMALL_NUMBER)) - (1 - mu) * 
                spc.polygamma(1, np.fmax(beta, SMALL_NUMBER))
                )'''
        
            return ( 2*(1 - safe_sigma**2) / safe_sigma**5 ) * ( 
                mu*spc.polygamma(1, safe_alpha) - (1 - mu) * 
                spc.polygamma(1, safe_beta)
                ) 

    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Optional[int | None] = None
    ) -> np.ndarray:
        if param == 0:
            return (np.repeat(np.mean(y, axis=axis), y.shape[0]) )/ 2 
        if param == 1:
            return np.repeat(0.5, y.shape[0])


