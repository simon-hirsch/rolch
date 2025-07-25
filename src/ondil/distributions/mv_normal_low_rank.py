from itertools import product
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, MultivariateDistributionMixin
from ..links import Identity, Log, MatrixDiag
from ..types import ParameterShapes


def batched_log_lilkelihood_normal_precision_low_rank(y, mu, mat_d, mat_v):
    """Fast evaluation of the batched log likelihood."""
    # k = y.shape[1]
    # cov = np.linalg.inv(mat_d + mat_v @ np.swapaxes(mat_v, -2, -1))
    # part1 = - k/2 * np.log(2 * np.pi)
    # part2 = - 1/2 * np.log(np.linalg.det(cov))
    # part3 = - 1 / 2 * np.sum((y - mu) * (np.linalg.inv(cov) @ (y - mu)[..., None]).squeeze(), 1)
    # return part1 + part2 + part3
    k = y.shape[1]
    precision = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    part1 = -k / 2 * np.log(2 * np.pi)
    part2 = 1 / 2 * np.log(np.linalg.det(precision))
    part3 = -1 / 2 * np.sum((y - mu) * (precision @ (y - mu)[..., None]).squeeze(), 1)
    return part1 + part2 + part3


def element_cov(mat_d, mat_v, i, j):
    unit = np.expand_dims(np.eye(mat_v.shape[-1]), 0)
    inner = np.swapaxes(mat_v, -1, -2) @ np.linalg.inv(mat_d) @ mat_v
    inner_inverse = np.linalg.inv(unit + inner)
    element = int(i == j) * 1 / mat_d[:, i, i] - np.sum(
        inner_inverse
        * (
            (mat_v[:, i, :] / np.expand_dims(mat_d[:, i, i], -1))[:, :, None]
            @ (mat_v[:, j, :] / np.expand_dims(mat_d[:, j, j], -1))[:, None, :]
        ),
        axis=(-2, -1),
    )
    return element


def partial1_mu_element(y, mat_mu, mat_d, mat_v, i):
    term1 = mat_d[:, i, i] * (y - mat_mu)[:, i]
    term2 = np.sum(
        mat_v[:, [i], :] * mat_v * np.expand_dims(y - mat_mu, -1), axis=(-2, -1)
    )
    # This is a slightly cleaner version of below
    # term2 = np.squeeze(mat_v[:, [i], :] @ mat_v.swapaxes(-1, -2) @ np.expand_dims((y - mat_mu), -1))
    return term1 + term2


def partial2_mu_element(y, mat_mu, mat_d, mat_v, i):
    # Diagonal elements of the inverse covariance matrix
    return -(mat_d[:, i, i] + np.sum(mat_v[:, i] ** 2, 1))


def partial1_D_element(y, mat_mu, mat_d, mat_v, i):
    omega = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    part_1 = 0.5 * np.linalg.inv(omega)[:, i, i]
    part_2 = -0.5 * (y[:, i] - mat_mu[:, i]) ** 2
    return part_1 + part_2


def partial2_D_element(y, mat_mu, mat_d, mat_v, i):
    omega = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    cov = np.linalg.inv(omega)
    return 0.5 * -cov[:, i, i] ** 2


def partial1_V_element(y, mat_mu, mat_d, mat_v, i, j):
    # TODO: Would be nice to calculate only the necessary rows
    # of OMEGA in the future maybe!

    # Derivation for part 2
    # zzT @ V
    # zzT[:, i, :] @ mat_v[:, :, j]
    # select the correct row of zzT before
    # sum(z * z[:, i], axis=-1)
    omega = mat_d + mat_v @ np.swapaxes(mat_v, -2, -1)
    part_1 = np.sum(np.linalg.inv(omega)[:, i, :] * mat_v[:, :, j], axis=1)
    part_2 = -np.sum(
        (y - mat_mu) * np.expand_dims((y[:, i] - mat_mu[:, i]), -1) * mat_v[:, :, j], -1
    )
    return part_1 + part_2


def partial2_V_element(y, mat_mu, mat_d, mat_v, i, j):
    d = mat_d.shape[1]
    omega = mat_d + mat_v @ mat_v.swapaxes(-1, -2)
    cov = np.linalg.inv(omega)
    sum1 = 0
    sum2 = 0
    for k, q in product(range(d), range(d)):
        sum1 += cov[:, i, i] * mat_v[:, q, j] * cov[:, q, k] * mat_v[:, k, j]
        sum2 += cov[:, i, q] * mat_v[:, q, j] * cov[:, i, k] * mat_v[:, k, j]
    term1 = cov[:, i, i] - sum1 - sum2
    term2 = -(y - mat_mu)[:, i] ** 2
    return term1 + term2


class MultivariateNormalInverseLowRank(MultivariateDistributionMixin, Distribution):

    corresponding_gamlss: str = None
    parameter_names = {0: "mu", 1: "diag_matrix", 2: "low_rank_matrix"}
    parameter_support = {0: (-np.inf, np.inf), 1: (0, np.inf), 2: (-np.inf, np.inf)}
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.multivariate_normal
    scipy_names = {"mu": "mean", "sigma": "cov"}
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.DIAGONAL_MATRIX,
        2: ParameterShapes.MATRIX,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link_1: LinkFunction = MatrixDiag(Log()),
        scale_link_2: LinkFunction = Identity(),
        rank: int = 3,
    ):
        super().__init__(
            links={
                0: loc_link,
                1: scale_link_1,
                2: scale_link_2,
            }
        )
        self.rank = rank
        self.is_multivariate = True
        self._adr_lower_diag = {0: False, 1: False, 2: False}
        self._regularization = "low_rank"  # or adr
        self._regularization_allowed = {0: False, 1: False, 2: True}
        self._scoring = "fisher"

    def fitted_elements(self, dim: int):
        return {0: dim, 1: dim, 2: dim * self.rank}

    @property
    def param_structure(self):
        return self._param_structure

    def index_flat_to_cube(self, k: int, d: int, param: int):
        if param == 0:
            return k
        if param == 1:
            i, j = np.diag_indices(d)
            return i[k], j[k]
        if param == 2:
            idx = [(j, i) for i, j in product(range(self.rank), range(d))]
            return idx[k][0], idx[k][1]

    def set_theta_element(
        self, theta: Dict, value: np.ndarray, param: int, k: int
    ) -> Dict:
        """Sets an element of theta for parameter param and place k.

        !!! Note
            This will mutate `theta`!

        Args:
            theta (Dict): Current fitted $\theta$
            value (np.ndarray): Value to set
            param (int): Distribution parameter
            k (int): Flat element index $k$

        Returns:
            Dict: Theta where element (param, k) is set to value.
        """
        if param == 0:
            theta[param][:, k] = value
        if (param == 1) | (param == 2):
            d = theta[0].shape[1]
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            theta[param][:, i, j] = value
        return theta

    def theta_to_params(self, theta: Dict[int, np.ndarray]):
        loc = theta[0]
        mat_d = theta[1]
        mat_v = theta[2]
        return loc, mat_d, mat_v

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map theta to `scipy` distribution parameters for the multivariate normal distribution.

        Args:
            theta (Dict[int, np.ndarray]): Fitted / predicted theta.

        Returns:
            Dict[str, np.ndarray]: Mapped predicted
        """
        out = {
            "mean": theta[0],
            "cov": np.linalg.inv(theta[1] + theta[2] @ theta[2].swapaxes(-1, -2)),
        }
        return out

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = False
    ):
        mu, mat_d, mat_v = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = partial1_mu_element(y, mu, mat_d, mat_v, i=k)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial1_D_element(y, mu, mat_d, mat_v, i=i)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial1_V_element(y, mu, mat_d, mat_v, i=i, j=j)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        mu, mat_d, mat_v = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = partial2_mu_element(y, mu, mat_d, mat_v, i=k)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial2_D_element(y, mu, mat_d, mat_v, i=i)
        if param == 2:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            deriv = partial2_V_element(y, mu, mat_d, mat_v, i=i, j=j)
        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-5)

        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link(y, i=i, j=j)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link_derivative(y, i=i, j=j)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link_second_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_link_second_derivative(y, i=i, j=j)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].inverse(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_inverse(y, i=i, j=j)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].inverse_derivative(y)
        if param == 1:
            i, j = self.index_flat_to_cube(k=k, d=d, param=param)
            return self.links[param].element_inverse_derivative(y, i=i, j=j)

    def initial_values(self, y: np.ndarray, param: int = 0):
        M = y.shape[0]
        if param == 0:
            return np.tile(np.mean(y, axis=0), (M, 1))
        if param == 1:
            mat_d = np.diag(1 / np.var(y, 0))
            return np.tile(mat_d, (M, 1, 1))
        if param == 2:
            M = y.shape[0]
            omega = np.linalg.inv(np.cov(y, rowvar=False))
            mat_d = np.diag(1 / np.var(y, 0))
            eig = np.linalg.eig(omega - mat_d)
            largest_ev = np.argsort(eig.eigenvalues)[-self.rank :][::-1]
            mat_v = eig.eigenvectors[:, largest_ev]
            return np.tile(mat_v, (M, 1, 1))

    def set_initial_guess(self, theta, param):
        return theta

    def cube_to_flat(self, x: np.ndarray, param: int):
        if param == 0:
            return x
        if param == 1:
            return np.copy(x.diagonal(axis1=1, axis2=2))
        if param == 2:
            return x.swapaxes(-1, -2).reshape((x.shape[0], np.prod(x.shape[1:])))

    def flat_to_cube(self, x: np.ndarray, param: int):
        if param == 0:
            return x
        if param == 1:
            d = x.shape[1]
            out = np.zeros((x.shape[0], d, d))
            out[:, np.arange(d), np.arange(d)] = x
            return out
        if param == 2:
            d = int(x.shape[1] // self.rank)
            return x.reshape((x.shape[0], self.rank, d)).transpose(0, 2, 1)

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        loc, mat_d, mat_v = self.theta_to_params(theta)
        return batched_log_lilkelihood_normal_precision_low_rank(
            y, loc, mat_d=mat_d, mat_v=mat_v
        )

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")
