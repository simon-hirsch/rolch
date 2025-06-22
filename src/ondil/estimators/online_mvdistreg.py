import copy
import time
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numba as nb
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ..base import Distribution, OndilEstimatorMixin
from ..base.distribution import ParameterShapes
from ..coordinate_descent import online_coordinate_descent_path
from ..gram import (
    init_forget_vector,
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)
from ..scaler import OnlineScaler
from ..utils import calculate_effective_training_length, handle_param_dict

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def indices_along_diagonal(D: int) -> List:
    """This functions generates a list of indices that will go along
    the diagonal first and then go along the off-diagonals for a upper/lower
    triangular matrix.

    Args:
        D (int): Dimension of the matrix.

    Returns:
        List: List of indices
    """
    K = []
    for i in range(D):
        K.append(i)
        for j in range(D, i + 1, -1):
            K.append((K[-1] + j))
    return K


@nb.jit(
    [
        "float32[:, :](float32[:], float32[:], float32[:, :], boolean)",
        "float64[:, :](float64[:], float64[:], float64[:, :], boolean)",
    ],
    parallel=False,
)
def fast_vectorized_interpolate(x, xp, fp, ascending=True):
    dim = fp.shape[1]
    out = np.zeros((x.shape[0], dim), dtype=fp.dtype)
    if not ascending:
        x_asc = x[::-1]
        xp_asc = xp[::-1]

    for i in range(dim):
        if ascending:
            out[:, i] = np.interp(
                x=x,
                xp=xp,
                fp=fp[:, i],
            )
        else:
            out[:, i] = np.interp(
                x=x_asc,
                xp=xp_asc,
                fp=fp[:, i][::-1],
            )[::-1]
    return out


def make_intercept(n_observations: int) -> np.ndarray:
    """Make the intercept series as N x 1 array.

    Args:
        y (np.ndarray): Response variable $Y$

    Returns:
        np.ndarray: Intercept array.
    """
    return np.ones((n_observations, 1))


def make_model_array(X, eq, fit_intercept):
    n = X.shape[0]

    # TODO: Check difference between np.array and list more explicitly?
    if isinstance(eq, str) and (eq == "intercept"):
        if not fit_intercept:
            raise ValueError(
                "fit_intercept[param] is false, but equation says intercept."
            )
        out = make_intercept(n_observations=n)
    else:
        if isinstance(eq, str) and (eq == "all"):
            if isinstance(X, np.ndarray):
                out = X
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                out = X.to_numpy()
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                out = X.to_numpy()
        elif isinstance(eq, np.ndarray) | isinstance(eq, list):
            if isinstance(X, np.ndarray):
                out = X[:, eq]
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                out = X.loc[:, eq]
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                out = X.select(eq).to_numpy()
        else:
            raise ValueError("Did not understand equation. Please check.")

        if fit_intercept:
            out = np.hstack((make_intercept(n), out))

    return out


def get_J_from_equation(self, X: np.ndarray):
    J = {}
    for p in range(self.distribution.n_params):
        if isinstance(self.equation[p], str):
            if self.equation[p] == "all":
                J[p] = X.shape[1] + int(self.fit_intercept[p])
            if self.equation[p] == "intercept":
                J[p] = 1
        elif isinstance(self.equation[p], np.ndarray) or isinstance(
            self.equation[p], list
        ):
            J[p] = len(self.equation[p]) + int(self.fit_intercept[p])
        else:
            raise ValueError("Something unexpected happened")
    return J


def information_criteria_likelihood(
    log_likelihood: Union[float, np.ndarray],
    n_observations: Union[float, np.ndarray],
    n_parameters: Union[float, np.ndarray],
    ic: Literal["aic", "aicc", "bic", "hqc"],
) -> np.ndarray:
    if ic == "aic":
        value = 2 * n_parameters - 2 * log_likelihood
    elif ic == "aicc":
        value = (
            2 * n_parameters
            - 2 * log_likelihood
            + (
                (2 * n_parameters**2 + 2 * n_parameters)
                / (n_observations - n_parameters - 1)
            )
        )
    elif ic == "bic":
        value = n_parameters * np.log(n_observations) - 2 * log_likelihood
    elif ic == "hqc":
        value = n_parameters * np.log(np.log(n_observations)) - 2 * log_likelihood
    else:
        raise ValueError("Did not recognize ic.")
    return value


@nb.njit()
def get_max_lambda(x_gram: np.ndarray, y_gram: np.ndarray, is_regularized: np.ndarray):
    if np.all(is_regularized):
        max_lambda = np.max(np.abs(y_gram))
    elif np.sum(~is_regularized) == 1:
        intercept = y_gram[is_regularized] / np.diag(x_gram)[~is_regularized]
        max_lambda = np.max(
            np.abs(y_gram.flatten() - x_gram[~is_regularized] * intercept)
        )
    else:
        raise NotImplementedError("Currently not implemented")
    return max_lambda


def get_adr_regularization_distance(d: int, lower_diag: bool = True):
    if lower_diag:
        j, i = np.triu_indices(d, k=0)
    else:
        i, j = np.triu_indices(d, k=0)
    distance = np.abs(i - j)
    return distance


def get_low_rank_regularization_distance(d, r):
    return np.concatenate([np.repeat(i + 1, d) for i in range(r)])


class MultivariateOnlineDistributionalRegressionADRPath(
    OndilEstimatorMixin, RegressorMixin, BaseEstimator
):

    def __init__(
        self,
        distribution: Distribution,
        equation: Dict,
        forget: float | Dict = 0.0,
        learning_rate: float = 0.0,
        fit_intercept: bool = True,
        scale_inputs: bool = True,
        verbose: int = 1,
        method: Literal["ols", "lasso"] | Dict[int, Literal["ols", "lasso"]] = "ols",
        ic: Literal["ll", "aic", "bic", "hqc", "max"] = "aic",
        iteration_along_diagonal: bool = False,
        approx_fast_model_selection: bool = True,
        max_regularisation_size: int | None = None,
        early_stopping: bool = True,
        early_stopping_criteria: Literal["ll", "aic", "bic", "hqc", "max"] = "aic",
        early_stopping_abs_tol: float = 0.001,
        early_stopping_rel_tol: float = 0.001,
        weight_delta: float | Dict[int, float] = 1.0,
        max_iterations_inner: int = 10,
        max_iterations_outer: int = 10,
        lambda_targeting: bool = False,
        lambda_n: int = 100,
        lambda_eps: float = 1e-4,
        dampen_estimation: bool | int = False,
        generic_scoring: bool = True,
        debug: bool = False,
        overshoot_correction: Dict | None = None,
    ):
        self.distribution = distribution

        # For simplicity
        self.handle_default(forget, 0.0, "forget")
        self.handle_default(method, "ols", "method")
        self.handle_default(fit_intercept, True, "fit_intercept")
        self.handle_default(dampen_estimation, int(False), "dampen_estimation")
        self.handle_default(overshoot_correction, None, "overshoot_correction")
        self.handle_default(weight_delta, 1.0, "weight_delta")

        self.method = method
        self.beta = {}
        self.beta_path = {}

        self.learning_rate = learning_rate
        self.equation = equation
        self.iteration_along_diagonal = iteration_along_diagonal

        # Early stopping
        self.max_regularisation_size = max_regularisation_size
        self.early_stopping = early_stopping
        self.early_stopping_criteria = early_stopping_criteria
        self.early_stopping_abs_tol = early_stopping_abs_tol
        self.early_stopping_rel_tol = early_stopping_rel_tol

        # TODO: This has currently no effect.
        # TODO: Allow for regularized intercepts.
        self.regularize_intercept = {0: False, 1: True}

        # Scaler
        self.scale_inputs = scale_inputs
        self.scaler = OnlineScaler(
            forget=self.learning_rate, to_scale=self.scale_inputs
        )

        # For LASSO
        self.lambda_targeting = lambda_targeting
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.ic = ic
        self.approx_fast_model_selection = approx_fast_model_selection

        # For (probably) faster + more stable estimation
        # self.dampen_estimation = int(dampen_estimation)

        # Iterations and other internal stuff
        self.max_iterations_outer = max_iterations_outer
        self.max_iterations_inner = max_iterations_inner
        self.rel_tol_inner = 1e-3
        self.abs_tol_inner = 1e-3
        self.rel_tol_outer = 1e-3
        self.abs_tol_outer = 1e-3

        self.generic_scoring = generic_scoring

        # For pretty printing.
        # TODO: This can be moved to top classes
        self._verbose_prefix = f"[{self.__class__.__name__}]"
        self._verbose_end = {1: "\n", 2: "\r", 3: "\r"}
        self.verbose = verbose
        self.debug = debug

    def print_message(self, level, string):
        if level <= self.verbose:
            print(
                self._verbose_prefix,
                string,
                end=self._verbose_end[min(self.verbose, level + 1)],
            )

    # Same for Univariate and Multivariate
    def handle_default(self, value, default, name):
        handle_param_dict(
            self,
            param=value,
            default=default,
            name=name,
            n_params=self.distribution.n_params,
        )

    def make_iteration_indices(self, param: int):

        if (
            self.distribution.parameter_shape
            in [
                ParameterShapes.SQUARE_MATRIX,
                ParameterShapes.LOWER_TRIANGULAR_MATRIX,
                ParameterShapes.UPPER_TRIANGULAR_MATRIX,
            ]
        ) & self.iteration_along_diagonal:
            index = np.arange(indices_along_diagonal(self.D[param]))
        else:
            index = np.arange(self.K[param])

        return index

    ## TODO: This needs to be handled by the estimation method!
    def make_gram(self, x: np.ndarray, w: np.ndarray, param: int) -> np.ndarray:
        """
        Make the Gram matrix.

        Args:
            x (np.ndarray): Covariate matrix.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Gram matrix.
        """
        if self.method[param] == "ols":
            return init_inverted_gram(X=x, w=w, forget=self.forget[param])
        elif self.method[param] == "lasso":
            return init_gram(X=x, w=w, forget=self.forget[param])

    def make_y_gram(
        self, x: np.ndarray, y: np.ndarray, w: np.ndarray, param: int
    ) -> np.ndarray:
        """
        Make the Gram matrix for the response variable.

        Args:
            x (np.ndarray): Covariate matrix.
            y (np.ndarray): Response variable.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Gram matrix for the response variable.
        """
        return init_y_gram(X=x, y=y, w=w, forget=self.forget[param])

    def update_gram(
        self, gram: np.ndarray, x: np.ndarray, w: np.ndarray, param: int
    ) -> np.ndarray:
        """
        Update the Gram matrix.

        Args:
            gram (np.ndarray): Current Gram matrix.
            x (np.ndarray): Covariate matrix.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Updated Gram matrix.
        """
        if self.method[param] == "ols":
            return update_inverted_gram(gram, X=x, w=w, forget=self.forget[param])
        if self.method[param] == "lasso":
            return update_gram(gram, X=x, w=w, forget=self.forget[param])

    def update_y_gram(
        self, gram: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, param: int
    ) -> np.ndarray:
        """
        Update the Gram matrix for the response variable.

        Args:
            gram (np.ndarray): Current Gram matrix for the response variable.
            x (np.ndarray): Covariate matrix.
            y (np.ndarray): Response variable.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Updated Gram matrix for the response variable.
        """
        return update_y_gram(gram, X=x, y=y, w=w, forget=self.forget[param])

    @staticmethod
    def _make_intercept(n_observations: int) -> np.ndarray:
        """Make the intercept series as N x 1 array.

        Args:
            y (np.ndarray): Response variable $Y$

        Returns:
            np.ndarray: Intercept array.
        """
        return np.ones((n_observations, 1))

    @staticmethod
    def _add_lags(
        y: np.ndarray, x: np.ndarray, lags: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add lagged variables to the response and covariate matrices.

        Args:
            y (np.ndarray): Response variable.
            x (np.ndarray): Covariate matrix.
            lags (Union[int, np.ndarray]): Number of lags to add.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the updated response and covariate matrices.
        """
        if lags == 0:
            return y, x

        if isinstance(lags, int):
            lags = np.arange(1, lags + 1, dtype=int)

        max_lag = np.max(lags)
        lagged = np.stack([np.roll(y, i) for i in lags], axis=1)[max_lag:, :]
        new_x = np.hstack((x, lagged))[max_lag:, :]
        new_y = y[max_lag:]
        return new_y, new_x

    def _make_matrix_or_intercept(
        self, n_observations: int, x: np.ndarray, add_intercept: bool, param: int
    ):
        """
        Make the covariate matrix or an intercept array based on the input.

        Args:
            y (np.ndarray): Response variable.
            x (np.ndarray): Covariate matrix.
            add_intercept (bool): Flag indicating whether to add an intercept.
            param (int): Parameter index.

        Returns:
            np.ndarray: Matrix or intercept array.
        """
        if x is None:
            return self._make_intercept(n_observations=n_observations)
        elif add_intercept[param]:
            return self._add_intercept(x=x, param=param)
        else:
            return x

    # Different UV - MV
    def _make_initial_theta(self, y: np.ndarray):
        theta = {
            a: {
                p: self.distribution.initial_values(y, p)
                for p in range(self.distribution.n_params)
            }
            for a in range(self.A)
        }
        # Handle AD-R Regularization
        for a in range(self.A):
            for p in range(self.distribution.n_params):
                if self.distribution._regularization_allowed[p]:
                    if self.distribution._regularization == "adr":
                        mask = (
                            self.adr_distance[p]
                            >= self.adr_mapping_index_to_max_distance[p][a]
                        )
                        regularized = self.distribution.cube_to_flat(theta[a][p], p)
                        regularized[:, mask] = regularized[:, mask] = 0
                        theta[a][p] = self.distribution.flat_to_cube(regularized, p)
                    if self.distribution._regularization == "low_rank":
                        mask = (
                            self.adr_distance[p]
                            >= self.adr_mapping_index_to_max_distance[p][a]
                        )

                        regularized = self.distribution.cube_to_flat(theta[a][p], p)
                        regularized[:, mask] = regularized[:, mask] = 0
                        theta[a][p] = self.distribution.flat_to_cube(regularized, p)

        return theta

    # Only MV
    def is_element_adr_regularized(self, p: int, k: int, a: int):
        if not self.distribution._regularization_allowed[p]:
            return False
        else:
            if self.distribution._regularization == "adr":
                return (
                    self.adr_distance[p][k]
                    >= self.adr_mapping_index_to_max_distance[p][a]
                )
            if self.distribution._regularization == "low_rank":
                return (
                    self.adr_distance[p][k]
                    >= self.adr_mapping_index_to_max_distance[p][a]
                )

    # Only MV
    def prepare_adr_regularization(self) -> None:
        self.adr_distance = {}
        self.adr_mapping_index_to_max_distance = {
            p: np.arange(1, self.A + 1)
            for p in range(self.distribution.n_params)
            if self.distribution._regularization_allowed[p]
        }
        for p in range(self.distribution.n_params):
            if self.distribution._regularization_allowed[p]:
                if self.distribution._regularization == "adr":
                    self.adr_distance[p] = get_adr_regularization_distance(
                        d=self.D, lower_diag=self.distribution._adr_lower_diag
                    )
                if self.distribution._regularization == "low_rank":
                    self.adr_distance[p] = get_low_rank_regularization_distance(
                        d=self.D, r=self.distribution.rank
                    )

    # Different UV-MV
    def get_number_of_covariates(self, X: np.ndarray):
        J = {}
        for p in range(self.distribution.n_params):
            J[p] = {}
            for k in range(self.K[p]):
                if isinstance(self.equation[p][k], str):
                    if self.equation[p][k] == "all":
                        J[p][k] = X.shape[1] + int(self.fit_intercept[p])
                    if self.equation[p][k] == "intercept":
                        J[p][k] = 1
                elif isinstance(self.equation[p][k], np.ndarray) or isinstance(
                    self.equation[p][k], list
                ):
                    J[p][k] = len(self.equation[p][k]) + int(self.fit_intercept[p])
                else:
                    raise ValueError("Something unexpected happened")
        return J

    def _get_next_adr_start_values(self, theta, p, a):
        mask = self.adr_distance[p] >= self.adr_mapping_index_to_max_distance[p][a - 1]
        prev_est = self.distribution.cube_to_flat(theta[a - 1][p], p)
        regularized = self.distribution.cube_to_flat(theta[a][p], p)
        regularized[:, ~mask] = prev_est[:, ~mask]
        start_value = self.distribution.flat_to_cube(regularized, p)
        return start_value

    # Different UV-MV
    def validate_equation(self, equation):
        if equation is None:
            warnings.warn(
                f"[{self.__class__.__name__}] "
                "Equation is not specified. "
                "Per default, will estimate the first distribution parameter by all covariates found in X. "
                "All other distribution parameters will be estimated by an intercept."
            )
            equation = {
                p: (
                    {k: "all" for k in range(self.K[p])}
                    if p == 0
                    else {k: "intercept" for k in range(self.K[p])}
                )
                for p in range(self.distribution.n_params)
            }
        else:
            for p in range(self.distribution.n_params):
                # Check that all distribution parameters are in the equation.
                # If not, add intercept.
                if p not in equation.keys():
                    print(
                        f"{self.__class__.__name__}",
                        f"Distribution parameter {p} is not in equation.",
                        "All elements of the parameter will be estimated by an intercept.",
                    )
                    equation[p] = {k: "intercept" for k in range(self.K[p])}

                else:
                    for k in range(self.K[p]):
                        if k not in equation[p].keys():
                            print(
                                f"{self._verbose_prefix}",
                                f"Distribution parameter {p}, element {k} is not in equation.",
                                "Element of the parameter will be estimated by an intercept.",
                            )
                            equation[p][k] = "intercept"

                    if not (
                        isinstance(equation[p][k], np.ndarray)
                        or (equation[p][k] in ["all", "intercept"])
                    ):
                        if not (
                            isinstance(equation[p], list) and (HAS_PANDAS | HAS_POLARS)
                        ):
                            raise ValueError(
                                "The equation should contain of either: \n"
                                " - a numpy array of dtype int, \n"
                                " - a list of string column names \n"
                                " - or the strings 'all' or 'intercept' \n"
                                f"you have passed {equation[p]} for the distribution parameter {p}."
                            )

        return equation

    # Different UV - MV
    def fit(self, X, y):

        # Set fixed values
        self.n_observations = y.shape[0]
        self.n_effective_training = calculate_effective_training_length(
            forget=self.learning_rate, n_obs=self.n_observations
        )
        self.D = y.shape[1]

        if self.distribution._regularization == "adr":
            self.A = self.D
        if self.distribution._regularization == "low_rank":
            self.A = self.distribution.rank + 1
        if self.max_regularisation_size is not None:
            self.A = np.fmin(self.A, self.max_regularisation_size)

        self.K = self.distribution.fitted_elements(self.D)
        self.equation = self.validate_equation(self.equation)
        self.J = self.get_number_of_covariates(X)

        # Without prior information, the optimal ADR regularization is
        # The largest model
        self.optimal_adr = self.A

        # Handle scaling
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X=X)

        # Some stuff
        self.is_regularized = {
            p: {k: np.repeat(True, self.J[p][k]) for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }

        self.iter_index = {
            p: self.make_iteration_indices(p) for p in range(self.distribution.n_params)
        }

        self.prepare_adr_regularization()
        theta = self._make_initial_theta(y)

        # Current information
        self.current_likelihood = np.array(
            [
                np.sum(self.distribution.log_likelihood(y=y, theta=theta[a]))
                for a in range(self.A)
            ]
        )
        self.model_selection = {
            p: {a: {} for a in range(self.A)} for p in range(self.distribution.n_params)
        }
        self.x_gram = {
            p: {
                k: np.empty((self.A, self.J[p][k], self.J[p][k]))
                for k in range(self.K[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.y_gram = {
            p: {k: np.empty((self.A, self.J[p][k])) for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }
        self.beta_path = {
            p: {
                k: np.zeros((self.A, self.lambda_n, self.J[p][k]))
                for k in range(self.K[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.beta = {
            p: {k: np.zeros((self.A, self.J[p][k])) for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }

        # Some information about the different iterations
        self.iteration_count = np.zeros(
            (self.max_iterations_outer, self.distribution.n_params, self.A), dtype=int
        )
        self.iteration_likelihood = np.zeros(
            (
                self.max_iterations_outer,
                self.max_iterations_inner,
                self.distribution.n_params,
                self.A,
            )
        )
        # For model selection
        self.lambda_min_current = {
            p: {k: {} for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }
        self.lambda_max_current = {
            p: {k: {} for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }
        self.lambda_opt_current = {
            p: {k: {} for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }
        self.lambda_path_current = {
            p: {k: {} for k in range(self.K[p])}
            for p in range(self.distribution.n_params)
        }
        self.lambda_min = {
            p: {
                k: np.zeros(
                    (self.max_iterations_outer, self.max_iterations_inner, self.A)
                )
                for k in range(self.K[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.lambda_max = {
            p: {
                k: np.zeros(
                    (self.max_iterations_outer, self.max_iterations_inner, self.A)
                )
                for k in range(self.K[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.lambda_opt = {
            p: {
                k: np.zeros(
                    (self.max_iterations_outer, self.max_iterations_inner, self.A)
                )
                for k in range(self.K[p])
            }
            for p in range(self.distribution.n_params)
        }
        self.lambda_grid = {
            p: {
                k: np.zeros(
                    (
                        self.max_iterations_outer,
                        self.max_iterations_inner,
                        self.A,
                        self.lambda_n,
                    )
                )
                for k in range(self.K[p])
            }
            for p in range(self.distribution.n_params)
        }

        self.model_selection_ll = {
            p: np.zeros((self.A, self.lambda_n, self.K[p]))
            for p in range(self.distribution.n_params)
        }

        # Call the fit
        self._outer_fit(X=X_scaled, y=y, theta=theta)

        if self.verbose > 0:  # Level 1 Message
            print(
                self._verbose_prefix,
                "Finished fitting distribution parameters.",
                end=self._verbose_end[1],
            )

    def _outer_fit(self, X, y, theta):

        adr_start = time.time()

        for a in range(self.A):
            adr_it_start = time.time()
            outer_start = time.time()

            global_old_likelihood = 0
            converged = False
            decreasing = False

            for outer_iteration in range(self.max_iterations_outer):
                outer_it_start = time.time()
                # Main outer loop
                for p in range(self.distribution.n_params):
                    message = (
                        f"Outer Iteration: {outer_iteration}, "
                        f"fitting Distribution parameter {p}, AD-r step {a}"
                    )
                    self.print_message(level=2, string=message)
                    theta = self._inner_fit(
                        X=X,
                        y=y,
                        theta=theta,
                        outer_iteration=outer_iteration,
                        p=p,
                        a=a,
                    )
                # Get the global likelihood
                global_likelihood = self.current_likelihood[a]

                converged, _ = self._check_outer_convergence(
                    global_old_likelihood, global_likelihood
                )

                # start value for the next AD-R step
                if (
                    converged
                    | decreasing
                    | (outer_iteration == self.max_iterations_outer - 1)
                ):
                    if a < (self.A - 1):
                        for p in range(self.distribution.n_params):
                            if not self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = copy.deepcopy(theta[a][p])
                            if self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = self._get_next_adr_start_values(
                                    theta, p, a + 1
                                )

                # Timings
                outer_it_end = time.time()
                outer_it_time = outer_it_end - outer_it_start
                outer_it_avg = (outer_it_end - outer_start) / (outer_iteration + 1)
                outer_pred_time = outer_it_time * (
                    self.max_iterations_outer - outer_iteration - 1
                )
                message = (
                    f"Last outer iteration {outer_iteration} took {round(outer_it_time, 1)} sec. "
                    f"Average outer iteration took {round(outer_it_avg, 1)} sec. "
                    f"Expected to be finished in max {round(outer_pred_time, 1)} sec. "
                )
                self.print_message(level=2, string=message)

                # End timing for the iteration
                adr_it_end = time.time()

                if converged:
                    break
                else:
                    global_old_likelihood = global_likelihood

            # Timings
            adr_it_last = adr_it_end - adr_it_start
            adr_it_avg = (adr_it_end - adr_start) / (a + 1)
            message = (
                f"Last ADR iteration {a} took {round(adr_it_last, 1)} sec. "
                f"Average ADR iteration took {round(adr_it_avg, 1)} sec. "
            )
            self.print_message(level=1, string=message)

            # Calculate the improvement
            if self.early_stopping_criteria == "ll":
                # Check the lilelihood for early stopping
                self.improvement_abs = -np.diff(self.current_likelihood)
                self.improvement_abs_scaled = (
                    -self.improvement_abs / self.n_effective_training
                )
                self.improvement_rel = (
                    self.improvement_abs / self.current_likelihood[-1:]
                )

            elif self.early_stopping_criteria in ["aic", "bic", "hqc", "max"]:
                self.early_stopping_n_params = np.array(
                    [self.count_nonzero_coef(self.beta, a) for a in range(self.A)]
                )
                self.early_stopping_ic = information_criteria_likelihood(
                    log_likelihood=self.current_likelihood,
                    n_observations=self.n_effective_training,
                    n_parameters=self.early_stopping_n_params,
                    ic=self.early_stopping_criteria,
                )

                self.improvement_abs = -np.diff(self.early_stopping_ic)
                self.improvement_abs_scaled = self.improvement_abs
                self.improvement_rel = (
                    self.improvement_abs / self.early_stopping_ic[-1:]
                )
            else:
                raise ValueError(
                    "Did not recognice criteria AD-r regularization stopping criteria."
                )

            if self.early_stopping and (a > 0) and (a < self.A - 1):
                # In the last step, it does not make sense to "early stop"

                if (
                    self.improvement_abs_scaled[a - 1] < self.early_stopping_abs_tol
                ) or (self.improvement_rel[a - 1] < self.early_stopping_rel_tol):
                    if self.verbose > 0:
                        print(
                            self._verbose_prefix,
                            f"Early stopping due to AD-r-regression. "
                            f"Last inrcease in r lead to relative improvement: {self.improvement_rel[a-1]}, scaled absolute improvement {self.improvement_abs_scaled[a-1]}",
                        )
                    if self.improvement_rel[a - 1] > 0:
                        self.optimal_adr = a
                    elif self.improvement_rel[a - 1] < 0:
                        self.optimal_adr = a - 1

                    # TODO: What to put in here?
                    self.improvement_abs[(a + 1) :] = 0
                    self.improvement_rel[(a + 1) :] = 0
                    self.improvement_abs_scaled[(a + 1) :] = 0
                    break
            else:
                # The largest theta is the optimal one
                # But might be overfit
                self.optimal_adr = a

        self.theta = theta
        self.optimal_theta = self.theta[self.optimal_adr]
        self.last_fit_adr_max = self.optimal_adr

    @staticmethod
    def count_nonzero_coef(beta, adr):
        non_zero = 0
        for p, coef in beta.items():
            non_zero += int(np.sum([np.sum(c[adr, :] != 0) for k, c in coef.items()]))
        return non_zero

    def _count_coef_to_be_fitted_param(self, adr: int, param: int, k: int):
        """Count the coefficients that should be fitted for this param"""
        idx_k = self.iter_index[param].tolist().index(k)
        if self.distribution._regularization_allowed[param]:
            count = 0
            for kk in self.iter_index[param][idx_k:]:
                count += int(not self.is_element_adr_regularized(param, kk, adr))
        else:
            count = len(self.iter_index[param][idx_k:])
        return count

    def count_coef_to_be_fitted(
        self, outer_iteration: int, inner_iteration: int, adr: int, param: int, k: int
    ):
        """Count all coefficients that should be fitted."""
        if (outer_iteration > 0) or (inner_iteration > 0):
            count = 0
        else:
            count = 0
            for p in range(param, self.distribution.n_params):
                count += self._count_coef_to_be_fitted_param(
                    adr=adr, param=p, k=k if p == param else 0
                )
        return count

    @staticmethod
    def _check_convergence(old_value, new_value, rel_tol, abs_tol) -> Tuple[bool, bool]:
        # Account for floating point acc
        if np.abs(new_value - old_value) / np.abs(new_value) < rel_tol:
            converged = True
            decreasing = False
        elif np.abs(new_value - old_value) < abs_tol:
            converged = True
            decreasing = False
        elif (new_value < old_value) & (np.abs(new_value - old_value) > 1e-10):
            converged = False
            decreasing = True
        else:
            converged = False
            decreasing = False
        return converged, decreasing

    def _check_inner_convergence(self, old_value, new_value) -> Tuple[bool, bool]:
        return self._check_convergence(
            old_value=old_value,
            new_value=new_value,
            rel_tol=self.rel_tol_inner,
            abs_tol=self.abs_tol_inner,
        )

    def _check_outer_convergence(self, old_value, new_value) -> Tuple[bool, bool]:
        return self._check_convergence(
            old_value=old_value,
            new_value=new_value,
            rel_tol=self.rel_tol_outer,
            abs_tol=self.abs_tol_outer,
        )

    def get_dampened_prediction(
        self,
        prediction: np.ndarray,
        eta: np.ndarray,
        outer_iteration: int,
        inner_iteration: int,
        param: int,
    ):
        if (outer_iteration == 0) & (inner_iteration < self.dampen_estimation[param]):
            out = (prediction * self.dampen_estimation[param] + eta) / (
                self.dampen_estimation[param] + 1
            )
        else:
            out = prediction
        return out

    def _inner_fit(self, y, X, theta, outer_iteration, a, p):

        converged = False
        decreasing = False
        old_likelihood = self.current_likelihood[a]

        weights_forget = init_forget_vector(
            forget=self.learning_rate,
            size=self.n_observations,
        )

        for inner_iteration in range(self.max_iterations_inner):

            # If the likelihood is at some point decreasing, we're breaking
            # Hence we need to store previous iteration values:
            if (inner_iteration > 0) | (outer_iteration > 0):
                prev_theta = copy.copy(theta)
                prev_x_gram = copy.copy(self.x_gram[p])
                prev_y_gram = copy.copy(self.y_gram[p])
                prev_model_selection = copy.copy(self.model_selection)
                prev_beta = copy.copy(self.beta)
                prev_beta_path = copy.copy(self.beta_path)

            # Iterate through all elements of the distribution parameter
            for k in self.iter_index[p]:
                if self.debug:
                    print("Fitting", outer_iteration, inner_iteration, p, k, a)

                if self.is_element_adr_regularized(p=p, k=k, a=a):
                    self.beta[p][k][a] = np.zeros(self.J[p][k])
                    self.beta_path[p][k][a] = np.zeros((self.lambda_n, self.J[p][k]))
                else:
                    if self.generic_scoring:
                        if (inner_iteration == 0) and (outer_iteration == 0):
                            theta[a] = self.distribution.set_initial_guess(theta[a], p)

                        eta = self.distribution.link_function(theta[a][p], p)
                        eta = self.distribution.cube_to_flat(eta, param=p)

                        # Derivatives wrt to the parameter
                        dl1dp1 = self.distribution.element_dl1_dp1(
                            y, theta=theta[a], param=p, k=k
                        )
                        dl2dp2 = self.distribution.element_dl2_dp2(
                            y, theta=theta[a], param=p, k=k, clip=False
                        )

                        dl1_link = self.distribution.link_function_derivative(
                            theta[a][p], p
                        )
                        dl2_link = self.distribution.link_function_second_derivative(
                            theta[a][p],
                            p,
                        )
                        dl1_link = self.distribution.cube_to_flat(dl1_link, param=p)
                        dl1_link = dl1_link[:, k]
                        dl2_link = self.distribution.cube_to_flat(dl2_link, param=p)
                        dl2_link = dl2_link[:, k]

                        dl1_deta1 = dl1dp1 * (1 / dl1_link)
                        dl2_deta2 = (
                            dl2dp2 * dl1_link - dl1dp1 * dl2_link
                        ) / dl1_link**3

                        wt = np.fmax(-dl2_deta2, 1e-10)
                        wv = eta[:, k] + dl1_deta1 / wt

                    else:
                        # TODO: This should be optimized at some future point.
                        # Still need to think about a good setup for the link
                        # and the flat-cube problem
                        eta = self.distribution.link_function(theta[a][p], p)
                        dr = 1 / self.distribution.cube_to_flat(
                            self.distribution.link_inverse_derivative(eta, param=p),
                            param=p,
                        )
                        eta = self.distribution.cube_to_flat(eta, param=p)
                        dr = dr[:, k]

                        dl1dp1 = self.distribution.element_score(
                            y, theta=theta[a], param=p, k=k
                        )
                        dl2dp2 = self.distribution.element_hessian(
                            y, theta=theta[a], param=p, k=k
                        )

                        if self.distribution._scoring == "fisher":
                            # For Fisher scoring,
                            # score == deriviative wrt parameter
                            # hessian = (expected) 2nd derivative wrt parameter
                            # "Pseudo Fisher Scoring"
                            wt = -(dl2dp2 / (dr * dr))
                            wv = eta[:, k] + dl1dp1 / (dr * wt)
                        elif self.distribution._scoring == "newton_rapson":
                            # For Newton-Rapson,
                            # score == derivative wrt predictor
                            # hessian == 2nd derivative wrt parameter
                            wt = -dl2dp2
                            wv = eta[:, k] + dl1dp1 / wt
                        else:
                            raise ValueError("Unknown scoring method.")

                    # Base Estimator abstracts the method away
                    # This will create inverted grams for OLS and
                    # non-inverted grams for LASSO
                    x = make_model_array(
                        X=X,
                        eq=self.equation[p][k],
                        fit_intercept=self.fit_intercept[p],
                    )
                    if self.debug:
                        print(
                            "Rank of X",
                            outer_iteration,
                            inner_iteration,
                            p,
                            k,
                            a,
                            np.linalg.matrix_rank(x),
                        )

                    self.x_gram[p][k][a] = self.make_gram(
                        x=x,
                        w=wt ** self.weight_delta[p],
                        param=p,
                    )
                    self.y_gram[p][k][a] = self.make_y_gram(
                        x=x,
                        y=wv,
                        w=wt ** self.weight_delta[p],
                        param=p,
                    ).squeeze()
                    if self.method[p] == "lasso":

                        if self.lambda_targeting and (
                            (inner_iteration + outer_iteration) > 0
                        ):
                            u_divisor = (inner_iteration + outer_iteration) / (
                                1 + (inner_iteration + outer_iteration)
                            )
                            l_divisor = 1 / (1 + (inner_iteration + outer_iteration))
                            # divisor = 1/2

                            lambda_max = (
                                self.lambda_max_current[p][k][a]
                                + self.lambda_opt_current[p][k][a]
                            ) * u_divisor
                            lambda_min = (
                                self.lambda_min_current[p][k][a]
                                + self.lambda_opt_current[p][k][a]
                            ) * l_divisor

                            if (
                                self.lambda_min_current[p][k][a]
                                == self.lambda_opt_current[p][k][a]
                            ):
                                lambda_min = lambda_max * self.lambda_eps
                            if (
                                self.lambda_max_current[p][k][a]
                                == self.lambda_opt_current[p][k][a]
                            ):
                                lambda_max = get_max_lambda(
                                    self.x_gram[p][k][a],
                                    self.y_gram[p][k][a],
                                    self.is_regularized[p][k],
                                )
                            lambda_path = np.geomspace(
                                lambda_max, lambda_min, self.lambda_n
                            )
                            start_beta_path = fast_vectorized_interpolate(
                                lambda_path,
                                self.lambda_path_current[p][k][a],
                                self.beta_path[p][k][max(a - 1, 0)],
                                ascending=True,
                            )

                        else:
                            lambda_max = get_max_lambda(
                                self.x_gram[p][k][a],
                                self.y_gram[p][k][a],
                                self.is_regularized[p][k],
                            )
                            lambda_min = lambda_max * self.lambda_eps
                            lambda_path = np.geomspace(
                                lambda_max, lambda_min, self.lambda_n
                            )

                            start_beta_path = self.beta_path[p][k][max(a - 1, 0)]

                        # For model selection
                        self.lambda_max[p][k][
                            outer_iteration, inner_iteration, a
                        ] = lambda_max
                        self.lambda_min[p][k][outer_iteration, inner_iteration, a] = (
                            lambda_max * self.lambda_eps
                        )
                        self.lambda_grid[p][k][
                            outer_iteration, inner_iteration, a
                        ] = lambda_path

                        self.lambda_max_current[p][k][a] = lambda_max
                        self.lambda_min_current[p][k][a] = lambda_min
                        self.lambda_path_current[p][k][a] = lambda_path

                        self.beta_path[p][k][a], _ = online_coordinate_descent_path(
                            x_gram=self.x_gram[p][k][a],
                            y_gram=self.y_gram[p][k][a],
                            beta_path=start_beta_path,
                            lambda_path=lambda_path,
                            is_regularized=self.is_regularized[p][k],
                            beta_lower_bound=np.repeat(-np.inf, self.J[p][k]),
                            beta_upper_bound=np.repeat(np.inf, self.J[p][k]),
                            which_start_value="average",
                            selection="cyclic",
                            tolerance=1e-4,
                            max_iterations=1000,
                            alpha=1,
                            early_stop=0,
                        )
                        eta_elem = x @ self.beta_path[p][k][a].T
                        theta_elem = self.distribution.element_link_inverse(
                            eta_elem, param=p, k=k, d=self.D
                        )

                        opt_ic = self._fit_model_selection(
                            y=y,
                            theta_fit=theta_elem,
                            theta=theta,
                            outer_iteration=outer_iteration,
                            inner_iteration=inner_iteration,
                            a=a,
                            k=k,
                            param=p,
                        )

                        self.lambda_opt[p][k][outer_iteration, inner_iteration, a] = (
                            lambda_path[opt_ic]
                        )
                        self.lambda_opt_current[p][k][a] = lambda_path[opt_ic]

                        # select optimal beta and theta
                        self.beta[p][k][a] = self.beta_path[p][k][a][opt_ic, :]
                        theta[a] = self.distribution.set_theta_element(
                            theta[a], theta_elem[:, opt_ic], param=p, k=k
                        )
                    elif self.method[p] == "ols":
                        self.beta[p][k][a] = (
                            self.x_gram[p][k][a] @ self.y_gram[p][k][a]
                        )  # ).squeeze()
                    else:
                        raise ValueError("Method not recognized")

                    eta[:, k] = self.get_dampened_prediction(
                        prediction=np.squeeze(x @ self.beta[p][k][a]),
                        eta=eta[:, k],
                        inner_iteration=inner_iteration,
                        outer_iteration=outer_iteration,
                        param=p,
                    )
                    theta[a][p] = self.distribution.link_inverse(
                        self.distribution.flat_to_cube(eta, param=p), param=p
                    )
                    if (self.overshoot_correction[p] is not None) and (
                        inner_iteration + outer_iteration < 1
                    ):
                        theta[a] = self.distribution.set_theta_element(
                            theta[a],
                            theta[a][p][:, k] + self.overshoot_correction[p][k],
                            param=p,
                            k=k,
                        )

                self.current_likelihood[a] = (
                    self.distribution.log_likelihood(y, theta=theta[a]) * weights_forget
                ).sum()

            ## Check the most important convergence measures here now
            if inner_iteration == (self.max_iterations_inner - 1):
                warnings.warn(
                    "Reached max inner iterations. Algorithm may or may not be converged."
                )
            self.iteration_count[outer_iteration, p, a] = inner_iteration
            self.iteration_likelihood[outer_iteration, inner_iteration, p, a] = (
                self.current_likelihood[a]
            )
            if self.verbose >= 2:  # Level 2 Message
                print(
                    self._verbose_prefix,
                    f"Outer iteration: {outer_iteration}, inner iteration {inner_iteration}, parameter {p}, AD-R {a}:",
                    f"current likelihood: {self.current_likelihood[a]},",
                    f"previous iteration likelihood {self.iteration_likelihood[outer_iteration, inner_iteration-1, p, a] if inner_iteration > 0 else self.current_likelihood[a]}",
                    end=self._verbose_end[self.verbose],
                )

            if (inner_iteration > 0) & (
                (inner_iteration > self.dampen_estimation[p]) | (outer_iteration > 0)
            ):
                converged, decreasing = self._check_inner_convergence(
                    old_value=old_likelihood, new_value=self.current_likelihood[a]
                )

                if converged:
                    break
                else:
                    # For the next iteration
                    old_likelihood = self.current_likelihood[a]

            # If the LL is decreasing, we're resetting to the previous iteration
            if ((outer_iteration > 0) | (inner_iteration > 1)) & decreasing:
                warnings.warn("Likelihood is decreasing. Breaking.")
                # Reset to values from the previous iteration
                theta = prev_theta
                self.model_selection = prev_model_selection
                self.x_gram[p] = prev_x_gram
                self.y_gram[p] = prev_y_gram
                self.beta = prev_beta
                self.beta_path = prev_beta_path
                self.current_likelihood[a] = old_likelihood
                break

        return theta

    def _fit_model_selection(
        self,
        y,
        theta_fit,
        theta,
        outer_iteration,
        inner_iteration,
        a: int,
        k: int,
        param: int,
    ):

        weights_forget = init_forget_vector(
            self.learning_rate,
            self.n_observations,
        )
        # Model selection
        if self.ic == "max":
            opt_ic = self.lambda_n - 1
        else:
            theta_ll = copy.deepcopy(theta[a])
            theta_elem_delta = np.diff(theta_fit, axis=1)
            theta_ms = self.distribution.set_theta_element(
                theta_ll, theta_fit[:, 0], param=param, k=k
            )
            approx_ll = np.sum(
                self.distribution.log_likelihood(y, theta_ll) * weights_forget
            )
            approx_ll = np.repeat(approx_ll, 100)
            for l_idx in range(1, self.lambda_n):
                theta_ms = self.distribution.set_theta_element(
                    theta_ll, theta_fit[:, l_idx], param=param, k=k
                )
                if self.approx_fast_model_selection:
                    approx_ll[l_idx] = approx_ll[l_idx - 1] + np.sum(
                        self.distribution.element_dl1_dp1(y, theta_ms, param=param, k=k)
                        * theta_elem_delta[:, l_idx - 1]
                        * weights_forget
                    )
                else:
                    approx_ll[l_idx] = np.sum(
                        self.distribution.log_likelihood(
                            y,
                            theta=theta_ms,
                        )
                        * weights_forget
                    )

            # Count number of nonzero coefficients
            # Subtract current beta if already fitted
            # If in the first iteration, add intercept
            # for all to-be-fitted parameters
            # that are not AD-R regularized
            nonzero = self.count_nonzero_coef(self.beta, adr=a)
            nonzero = nonzero + int(np.sum(self.beta[param][k][a, :] != 0))
            nonzero = nonzero + self.count_coef_to_be_fitted(
                outer_iteration, inner_iteration, adr=a, param=param, k=k
            )

            ic = information_criteria_likelihood(
                approx_ll, self.n_observations, nonzero, self.ic
            )
            self.model_selection[param][a][k] = {
                "ll": approx_ll,
                "non_zero": nonzero,
                "ic": ic,
            }
            opt_ic = np.argmin(ic)

        return opt_ic

    def _update_model_selection(
        self,
        y,
        theta_fit: np.ndarray,
        theta: Dict,
        outer_iteration: int,
        inner_iteration: int,
        a: int,
        k: int,
        param: int,
    ) -> int:
        """
        Update the model selection.

        Returns the optimal IC's index.
        """
        weights_forget = init_forget_vector(
            self.learning_rate,
            self.n_observations_step,
        )

        if self.ic == "max":
            opt_ic = self.lambda_n - 1
        else:
            # Model selection
            theta_ll = copy.deepcopy(theta[a])

            theta_elem_delta = np.diff(theta_fit, axis=1)
            theta_ms = self.distribution.set_theta_element(
                theta_ll, theta_fit[:, 0], param=param, k=k
            )
            approx_ll = np.sum(
                self.distribution.log_likelihood(y, theta_ll) * weights_forget
            )
            approx_ll = np.repeat(approx_ll, 100)
            for l_idx in range(1, self.lambda_n):
                theta_ms = self.distribution.set_theta_element(
                    theta_ll, theta_fit[:, l_idx], param=param, k=k
                )
                if self.approx_fast_model_selection:
                    approx_ll[l_idx] = approx_ll[l_idx - 1] + np.sum(
                        (
                            self.distribution.element_dl1_dp1(
                                y, theta_ms, param=param, k=k
                            )
                            * theta_elem_delta[:, l_idx - 1]
                        )
                        * weights_forget
                    )
                else:
                    approx_ll[l_idx] = np.sum(
                        self.distribution.log_likelihood(
                            y,
                            theta=theta_ms,
                        )
                        * weights_forget
                    )
            approx_ll = approx_ll + (
                self.model_selection_old[param][a][k]["ll"]
                * (1 - self.learning_rate) ** self.n_observations_step
            )
            # Count number of nonzero coefficients
            # Subtract current beta if already fitted
            # If in the first iteration, add intercept
            # for all to-be-fitted parameters
            # that are not AD-R regularized
            nonzero = self.count_nonzero_coef(self.beta, adr=a)
            nonzero = nonzero + int(np.sum(self.beta[param][k][a, :] != 0))
            nonzero = nonzero + self.count_coef_to_be_fitted(
                outer_iteration, inner_iteration, adr=a, param=param, k=k
            )

            ic = information_criteria_likelihood(
                approx_ll, self.n_observations, nonzero, self.ic
            )
            opt_ic = np.argmin(ic)

            self.model_selection[param][a][k] = {
                "ll": approx_ll,
                "non_zero": nonzero,
                "ic": ic,
            }

        return opt_ic

    # Different UV - MV
    def predict(
        self,
        X: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:

        if X is None:
            X_scaled = np.ones((1, 1))
            N = 1
            print(self._verbose_prefix, "X is None. Prediction will have length 1.")
        else:
            X_scaled = self.scaler.transform(X=X)
            N = X.shape[0]
        out = {}

        for p in range(self.distribution.n_params):
            array = np.zeros((N, self.K[p]))
            for k in range(self.K[p]):
                array[:, k] = (
                    make_model_array(
                        X=X_scaled,
                        eq=self.equation[p][k],
                        fit_intercept=self.fit_intercept[p],
                    )
                    @ self.beta[p][k][self.optimal_adr, :]
                ).squeeze()
            out[p] = self.distribution.flat_to_cube(array, p)
            out[p] = self.distribution.link_inverse(out[p], p)
        return out

    def predict_all_adr(
        self,
        X: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:

        if X is None:
            X_scaled = np.ones((1, 1))
            N = 1
            print(self._verbose_prefix, "X is None. Prediction will have length 1.")
        else:
            X_scaled = self.scaler.transform(X=X)
            N = X.shape[0]
        out = {}
        for a in range(self.A):
            out[a] = {}
            for p in range(self.distribution.n_params):
                array = np.zeros((N, self.K[p]))
                for k in range(self.K[p]):
                    array[:, k] = (
                        make_model_array(
                            X=X_scaled,
                            eq=self.equation[p][k],
                            fit_intercept=self.fit_intercept[p],
                        )
                        @ self.beta[p][k][a, :]
                    ).squeeze()
                out[a][p] = self.distribution.flat_to_cube(array, p)
                out[a][p] = self.distribution.link_inverse(out[a][p], p)

        return out

    # Different UV - MV
    def update(self, X, y) -> None:
        self.n_observations += y.shape[0]
        self.n_observations_step = y.shape[0]
        self.n_effective_training = calculate_effective_training_length(
            forget=self.learning_rate, n_obs=self.n_observations
        )
        theta = self.predict_all_adr(X)
        self.scaler.partial_fit(X=X)
        X_scaled = self.scaler.transform(X=X)

        self.x_gram_old = copy.deepcopy(self.x_gram)
        self.y_gram_old = copy.deepcopy(self.y_gram)
        self.model_selection_old = copy.deepcopy(self.model_selection)
        self.old_likelihood = self.current_likelihood + 0
        self.old_likelihood_discounted = (
            1 - self.learning_rate
        ) ** self.n_observations_step * self.old_likelihood
        self.current_likelihood = self.old_likelihood_discounted + np.array(
            [
                np.sum(
                    self.distribution.log_likelihood(y=y, theta=theta[a])
                    * init_forget_vector(self.learning_rate, y.shape[0])
                )
                for a in range(self.A)
            ]
        )
        self._outer_update(X=X_scaled, y=y, theta=theta)

    # Different UV - MV
    def _inner_update(self, X, y, theta, outer_iteration, a, p):

        converged = False
        decreasing = False
        old_likelihood = self.current_likelihood[a]
        weights_forget = init_forget_vector(
            self.learning_rate, self.n_observations_step
        )

        for inner_iteration in range(self.max_iterations_inner):

            # If the likelihood is at some point decreasing, we're breaking
            # Hence we need to store previous iteration values:
            if (inner_iteration > 0) | (outer_iteration > 0):
                prev_theta = copy.copy(theta)
                prev_x_gram = copy.copy(self.x_gram[p])
                prev_y_gram = copy.copy(self.y_gram[p])
                prev_model_selection = copy.copy(self.model_selection)
                prev_beta = copy.copy(self.beta)
                prev_beta_path = copy.copy(self.beta_path)

            for k in self.iter_index[p]:
                # Handle AD-R Regularization
                if self.is_element_adr_regularized(p=p, k=k, a=a):
                    self.beta[p][k][a] = np.zeros(self.J[p][k])
                    self.beta_path[p][k][a] = np.zeros((self.lambda_n, self.J[p][k]))

                else:
                    if self.generic_scoring:
                        eta = self.distribution.link_function(theta[a][p], p)
                        eta = self.distribution.cube_to_flat(eta, param=p)

                        # Derivatives wrt to the parameter
                        dl1dp1 = self.distribution.element_dl1_dp1(
                            y, theta=theta[a], param=p, k=k
                        )
                        dl2dp2 = self.distribution.element_dl2_dp2(
                            y, theta=theta[a], param=p, k=k
                        )

                        dl1_link = self.distribution.link_function_derivative(
                            theta[a][p], p
                        )
                        dl2_link = self.distribution.link_function_second_derivative(
                            theta[a][p], p
                        )
                        dl1_link = self.distribution.cube_to_flat(dl1_link, param=p)
                        dl1_link = dl1_link[:, k]
                        dl2_link = self.distribution.cube_to_flat(dl2_link, param=p)
                        dl2_link = dl2_link[:, k]

                        dl1_deta1 = dl1dp1 * (1 / dl1_link)
                        dl2_deta2 = (
                            dl2dp2 * dl1_link - dl1dp1 * dl2_link
                        ) / dl1_link**3

                        wt = np.fmax(-dl2_deta2, 1e-10)
                        wv = eta[:, k] + dl1_deta1 / wt

                    else:
                        # TODO: This should be optimized at some future point.
                        # Still need to think about a good setup for the link
                        # and the flat-cube problem
                        eta = self.distribution.link_function(theta[a][p], p)
                        dr = 1 / self.distribution.cube_to_flat(
                            self.distribution.link_inverse_derivative(eta, param=p),
                            param=p,
                        )
                        eta = self.distribution.cube_to_flat(eta, param=p)
                        dr = dr[:, k]

                        dl1dp1 = self.distribution.element_score(
                            y, theta=theta[a], param=p, k=k
                        )
                        dl2dp2 = self.distribution.element_hessian(
                            y, theta=theta[a], param=p, k=k
                        )
                        if self.distribution._scoring == "fisher":
                            wt = -(dl2dp2 / (dr * dr))
                            wv = eta[:, k] + dl1dp1 / (dr * wt)
                        elif self.distribution._scoring == "newton_rapson":
                            wt = -dl2dp2
                            wv = eta[:, k] + dl1dp1 / wt
                        else:
                            raise ValueError("Unknown scoring method.")

                    # Base Estimator abstracts the method away
                    # This will create inverted grams for OLS and
                    # non-inverted grams for LASSO
                    x = make_model_array(
                        X=X,
                        eq=self.equation[p][k],
                        fit_intercept=self.fit_intercept[p],
                    )
                    self.x_gram[p][k][a] = self.update_gram(
                        gram=self.x_gram_old[p][k][a],
                        x=x,
                        w=wt ** self.weight_delta[p],
                        param=p,
                    )
                    self.y_gram[p][k][a] = self.update_y_gram(
                        gram=np.expand_dims(self.y_gram_old[p][k][a], -1),
                        x=x,
                        y=wv,
                        w=wt ** self.weight_delta[p],
                        param=p,
                    ).squeeze()
                    if self.method[p] == "lasso":

                        if self.lambda_targeting and (
                            (inner_iteration + outer_iteration) > 0
                        ):
                            u_divisor = (inner_iteration + outer_iteration) / (
                                1 + (inner_iteration + outer_iteration)
                            )
                            l_divisor = 1 / (1 + (inner_iteration + outer_iteration))
                            # divisor = 1/2

                            lambda_max = (
                                self.lambda_max_current[p][k][a]
                                + self.lambda_opt_current[p][k][a]
                            ) * u_divisor
                            lambda_min = (
                                self.lambda_min_current[p][k][a]
                                + self.lambda_opt_current[p][k][a]
                            ) * l_divisor

                            if (
                                self.lambda_min_current[p][k][a]
                                == self.lambda_opt_current[p][k][a]
                            ):
                                lambda_min = lambda_max * self.lambda_eps
                            if (
                                self.lambda_max_current[p][k][a]
                                == self.lambda_opt_current[p][k][a]
                            ):
                                lambda_max = get_max_lambda(
                                    self.x_gram[p][k][a],
                                    self.y_gram[p][k][a],
                                    self.is_regularized[p][k],
                                )
                            lambda_path = np.geomspace(
                                lambda_max, lambda_min, self.lambda_n
                            )
                            start_beta_path = fast_vectorized_interpolate(
                                lambda_path,
                                self.lambda_path_current[p][k][a],
                                self.beta_path[p][k][max(a - 1, 0)],
                                ascending=True,
                            )

                        else:
                            lambda_max = get_max_lambda(
                                self.x_gram[p][k][a],
                                self.y_gram[p][k][a],
                                self.is_regularized[p][k],
                            )
                            lambda_min = lambda_max * self.lambda_eps
                            lambda_path = np.geomspace(
                                lambda_max, lambda_min, self.lambda_n
                            )

                            start_beta_path = self.beta_path[p][k][max(a - 1, 0)]

                        # For model selection
                        self.lambda_max[p][k][
                            outer_iteration, inner_iteration, a
                        ] = lambda_max
                        self.lambda_min[p][k][outer_iteration, inner_iteration, a] = (
                            lambda_max * self.lambda_eps
                        )
                        self.lambda_grid[p][k][
                            outer_iteration, inner_iteration, a
                        ] = lambda_path

                        self.lambda_max_current[p][k][a] = lambda_max
                        self.lambda_min_current[p][k][a] = lambda_min
                        self.lambda_path_current[p][k][a] = lambda_path

                        self.beta_path[p][k][a], _ = online_coordinate_descent_path(
                            x_gram=self.x_gram[p][k][a],
                            y_gram=self.y_gram[p][k][a],
                            beta_path=start_beta_path,
                            lambda_path=lambda_path,
                            is_regularized=self.is_regularized[p][k],
                            beta_lower_bound=np.repeat(-np.inf, self.J[p][k]),
                            beta_upper_bound=np.repeat(np.inf, self.J[p][k]),
                            which_start_value="average",
                            selection="cyclic",
                            tolerance=1e-4,
                            max_iterations=1000,
                            alpha=1,
                            early_stop=0,
                        )
                        eta_elem = x @ self.beta_path[p][k][a].T
                        theta_elem = self.distribution.element_link_inverse(
                            eta_elem, param=p, k=k, d=self.D
                        )

                        opt_ic = self._update_model_selection(
                            y=y,
                            theta=theta,
                            theta_fit=theta_elem,
                            outer_iteration=outer_iteration,
                            inner_iteration=inner_iteration,
                            param=p,
                            k=k,
                            a=a,
                        )
                        # Select the optimal beta
                        self.beta[p][k][a] = self.beta_path[p][k][a][opt_ic, :]
                        theta[a] = self.distribution.set_theta_element(
                            theta[a], theta_elem[:, opt_ic], param=p, k=k
                        )
                    elif self.method[p] == "ols":
                        self.beta[p][k][a] = self.x_gram[p][k][a] @ self.y_gram[p][k][a]
                    else:
                        raise ValueError("Method not recognized")

                    eta[:, k] = np.squeeze(x @ self.beta[p][k][a])
                    theta[a][p] = self.distribution.link_inverse(
                        self.distribution.flat_to_cube(eta, param=p), param=p
                    )

                self.current_likelihood[a] = (
                    np.sum(
                        self.distribution.log_likelihood(y, theta=theta[a])
                        * weights_forget
                    )
                    + self.old_likelihood_discounted[a]
                )

            self.iteration_count[outer_iteration, p] = inner_iteration
            self.iteration_likelihood[outer_iteration, inner_iteration, p, a] = (
                self.current_likelihood[a]
            )

            # Are we in the last iteration
            if inner_iteration == (self.max_iterations_inner - 1):
                warnings.warn(
                    "Reached max inner iterations. Algorithm may or may not be converged."
                )

            # Are we converged
            if inner_iteration > 0:
                converged, decreasing = self._check_inner_convergence(
                    old_value=old_likelihood, new_value=self.current_likelihood[a]
                )

                if converged:
                    break
                else:
                    # For the next iteration
                    old_likelihood = self.current_likelihood[a]

            # Are we diverging?
            if ((outer_iteration > 0) | (inner_iteration > 1)) & decreasing:
                warnings.warn("Likelihood is decreasing. Breaking.")
                # Reset to values from the previous iteration
                theta = prev_theta
                self.model_selection = prev_model_selection
                self.x_gram[p] = prev_x_gram
                self.y_gram[p] = prev_y_gram
                self.beta = prev_beta
                self.beta_path = prev_beta_path
                self.current_likelihood[a] = old_likelihood
                break

        return theta

    # Different UV - MV
    def _outer_update(self, X, y, theta):

        adr_start = time.time()
        for a in range(min(self.A, self.last_fit_adr_max + 1)):
            adr_it_start = time.time()
            outer_start = time.time()

            global_old_likelihood = 0
            converged = False

            for outer_iteration in range(self.max_iterations_outer):
                outer_it_start = time.time()
                # Main Loop
                for p in range(self.distribution.n_params):
                    message = (
                        f"Outer Iteration: {outer_iteration}, "
                        f"fitting Distribution parameter {p}, AD-r step {a}"
                    )
                    self.print_message(level=2, string=message)
                    theta = self._inner_update(
                        X=X, y=y, theta=theta, outer_iteration=outer_iteration, p=p, a=a
                    )

                # Get global LL
                global_likelihood = self.current_likelihood[a]
                converged, _ = self._check_outer_convergence(
                    global_old_likelihood, global_likelihood
                )

                # start value for the next AD-R step
                if converged | (outer_iteration == self.max_iterations_outer - 1):
                    if a < (self.A - 1):
                        for p in range(self.distribution.n_params):
                            if not self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = copy.deepcopy(theta[a][p])
                            if self.distribution._regularization_allowed[p]:
                                theta[(a + 1)][p] = self._get_next_adr_start_values(
                                    theta, p, a + 1
                                )

                # Timings
                outer_it_end = time.time()
                outer_it_time = outer_it_end - outer_it_start
                outer_it_avg = (outer_it_end - outer_start) / (outer_iteration + 1)
                outer_pred_time = outer_it_time * (
                    self.max_iterations_outer - outer_iteration - 1
                )
                message = (
                    f"Last outer iteration {outer_iteration} took {round(outer_it_time, 1)} sec. "
                    f"Average outer iteration took {round(outer_it_avg, 1)} sec. "
                    f"Expected to be finished in max {round(outer_pred_time, 1)} sec. "
                )
                self.print_message(level=2, string=message)

                # End timing for the iteration
                adr_it_end = time.time()

                if converged:
                    break
                else:
                    global_old_likelihood = global_likelihood

            # Next AD-R iteration
            adr_it_last = adr_it_end - adr_it_start
            adr_it_avg = (adr_it_end - adr_start) / (a + 1)

            if self.verbose > 1:  # Level 1 message
                print(
                    self._verbose_prefix,
                    f"Last ADR iteration {a} took {round(adr_it_last, 1)} sec. "
                    f"Average ADR iteration took {round(adr_it_avg, 1)} sec. ",
                    end=self._verbose_end[min(self.verbose, 1)],
                )

            # Calculate the improvement
            if self.early_stopping_criteria == "ll":
                # Check the likelihood for early stopping
                self.improvement_abs = -np.diff(self.current_likelihood)
                self.improvement_abs_scaled = (
                    -self.improvement_abs / self.n_effective_training
                )
                self.improvement_rel = (
                    self.improvement_abs / self.current_likelihood[-1:]
                )

            elif self.early_stopping_criteria in ["aic", "bic", "hqc", "max"]:
                self.early_stopping_n_params = np.array(
                    [self.count_nonzero_coef(self.beta, a) for a in range(self.A)]
                )
                self.early_stopping_ic = information_criteria_likelihood(
                    log_likelihood=self.current_likelihood,
                    n_observations=self.n_effective_training,
                    n_parameters=self.early_stopping_n_params,
                    ic=self.early_stopping_criteria,
                )

                self.improvement_abs = -np.diff(self.early_stopping_ic)
                self.improvement_abs_scaled = self.improvement_abs
                self.improvement_rel = (
                    self.improvement_abs / self.early_stopping_ic[-1:]
                )
            else:
                raise ValueError(
                    "Did not recognice criteria AD-r regularization stopping criteria."
                )

            if (
                self.early_stopping
                and (a > 0)
                and (
                    a < min(self.A - 1, self.last_fit_adr_max)
                )  # In the last step, it does not make sense to "early stop"
            ):
                if (
                    self.improvement_abs_scaled[a - 1] < self.early_stopping_abs_tol
                ) or (self.improvement_rel[a - 1] < self.early_stopping_rel_tol):
                    if self.verbose > 0:
                        print(
                            self._verbose_prefix,
                            f"Early stopping due to AD-r-regression. "
                            f"Last inrcease in r lead to relative improvement: {self.improvement_rel[a-1]}, scaled absolute improvement {self.improvement_abs_scaled[a-1]}",
                        )
                    if self.improvement_rel[a - 1] > 0:
                        self.optimal_adr = a
                    elif self.improvement_rel[a - 1] < 0:
                        self.optimal_adr = a - 1

                    # TODO: What to put in here?
                    self.improvement_abs[(a + 1) :] = 0
                    self.improvement_rel[(a + 1) :] = 0
                    self.improvement_abs_scaled[(a + 1) :] = 0
                    break
            else:
                # The largest theta is the optimal one
                # But might be overfit
                self.optimal_adr = a

        self.theta = theta
        self.optimal_theta = self.theta[self.optimal_adr]
        self.last_fit_adr_max = self.optimal_adr
