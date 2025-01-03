import copy
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rolch import HAS_PANDAS, HAS_POLARS
from rolch.base import Distribution
from rolch.gram import init_forget_vector
from rolch.information_criteria import select_best_model_by_information_criterion
from rolch.methods import get_estimation_method
from rolch.scaler import OnlineScaler
from rolch.utils import calculate_effective_training_length, online_mean_update

if HAS_PANDAS:
    import pandas as pd
if HAS_POLARS:
    import polars as pl


class OnlineGamlss:
    """The online/incremental GAMLSS class."""

    def __init__(
        self,
        distribution: Distribution,
        equation: Dict,
        forget: float = 0.0,
        method: str = "ols",
        scale_inputs: bool = True,
        fit_intercept: Union[bool, Dict] = True,
        regularize_intercept: Union[bool, Dict] = False,
        ic: Union[str, Dict] = "aic",
        max_it_outer: int = 30,
        max_it_inner: int = 30,
        abs_tol_outer: float = 1e-3,
        abs_tol_inner: float = 1e-3,
        rel_tol_outer: float = 1e-5,
        rel_tol_inner: float = 1e-5,
        rss_tol_inner: float = 1.5,
    ):
        """Initialise the online GAMLSS Model

        Args:
            distribution (rolch.Distribution): The parametric distribution
            equation (Dict): The modelling equation. Follows the schema `{parameter[int]: column_identifier}`, where column_identifier can be either the strings `'all'`, `'intercept'` or a np.array of ints indicating the columns.
            forget (float, optional): The forget factor. Defaults to 0.0.
            method (str, optional): The estimation method. Defaults to `"ols"`.
            scale_inputs (Optional[Dict], optional): Whether to scale the input matrices. Defaults to True
            beta_bounds (Dict[int, Tuple]): Dictionary of bounds for the different parameters.
            estimation_kwargs (Optional[Dict], optional): Dictionary of estimation method kwargs. Defaults to None.
            max_it_outer (int, optional): Maximum outer iterations for the RS algorithm. Defaults to 30.
            max_it_inner (int, optional): Maximum inner iterations for the RS algorithm. Defaults to 30.
            abs_tol_outer (float, optional): Absolute tolerance on the Deviance in the outer fit. Defaults to 1e-3.
            abs_tol_inner (float, optional): Absolute tolerance on the Deviance in the inner fit. Defaults to 1e-3.
            rel_tol_outer (float, optional): Relative tolerance on the Deviance in the outer fit. Defaults to 1e-20.
            rel_tol_inner (float, optional): Relative tolerance on the Deviance in the inner fit. Defaults to 1e-20.
            rss_tol_inner (float, optional): Tolerance for increasing RSS in the inner fit. Defaults to 1.5.
        """

        self.distribution = distribution
        self.equation = self._process_equation(equation)
        self._process_attribute(fit_intercept, True, "fit_intercept")
        self._process_attribute(regularize_intercept, False, "regularize_intercept")
        self._process_attribute(forget, default=0.0, name="forget")
        self._process_attribute(ic, "aic", name="ic")

        # Get the estimation method
        self._process_attribute(method, default="ols", name="method")
        self._method = {p: get_estimation_method(m) for p, m in self.method.items()}

        self.scaler = OnlineScaler(do_scale=scale_inputs, intercept=False)
        self.do_scale = scale_inputs

        # These are global for all distribution parameters
        self.max_it_outer = max_it_outer
        self.max_it_inner = max_it_inner
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner
        self.rss_tol_inner = rss_tol_inner

        self.is_regularized = {}
        self.rss = {}

    def _process_attribute(self, attribute: Any, default: Any, name: str) -> None:
        if isinstance(attribute, dict):
            for p in range(self.distribution.n_params):
                if p not in attribute.keys():
                    warnings.warn(
                        f"[{self.__class__.__name__}] "
                        f"No value given for parameter {name} for distribution "
                        f"parameter {p}. Setting default value {default}.",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    if isinstance(default, dict):
                        attribute[p] = default[p]
                    else:
                        attribute[p] = default
        else:
            # No warning since we expect that floats/strings/ints are either the defaults
            # Or given on purpose for all params the ame
            attribute = {p: attribute for p in range(self.distribution.n_params)}

        setattr(self, name, attribute)

    def _process_equation(self, equation: Dict):
        """Preprocess the equation object and validate inputs."""
        if equation is None:
            warnings.warn(
                f"[{self.__class__.__name__}] "
                "Equation is not specified. "
                "Per default, will estimate the first distribution parameter by all covariates found in X. "
                "All other distribution parameters will be estimated by an intercept."
            )
            equation = {
                p: "all" if p == 0 else "intercept"
                for p in range(self.distribution.n_params)
            }
        else:
            for p in range(self.distribution.n_params):
                # Check that all distribution parameters are in the equation.
                # If not, add intercept.
                if p not in equation.keys():
                    warnings.warn(
                        f"[{self.__class__.__name__}] "
                        f"Distribution parameter {p} is not in equation. "
                        f"The parameter will be estimated by an intercept."
                    )
                    equation[p] = "intercept"

                if not (
                    isinstance(equation[p], np.ndarray)
                    or (equation[p] in ["all", "intercept"])
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

    @staticmethod
    def _make_intercept(n_observations: int) -> np.ndarray:
        """Make the intercept series as N x 1 array.

        Args:
            y (np.ndarray): Response variable $Y$

        Returns:
            np.ndarray: Intercept array.
        """
        return np.ones((n_observations, 1))

    def _is_intercept_only(self, param: int):
        """Check in the equation whether we model only as intercept"""
        if isinstance(self.equation[param], str):
            return self.equation[param] == "intercept"
        else:
            return False

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

    def make_model_array(self, X: Union[np.ndarray], param: int):
        eq = self.equation[param]
        n = X.shape[0]

        # TODO: Check difference between np.array and list more explicitly?
        if isinstance(eq, str) and (eq == "intercept"):
            if not self.fit_intercept[param]:
                raise ValueError(
                    "fit_intercept[param] is false, but equation says intercept."
                )
            out = self._make_intercept(n_observations=n)
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

            if self.fit_intercept[param]:
                out = np.hstack((self._make_intercept(n), out))

        return out

    def fit_beta_and_select_model(
        self,
        X,
        y,
        w,
        iteration_outer,
        iteration_inner,
        param,
    ):

        f = init_forget_vector(self.forget[param], self.n_obs)

        if not self._method[param]._path_based_method:
            beta_path = None
            beta = self._method[param].fit_beta(
                x_gram=self.x_gram[param],
                y_gram=self.y_gram[param],
                is_regularized=self.is_regularized[param],
            )
            # print(beta)
            residuals = y - X @ beta.T
            rss = np.sum(residuals**2 * w * f) / np.mean(w * f)

        else:
            beta_path = self._method[param].fit_beta_path(
                x_gram=self.x_gram[param],
                y_gram=self.y_gram[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y[:, None] - X @ beta_path.T
            rss = np.sum(residuals**2 * w[:, None] * f[:, None], axis=0)
            rss = rss / np.mean(w * f)
            model_params_n = np.sum(~np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training[param], model_params_n, rss, self.ic[param]
            )
            beta = beta_path[best_ic, :]

            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic

        self.residuals[param] = residuals
        self.weights[param] = w

        return beta, beta_path, rss

    def update_beta_and_select_model(
        self,
        X,
        y,
        w,
        iteration_outer,
        iteration_inner,
        param,
    ):

        denom = online_mean_update(
            self.mean_of_weights[param], w, self.forget[param], self.n_obs
        )

        if not self._method[param]._path_based_method:
            beta_path = None
            beta = self._method[param].update_beta(
                x_gram=self.x_gram_inner[param],
                y_gram=self.y_gram_inner[param],
                beta=self.betas[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y - X @ beta.T

            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget[param])
                * (self.rss[param] * self.mean_of_weights[param])
            ) / denom

        else:
            beta_path = self._method[param].update_beta_path(
                x_gram=self.x_gram_inner[param],
                y_gram=self.y_gram_inner[param],
                beta_path=self.beta_path[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y - X @ beta_path.T

            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget[param])
                * (self.rss[param] * self.mean_of_weights[param])
            ) / denom

            model_params_n = np.sum(np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training[param], model_params_n, rss, self.ic[param]
            )

            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic

            beta = beta_path[best_ic, :]

        return beta, beta_path, rss

    def _make_initial_fitted_values(self, y: np.ndarray) -> np.ndarray:
        out = np.stack(
            [
                self.distribution.initial_values(y, param=i)
                for i in range(self.distribution.n_params)
            ],
            axis=1,
        )
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit the online GAMLSS model.

        !!! note
            The user is only required to provide the design matrix $X$ for the first distribution parameters. If for some distribution parameter no design matrix is provided, `ROLCH` will model the parameter using an intercept.

        !!! note
            The provision of bounds for the coefficient vectors is only possible for LASSO/coordinate descent estimation.

        Args:
            X (np.ndarray): Data Matrix. Currently supporting only numpy, will support pandas and polars in the future.
            y (np.ndarray): Response variable $Y$.
            sample_weight (Optional[np.ndarray], optional): User-defined sample weights. Defaults to None.
            beta_bounds (Dict[int, Tuple], optional): Bounds for the $\beta$ in the coordinate descent algorithm. The user needs to provide a `dict` with a mapping of tuples to distribution parameters 0, 1, 2, and 3 potentially. Defaults to None.
        """
        self.n_obs = y.shape[0]
        self.n_training = {
            p: calculate_effective_training_length(self.forget[p], self.n_obs)
            for p in range(self.distribution.n_params)
        }

        if sample_weight is not None:
            w = sample_weight  # Align to sklearn API
        else:
            w = np.ones(y.shape[0])

        self.fv = self._make_initial_fitted_values(y=y)
        self.J = self.get_J_from_equation(X=X)

        # Fit scaler and transform
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }
        self.X_dict = X_dict
        self.X_scaled = X_scaled

        rss = {i: 0 for i in range(self.distribution.n_params)}

        self.x_gram = {}
        self.y_gram = {}
        self.weights = {}
        self.residuals = {}

        for p in range(self.distribution.n_params):
            is_regularized = np.repeat(True, self.J[p])
            if self.fit_intercept[p] and not (
                self.regularize_intercept[p] | self._is_intercept_only(p)
            ):
                is_regularized[0] = False
            self.is_regularized[p] = is_regularized

        self.beta_iterations = {i: {} for i in range(self.distribution.n_params)}
        self.beta_iterations_inner = {i: {} for i in range(self.distribution.n_params)}

        self.beta_path = {p: None for p in range(self.distribution.n_params)}
        self.betas = {p: np.zeros(self.J[p]) for p in range(self.distribution.n_params)}

        self.beta_path_iterations_inner = {
            i: {} for i in range(self.distribution.n_params)
        }
        self.beta_path_iterations = {i: {} for i in range(self.distribution.n_params)}

        self.rss_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.ic_iterations_inner = {i: {} for i in range(self.distribution.n_params)}

        # We need to track the sum of weights for each
        # distribution parameter for online model selection
        self.sum_of_weights = {}
        self.mean_of_weights = {}

        # TODO: Refactor this. Almost everything can be written into class attributes during fit!
        (
            self.global_dev,
            self.iteration_outer,
            self.rss,
        ) = self._outer_fit(
            X=X_dict,
            y=y,
            w=w,
            rss=rss,
        )
        print("Finished")

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Update the fit for the online GAMLSS Model.

        !!! warning
            Currently, the algorithm only takes single-step updates. Batch updates are planned for the first stable version.

        !!! note
            The `beta_bounds` from the initial fit are still valid for the update.

        Args:
            X (np.ndarray): Data Matrix. Currently supporting only numpy, will support and pandas in the future.
            y (np.ndarray): Response variable $Y$.
            sample_weight (Optional[np.ndarray], optional): User-defined sample weights. Defaults to None.
        """
        if sample_weight is not None:
            w = sample_weight  # Align to sklearn API
        else:
            w = np.ones(y.shape[0])

        self.n_obs += y.shape[0]
        self.n_training = {
            p: calculate_effective_training_length(self.forget[p], self.n_obs)
            for p in range(self.distribution.n_params)
        }

        # More efficient to do this?!
        # Since we get better start values
        self.fv = self.predict(X, what="response")

        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }

        ## Reset rss and iterations
        self.rss_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.ic_iterations_inner = {i: {} for i in range(self.distribution.n_params)}

        self.x_gram_inner = copy.copy(self.x_gram)
        self.y_gram_inner = copy.copy(self.y_gram)
        self.rss_inner = copy.copy(self.rss)
        self.sum_of_weights_inner = copy.copy(self.sum_of_weights)
        self.mean_of_weights_inner = copy.copy(self.mean_of_weights)

        (
            self.global_dev,
            self.iteration_outer,
        ) = self._outer_update(
            X=X_dict,
            y=y,
            w=w,
            rss=self.rss,
        )

        self.x_gram = copy.copy(self.x_gram_inner)
        self.y_gram = copy.copy(self.y_gram_inner)
        self.sum_of_weights = copy.copy(self.sum_of_weights_inner)
        self.mean_of_weights = copy.copy(self.mean_of_weights_inner)
        self.rss = copy.copy(self.rss_inner)

    def _outer_update(self, X, y, w, rss):
        ## for new observations:
        global_di = -2 * np.log(self.distribution.pdf(y, self.fv))
        global_dev = (1 - self.forget[0]) * self.global_dev + global_di
        global_dev_old = global_dev + 1000
        iteration_outer = 0

        betas = self.betas

        while True:
            # Check relative congergence
            if (
                np.abs(global_dev_old - global_dev) / np.abs(global_dev_old)
                < self.rel_tol_outer
            ):
                break

            if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                break

            if iteration_outer >= self.max_it_outer:
                break

            global_dev_old = global_dev
            iteration_outer += 1

            for param in range(self.distribution.n_params):

                self.beta_iterations_inner[param][iteration_outer] = {}
                self.beta_path_iterations_inner[param][iteration_outer] = {}

                self.rss_iterations_inner[param][iteration_outer] = {}
                self.ic_iterations_inner[param][iteration_outer] = {}

                (
                    global_dev,
                    self.rss_inner[param],
                ) = self._inner_update(
                    X=X,
                    y=y,
                    w=w,
                    rss=rss,
                    iteration_outer=iteration_outer,
                    param=param,
                    dv=global_dev,
                )

        return global_dev, iteration_outer

    def _outer_fit(self, X, y, w, rss):

        global_di = -2 * np.log(self.distribution.pdf(y, self.fv))
        global_dev = np.sum(w * global_di)
        global_dev_old = global_dev + 1000
        iteration_outer = 0

        while True:
            # Check relative congergence

            if (
                np.abs(global_dev_old - global_dev) / np.abs(global_dev_old)
                < self.rel_tol_outer
            ):
                break

            if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                break

            if iteration_outer >= self.max_it_outer:
                break

            global_dev_old = global_dev
            iteration_outer += 1

            for param in range(self.distribution.n_params):

                self.beta_iterations_inner[param][iteration_outer] = {}
                self.beta_path_iterations_inner[param][iteration_outer] = {}

                self.rss_iterations_inner[param][iteration_outer] = {}
                self.ic_iterations_inner[param][iteration_outer] = {}

                (
                    global_dev,
                    rss[param],
                ) = self._inner_fit(
                    X=X,
                    y=y,
                    w=w,
                    param=param,
                    iteration_outer=iteration_outer,
                    dv=global_dev,
                    rss=rss,
                )

                self.beta_iterations[param][iteration_outer] = self.betas[param]
                self.beta_path_iterations[param][iteration_outer] = self.beta_path[
                    param
                ]

        return (global_dev, iteration_outer, rss)

    def _inner_fit(
        self,
        X,
        y,
        w,
        iteration_outer,
        param,
        rss,
        dv,
    ):
        if iteration_outer > 1:
            rss = rss[param]

        di = -2 * np.log(self.distribution.pdf(y, self.fv))
        dv = np.sum(di * w)
        olddv = dv + 1

        # Use this for the while loop
        iteration_inner = 0
        while True:
            if iteration_inner >= self.max_it_inner:
                break

            # We allow for breaking in the inner iteration in
            # - the 1st outer iteration (iteration_outer = 1) after 1 inner iteration for each parameter --> SUM = 2
            # - the 2nd outer iteration (iteration_outer = 2) after 0 inner iteration for each parameter --> SUM = 2

            if (abs(olddv - dv) <= self.abs_tol_inner) & (
                (iteration_inner + iteration_outer) >= 2
            ):
                break

            if (abs(olddv - dv) / abs(olddv) < self.rel_tol_inner) & (
                (iteration_inner + iteration_outer) >= 2
            ):
                break

            iteration_inner += 1
            eta = self.distribution.link_function(self.fv[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, self.fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, self.fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            ## Update the X and Y Gramian and the weight
            self.x_gram[param] = self._method[param].init_x_gram(
                X=X[param], weights=(w * wt), forget=self.forget[param]
            )
            self.y_gram[param] = self._method[param].init_y_gram(
                X=X[param], y=wv, weights=(w * wt), forget=self.forget[param]
            )
            beta_new, beta_path_new, rss_new = self.fit_beta_and_select_model(
                X=X[param],
                y=wv,
                w=wt,
                param=param,
                iteration_inner=iteration_inner,
                iteration_outer=iteration_outer,
            )

            if iteration_inner > 1 or iteration_outer > 1:

                if self.method[param] == "ols":
                    if rss_new > (self.rss_tol_inner * rss):
                        break
                else:
                    ic_idx = self.ic_iterations_inner[param][iteration_outer][
                        iteration_inner
                    ]
                    if rss_new[ic_idx] > (self.rss_tol_inner * rss[ic_idx]):
                        break

            self.betas[param] = beta_new
            self.beta_path[param] = beta_path_new
            rss = rss_new

            eta = X[param] @ self.betas[param].T
            self.fv[:, param] = self.distribution.link_inverse(eta, param=param)

            di = -2 * np.log(self.distribution.pdf(y, self.fv))
            olddv = dv
            dv = np.sum(di * w)

            ## Sum of weights
            self.sum_of_weights[param] = np.sum(w * wt)
            self.mean_of_weights[param] = np.mean(w * wt)

            self.beta_iterations_inner[param][iteration_outer][
                iteration_inner
            ] = beta_new
            self.beta_path_iterations_inner[param][iteration_outer][
                iteration_inner
            ] = beta_path_new

        return (dv, rss)

    def _inner_update(
        self,
        X,
        y,
        w,
        rss,
        iteration_outer,
        dv,
        param,
    ):

        rss = rss[param]

        di = -2 * np.log(self.distribution.pdf(y, self.fv))
        dv = (1 - self.forget[0]) * self.global_dev + np.sum(di * w)
        olddv = dv + 1

        # Use this for the while loop
        iteration_inner = 0
        while True:
            if iteration_inner >= self.max_it_inner:
                break
            if (abs(olddv - dv) <= self.abs_tol_inner) & (
                (iteration_inner + iteration_outer) >= 2
            ):
                break
            if (abs(olddv - dv) / abs(olddv) < self.rel_tol_inner) & (
                (iteration_inner + iteration_outer) >= 2
            ):
                break

            iteration_inner += 1
            eta = self.distribution.link_function(self.fv[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            # mu, sigma, nu vs. self.fv?
            dl1dp1 = self.distribution.dl1_dp1(y, self.fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, self.fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, -1e10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            self.x_gram_inner[param] = self._method[param].update_x_gram(
                gram=self.x_gram[param],
                X=X[param],
                weights=(w * wt),
                forget=self.forget[param],
            )
            self.y_gram_inner[param] = self._method[param].update_y_gram(
                gram=self.y_gram[param],
                X=X[param],
                y=wv,
                weights=(w * wt),
                forget=self.forget[param],
            )
            beta_new, beta_path_new, rss_new = self.update_beta_and_select_model(
                X[param],
                y=wv,
                w=wt,
                iteration_inner=iteration_inner,
                iteration_outer=iteration_outer,
                param=param,
            )
            # Check if the local RSS are decreasing
            if self.method[param] == "ols":
                if rss_new > (self.rss_tol_inner * rss):
                    break
            else:
                ic_idx = self.ic_iterations_inner[param][iteration_outer][
                    iteration_inner
                ]
                if rss_new[ic_idx] > (self.rss_tol_inner * rss[ic_idx]):
                    break

            self.betas[param] = beta_new
            self.beta_path[param] = beta_path_new
            rss = rss_new

            eta = X[param] @ self.betas[param].T
            self.fv[:, param] = self.distribution.link_inverse(eta, param=param)

            self.sum_of_weights_inner[param] = (
                np.sum(w * wt) + (1 - self.forget[param]) * self.sum_of_weights[param]
            )
            self.mean_of_weights_inner[param] = (
                self.sum_of_weights_inner[param] / self.n_training[param]
            )

            olddv = dv

            di = -2 * np.log(self.distribution.pdf(y, self.fv))
            dv = np.sum(di * w) + (1 - self.forget[0]) * self.global_dev

        return (dv, rss)

    def predict(
        self,
        X: np.ndarray,
        what: str = "response",
        return_contributions: bool = False,
    ) -> np.ndarray:
        """Predict the distibution parameters given input data.

        Args:
            X (np.ndarray): Design matrix.
            what (str, optional): Predict the response or the link. Defaults to "response". Remember the  GAMLSS models $g(\\theta) = X^T\\beta$. Predict `"link"` will output $X^T\\beta$, predict `"response"` will output $g^{-1}(X^T\\beta)$. Usually, you want predict = `"response"`.
            return_contributions (bool, optional): Whether to return a `Tuple[prediction, contributions]` where the contributions of the individual covariates for each distribution parameter's predicted value is specified. Defaults to False.

        Raises:
            ValueError: Raises if `what` is not in `["link", "response"]`.

        Returns:
            np.ndarray: Predicted values for the distribution.
        """
        X_scaled = self.scaler.transform(x=X)
        X_dict = {
            p: self.make_model_array(X_scaled, p)
            for p in range(self.distribution.n_params)
        }
        prediction = [x @ b.T for x, b in zip(X_dict.values(), self.betas.values())]

        if return_contributions:
            contribution = [
                x * b.T for x, b in zip(X_dict.values(), self.betas.values())
            ]

        if what == "response":
            prediction = [
                self.distribution.link_inverse(p, param=i)
                for i, p in enumerate(prediction)
            ]
        elif what == "link":
            pass
        else:
            raise ValueError("Should be 'response' or 'link'.")

        prediction = np.stack(prediction, axis=1)

        if return_contributions:
            return (prediction, contribution)
        else:
            return prediction
