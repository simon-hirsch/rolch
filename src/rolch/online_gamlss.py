import copy
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np

from rolch.abc import Distribution, Estimator
from rolch.coordinate_descent import (
    DEFAULT_ESTIMATOR_KWARGS,
    online_coordinate_descent,
    online_coordinate_descent_path,
)
from rolch.gram import (
    init_forget_vector,
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_inverted_gram,
)
from rolch.information_criteria import select_best_model_by_information_criterion
from rolch.scaler import OnlineScaler
from rolch.utils import (
    calculate_effective_training_length,
    handle_param_dict,
    online_mean_update,
)


class OnlineGamlss(Estimator):
    """The online/incremental GAMLSS class."""

    def __init__(
        self,
        distribution: Distribution,
        equation: Dict,
        forget: float = 0.0,
        method: str = "ols",
        scale_inputs: bool = True,
        fit_intercept: Union[bool, Dict] = True,
        # Less important parameters
        beta_bounds: Dict[int, Tuple] = None,
        estimation_kwargs: Optional[Dict] = None,
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
        super().__init__(
            distribution=distribution,
            equation=equation,
            forget=forget,
            fit_intercept=fit_intercept,
            method=method,
        )

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

        # More specific args
        if beta_bounds is not None and self.method != "lasso":
            warnings.warn(
                f"[{self.__class__.__name__}] "
                f"Coefficient bounds can only be used if the estimation method == 'lasso'"
            )
        self.beta_bounds = {} if beta_bounds is None else beta_bounds
        for i, attribute in DEFAULT_ESTIMATOR_KWARGS.items():
            if (estimation_kwargs is not None) and (i in estimation_kwargs.keys()):
                setattr(self, i, estimation_kwargs[i])
            else:
                setattr(self, i, attribute)

        self.is_regularized = {}
        self.rss = {}

    def fit_beta(
        self,
        x_gram,
        y_gram,
        X,
        y,
        w,
        beta_path,
        iteration_outer,
        iteration_inner,
        param,
    ):

        f = init_forget_vector(self.forget[param], self.n_obs)

        if self.method == "ols":
            lambda_max = None
            lambda_path = None
            beta_path = None

            beta = (x_gram @ y_gram).flatten()
            residuals = y - X @ beta.T

            rss = np.sum(residuals**2 * w * f) / np.mean(w * f)

        elif (self.method == "lasso") & self._is_intercept_only(param=param):
            lambda_max = None
            lambda_path = None
            beta_path = None

            beta = online_coordinate_descent(
                x_gram,
                y_gram.flatten(),
                np.zeros(1),
                regularization=0.0,
                is_regularized=np.repeat(True, 1),
                beta_lower_bound=np.repeat(-np.inf, 1),
                beta_upper_bound=np.repeat(np.inf, 1),
                selection="cyclic",
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )[0]
            residuals = y - X @ beta.T

            rss = np.sum(residuals**2 * w * f, axis=0) / np.mean(w * f)

        elif self.method == "lasso":
            intercept = (
                y_gram[~self.is_regularized[param]]
                / np.diag(x_gram)[~self.is_regularized[param]]
            )
            lambda_max = np.max(np.abs(y_gram.flatten() - x_gram[0] * intercept))
            lambda_path = np.geomspace(
                lambda_max, lambda_max * self.lambda_eps[param], self.lambda_n
            )
            beta_path = online_coordinate_descent_path(
                x_gram=x_gram,
                y_gram=y_gram.flatten(),
                beta_path=beta_path,
                lambda_path=lambda_path,
                is_regularized=self.is_regularized[param],
                beta_lower_bound=self.beta_bounds[param][0],
                beta_upper_bound=self.beta_bounds[param][1],
                which_start_value="previous_lambda",
                selection="cyclic",
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )[0]

            residuals = y[:, None] - X @ beta_path.T

            rss = np.sum(residuals**2 * w[:, None] * f[:, None], axis=0) / np.mean(
                w * f
            )

            model_params_n = np.sum(~np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training[param], model_params_n, rss, self.ic[param]
            )
            beta = beta_path[best_ic, :]

            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic

        self.residuals[param] = residuals
        self.weights[param] = w

        return beta, beta_path, rss, lambda_max, lambda_path

    def update_beta(
        self,
        x_gram,
        y_gram,
        X,
        y,
        w,
        beta_path,
        iteration_outer,
        iteration_inner,
        param,
    ):

        denom = online_mean_update(
            self.mean_of_weights[param], w, self.forget[param], self.n_obs
        )

        if self.method == "ols":
            # Not relevant for OLS
            lambda_max = None
            lambda_path = None
            beta_path = None

            beta = (x_gram @ y_gram).flatten()
            residuals = y - X @ beta.T

            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget[param])
                * (self.rss[param] * self.mean_of_weights[param])
            ) / denom

        elif (self.method == "lasso") & self._is_intercept_only(param=param):
            lambda_max = None
            lambda_path = None
            beta_path = None

            beta = online_coordinate_descent(
                x_gram,
                y_gram.flatten(),
                np.zeros(1),
                regularization=0.0,
                is_regularized=np.repeat(True, 1),
                beta_lower_bound=np.repeat(-np.inf, 1),
                beta_upper_bound=np.repeat(np.inf, 1),
                selection="cyclic",
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )[0]
            residuals = y - X @ beta.T

            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget[param])
                * (self.rss[param] * self.mean_of_weights[param])
            ) / denom

        elif self.method == "lasso":
            intercept = (
                y_gram[~self.is_regularized[param]]
                / np.diag(x_gram)[~self.is_regularized[param]]
            )
            lambda_max = np.max(np.abs(y_gram.flatten() - x_gram[0] * intercept))
            lambda_path = np.geomspace(
                lambda_max, lambda_max * self.lambda_eps[param], self.lambda_n
            )
            beta_path = online_coordinate_descent_path(
                x_gram=x_gram,
                y_gram=y_gram.flatten(),
                beta_path=beta_path,
                lambda_path=lambda_path,
                is_regularized=self.is_regularized[param],
                beta_lower_bound=self.beta_bounds[param][0],
                beta_upper_bound=self.beta_bounds[param][1],
                which_start_value=self.start_value,
                selection="cyclic",
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )[0]
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

        return beta, beta_path, rss, lambda_max, lambda_path

    def _make_initial_fitted_values(self, y: np.ndarray) -> np.ndarray:
        fv = np.stack(
            [
                self.distribution.initial_values(y, param=i)
                for i in range(self.distribution.n_params)
            ],
            axis=1,
        )
        return fv

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

        fv = self._make_initial_fitted_values(y=y)
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

        beta_path = {
            p: np.empty((self.lambda_n, self.J[p]))
            for p in range(self.distribution.n_params)
        }
        rss = {i: 0 for i in range(self.distribution.n_params)}

        x_gram = {}
        y_gram = {}
        self.weights = {}
        self.residuals = {}

        ## Handle parameter bounds
        if self.beta_bounds is None:
            self.beta_bounds = {
                p: (np.repeat(-np.inf, self.J[p]), np.repeat(np.inf, self.J[p]))
                for p in range(self.distribution.n_params)
            }
        else:
            for p in set(range(self.distribution.n_params)).difference(
                set(self.beta_bounds.keys())
            ):
                self.beta_bounds[p] = (
                    np.repeat(-np.inf, self.J[p]),
                    np.repeat(np.inf, self.J[p]),
                )

        if self.method == "lasso":
            for p in range(self.distribution.n_params):
                is_regularized = np.repeat(True, self.J[p])
                if self.fit_intercept[p]:
                    is_regularized[0] = False
                self.is_regularized[p] = is_regularized

        self.beta_iterations = {i: {} for i in range(self.distribution.n_params)}
        self.beta_path_iterations = {i: {} for i in range(self.distribution.n_params)}

        self.beta_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.beta_path_iterations_inner = {
            i: {} for i in range(self.distribution.n_params)
        }
        self.rss_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.ic_iterations_inner = {i: {} for i in range(self.distribution.n_params)}

        # We need to track the sum of weights for each
        # distribution parameter for
        # model selection online
        self.sum_of_weights = {}
        self.mean_of_weights = {}

        # TODO: Refactor this. Almost everything can be written into class attributes during fit!
        (
            self.betas,
            self.beta_path,
            self.fv,
            self.global_dev,
            self.iteration_outer,
            self.x_gram,
            self.y_gram,
            self.rss,
            self.lambda_path,
            self.lambda_max,
        ) = self._outer_fit(
            X=X_dict,
            y=y,
            w=w,
            x_gram=x_gram,
            y_gram=y_gram,
            beta_path=beta_path,
            rss=rss,
            lambda_max={},
            lambda_path={},
            fv=fv,
        )

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
        fv = self.predict(X, what="response")

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

        self.lambda_max_inner = copy.copy(self.lambda_max)
        self.lambda_path_inner = copy.copy(self.lambda_path)

        (
            self.betas,
            self.beta_path,
            self.fv,
            self.global_dev,
            self.iteration_outer,
            self.x_gram,
            self.y_gram,
        ) = self._outer_update(
            X=X_dict,
            y=y,
            w=w,
            x_gram=self.x_gram,
            y_gram=self.y_gram,
            fv=fv,
            rss=self.rss,
            beta_path=self.beta_path,
        )

        self.x_gram = copy.copy(self.x_gram_inner)
        self.y_gram = copy.copy(self.y_gram_inner)
        self.sum_of_weights = copy.copy(self.sum_of_weights_inner)
        self.mean_of_weights = copy.copy(self.mean_of_weights_inner)
        self.rss = copy.copy(self.rss_inner)

        self.lambda_max = copy.copy(self.lambda_max_inner)
        self.lambda_path = copy.copy(self.lambda_path_inner)

    def _outer_update(self, X, y, w, x_gram, y_gram, beta_path, rss, fv):
        ## for new observations:
        global_di = -2 * np.log(self.distribution.pdf(y, fv))
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
                    fv,
                    global_dev,
                    betas[param],
                    beta_path[param],
                    self.x_gram_inner[param],
                    self.y_gram_inner[param],
                    self.rss_inner[param],
                    self.lambda_max_inner[param],
                    self.lambda_path_inner[param],
                ) = self._inner_update(
                    beta=betas[param],
                    beta_path=beta_path[param],
                    X=X,
                    y=y,
                    fv=fv,
                    w=w,
                    rss=rss,
                    x_gram=x_gram,
                    y_gram=y_gram,
                    iteration_outer=iteration_outer,
                    param=param,
                    dv=global_dev,
                    betas=betas,
                    lambda_max=self.lambda_max_inner,
                    lambda_path=self.lambda_path_inner,
                )
            # TODO (SH) Here?!
            # self.x_gram = x_gram
            # self.y_gram = y_gram

        return betas, beta_path, fv, global_dev, iteration_outer, x_gram, y_gram

    def _outer_fit(
        self, X, y, w, x_gram, y_gram, beta_path, rss, lambda_max, lambda_path, fv
    ):
        global_di = -2 * np.log(self.distribution.pdf(y, fv))
        global_dev = np.sum(w * global_di)
        global_dev_old = global_dev + 1000
        iteration_outer = 0

        betas = {}

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
                    fv,
                    global_dev,
                    betas[param],
                    beta_path[param],
                    x_gram,
                    y_gram,
                    rss[param],
                    lambda_max[param],
                    lambda_path[param],
                ) = self._inner_fit(
                    X=X,
                    y=y,
                    beta_path=beta_path[param],
                    fv=fv,
                    w=w,
                    x_gram=x_gram,
                    y_gram=y_gram,
                    param=param,
                    iteration_outer=iteration_outer,
                    dv=global_dev,
                    betas=betas,
                    rss=rss,
                    lambda_max=lambda_max,
                    lambda_path=lambda_path,
                )

                self.beta_iterations[param][iteration_outer] = betas[param]
                self.beta_path_iterations[param][iteration_outer] = beta_path[param]

        return (
            betas,
            beta_path,
            fv,
            global_dev,
            iteration_outer,
            x_gram,
            y_gram,
            rss,
            lambda_path,
            lambda_max,
        )

    def _inner_fit(
        self,
        X,
        y,
        w,
        x_gram,
        y_gram,
        beta_path,
        fv,
        iteration_outer,
        param,
        rss,
        dv,
        betas,
        lambda_max,
        lambda_path,
    ):
        if iteration_outer > 1:
            beta = betas[param]
            rss = rss[param]
            lambda_max = lambda_max[param]
            lambda_path = lambda_path[param]

        di = -2 * np.log(self.distribution.pdf(y, fv))
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
            eta = self.distribution.link_function(fv[:, param], param=param)
            # if iteration == 1:
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            # mu, sigma, nu vs. fv?
            dl1dp1 = self.distribution.dl1_dp1(y, fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)
            ## Update the X and Y Gramian and the weight

            x_gram[param] = self._make_x_gram(x=X[param], w=(w * wt), param=param)
            y_gram[param] = self._make_y_gram(x=X[param], y=wv, w=(w * wt), param=param)
            beta_new, beta_path_new, rss_new, lambda_max_new, lambda_path_new = (
                self.fit_beta(
                    x_gram[param],
                    y_gram[param],
                    X[param],
                    wv,
                    wt,
                    beta_path,
                    param=param,
                    iteration_inner=iteration_inner,
                    iteration_outer=iteration_outer,
                )
            )

            if iteration_inner > 1 or iteration_outer > 1:

                if self.method == "ols" or self._is_intercept_only(param=param):
                    if rss_new > (self.rss_tol_inner * rss):
                        break
                else:
                    ic_idx = self.ic_iterations_inner[param][iteration_outer][
                        iteration_inner
                    ]
                    if rss_new[ic_idx] > (self.rss_tol_inner * rss[ic_idx]):
                        break

            beta = beta_new
            beta_path = beta_path_new
            rss = rss_new
            lambda_max = lambda_max_new
            lambda_path = lambda_path_new

            eta = X[param] @ beta.T
            fv[:, param] = self.distribution.link_inverse(eta, param=param)

            di = -2 * np.log(self.distribution.pdf(y, fv))
            olddv = dv
            dv = np.sum(di * w)

            ## Sum of weights
            self.sum_of_weights[param] = np.sum(w * wt)
            self.mean_of_weights[param] = np.mean(w * wt)

            self.beta_iterations_inner[param][iteration_outer][iteration_inner] = beta
            self.beta_path_iterations_inner[param][iteration_outer][
                iteration_inner
            ] = beta_path

        return (fv, dv, beta, beta_path, x_gram, y_gram, rss, lambda_max, lambda_path)

    def _inner_update(
        self,
        beta,
        beta_path,
        X,
        y,
        w,
        x_gram,
        y_gram,
        rss,
        fv,
        iteration_outer,
        dv,
        param,
        betas,
        lambda_max,
        lambda_path,
    ):

        beta = betas[param]
        rss = rss[param]
        lambda_max = lambda_max[param]
        lambda_path = lambda_path[param]

        ## TODO REFACTOR: Will be returned if we converge in the first iteration
        ## Will be overwritten if don't converge in the first iteration
        x_gram_it = self.x_gram_inner[param]
        y_gram_it = self.y_gram_inner[param]

        di = -2 * np.log(self.distribution.pdf(y, fv))
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
            eta = self.distribution.link_function(fv[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            # mu, sigma, nu vs. fv?
            dl1dp1 = self.distribution.dl1_dp1(y, fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, -1e10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            x_gram_it = self._update_x_gram(
                gram=x_gram[param], x=X[param], w=float(w * wt), param=param
            )
            y_gram_it = self._update_y_gram(
                gram=y_gram[param], x=X[param], y=wv, w=float(w * wt), param=param
            )
            beta_new, beta_path_new, rss_new, lambda_max_new, lambda_path_new = (
                self.update_beta(
                    x_gram_it,
                    y_gram_it,
                    X[param],
                    y=wv,
                    w=wt,
                    beta_path=beta_path,
                    iteration_inner=iteration_inner,
                    iteration_outer=iteration_outer,
                    param=param,
                )
            )

            if self.method == "ols" or self._is_intercept_only(param=param):
                if rss_new > (self.rss_tol_inner * rss):
                    break
            else:
                ic_idx = self.ic_iterations_inner[param][iteration_outer][
                    iteration_inner
                ]
                if rss_new[ic_idx] > (self.rss_tol_inner * rss[ic_idx]):
                    break

            beta = beta_new
            beta_path = beta_path_new
            rss = rss_new
            lambda_max = lambda_max_new
            lambda_path = lambda_path_new

            eta = X[param] @ beta.T
            fv[:, param] = self.distribution.link_inverse(eta, param=param)

            self.sum_of_weights_inner[param] = (
                np.sum(w * wt) + (1 - self.forget[param]) * self.sum_of_weights[param]
            )
            self.mean_of_weights_inner[param] = (
                self.sum_of_weights_inner[param] / self.n_training[param]
            )

            olddv = dv

            di = -2 * np.log(self.distribution.pdf(y, fv))
            dv = np.sum(di * w) + (1 - self.forget[0]) * self.global_dev

        return (
            fv,
            dv,
            beta,
            beta_path,
            x_gram_it,
            y_gram_it,
            rss,
            lambda_max,
            lambda_path,
        )

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
