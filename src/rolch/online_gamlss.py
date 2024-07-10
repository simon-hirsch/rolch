import copy
from typing import Dict, Optional, Tuple, Union

import numpy as np

from rolch.coordinate_descent import (
    DEFAULT_ESTIMATOR_KWARGS,
    online_coordinate_descent,
    online_coordinate_descent_path,
)
from rolch.gram import init_gram, init_inverted_gram, init_y_gram, update_inverted_gram
from rolch.information_criteria import select_best_model_by_information_criterion
from rolch.scaler import OnlineScaler
from rolch.utils import calculate_effective_training_length


class OnlineGamlss:
    """The online/incremental GAMLSS class."""

    def __init__(
        self,
        distribution,
        forget: float = 0.0,
        method: str = "ols",
        do_scale: Optional[Dict] = None,
        # lambda_n: int = 100,
        # lambda_eps: float = 1e-3,
        # start_value: str = "previous_fit",
        estimation_kwargs: Optional[Dict] = None,
        max_it_outer: int = 30,
        max_it_inner: int = 30,
        abs_tol_outer: float = 1e-3,
        abs_tol_inner: float = 1e-3,
        rel_tol_outer: float = 1e-20,
        rel_tol_inner: float = 1e-20,
        rss_tol_inner: float = 1.5,
    ):
        """Initialise the online GAMLSS Model

        Args:
            distribution (rolch.Distribution): The parametric distribution
            forget (float, optional): The forget factor. Defaults to 0.0.
            method (str, optional): The estimation method. Defaults to "ols".
            do_scale (Optional[Dict], optional): Whether to scale the input matrices. Defaults to None, which implies that all inputs will be scaled. Note that the scaler assumes that the first column of each $X$ contains the intercept.
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
        self.forget = forget

        self.max_it_outer = max_it_outer
        self.max_it_inner = max_it_inner
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner
        self.rss_tol_inner = rss_tol_inner
        self.method = method  # lasso

        # self.lambda_n = lambda_n
        # self.lambda_eps = lambda_eps
        # self.start_value = start_value

        for i, attribute in DEFAULT_ESTIMATOR_KWARGS.items():
            if (estimation_kwargs is not None) and (i in estimation_kwargs.keys()):
                setattr(self, i, estimation_kwargs[i])
            else:
                setattr(self, i, attribute)

        if do_scale is not None:
            self.do_scale = do_scale
        else:
            self.do_scale = {i: True for i in range(self.distribution.n_params)}
        self.intercept = {i: True for i in range(self.distribution.n_params)}

        self.scalers = {
            i: OnlineScaler(
                forget=self.forget,
                intercept=self.intercept[i],
                do_scale=self.do_scale[i],
            )
            for i in range(self.distribution.n_params)
        }

        self.is_regularized = {}
        self.rss = {}

    def _scaler_train(self, X):
        for i, x in enumerate(X):
            self.scalers[i].fit(x)

    def _scaler_update(self, X):
        for i, x in enumerate(X):
            self.scalers[i].partial_fit(x)

    def _scaler_transform(self, X):
        return [self.scalers[i].transform(x) for i, x in enumerate(X)]

    def _make_intercept_or_x(self, x, N):
        if x is not None:
            # if x.shape[0] != N:
            #     raise ValueError("X should have the same length as Y.")
            return x
        else:
            return np.ones((N, 1))

    def _make_input_array_list(self, x0, x1, x2, x3):
        N = x0.shape[0]
        X = [x0]

        if 2 <= self.distribution.n_params:
            X.append(self._make_intercept_or_x(x1, N))
        if 3 <= self.distribution.n_params:
            X.append(self._make_intercept_or_x(x2, N))
        if 4 <= self.distribution.n_params:
            X.append(self._make_intercept_or_x(x3, N))

        return X

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
        if self.method == "ols":
            lambda_max = None
            lambda_path = None
            beta_path = None

            beta = (x_gram @ y_gram).flatten()
            residuals = y - X @ beta.T

            if self.method == "ols" or self.intercept_only[param]:
                rss = np.sum((residuals * w) ** 2, axis=0) / np.sum(w)
            else:
                rss = np.sum((residuals * w[:, None]) ** 2, axis=0) / np.sum(w)

        elif (self.method == "lasso") & self.intercept_only[param]:
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

            if self.method == "ols" or self.intercept_only[param]:
                rss = np.sum((residuals * w) ** 2, axis=0) / np.sum(w)
            elif self.method == "lasso":
                rss = np.sum((residuals * w[:, None]) ** 2, axis=0) / np.sum(w)

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

            residuals = X @ beta_path.T

            if self.method == "ols" or self.intercept_only[param]:
                rss = np.sum((residuals * w) ** 2, axis=0) / np.sum(w)
            elif self.method == "lasso":
                rss = np.sum((residuals * w[:, None]) ** 2, axis=0) / np.sum(w)

            model_params_n = np.sum(np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training, model_params_n, rss, self.ic
            )
            beta = beta_path[best_ic, :]

            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic

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
        if self.method == "ols":
            # Not relevant for OLS
            lambda_max = None
            lambda_path = None
            beta_path = None

            beta = (x_gram @ y_gram).flatten()
            residuals = y - X @ beta.T

            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget) * (self.rss[param] * self.sum_of_weights[param])
            ) / (self.sum_of_weights[param] * (1 - self.forget) + w)

        elif (self.method == "lasso") & self.intercept_only[param]:
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
                + (1 - self.forget) * (self.rss[param] * self.sum_of_weights[param])
            ) / (self.sum_of_weights[param] * (1 - self.forget) + w)

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
                + (1 - self.forget) * (self.rss[param] * self.sum_of_weights[param])
            ) / (self.sum_of_weights[param] * (1 - self.forget) + w)

            model_params_n = np.sum(np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training, model_params_n, rss, self.ic
            )

            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic

            beta = beta_path[best_ic, :]

        return beta, beta_path, rss, lambda_max, lambda_path

    def _make_x_gram_list(self, X, w):
        """Make the Gramian Matrices G = X.T @ W @ GAMMA @ X"""
        return [init_gram(x, w, self.forget) for x in X]

    def fit(
        self,
        y: np.ndarray,
        x0: np.ndarray,
        x1: Optional[np.ndarray] = None,
        x2: Optional[np.ndarray] = None,
        x3: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        beta_bounds: Dict[int, Tuple] = None,
    ):
        """Fit the online GAMLSS model.

        !!! note
            The user is only required to provide the design matrix $X$ for the first distribution parameters. If for some distribution parameter no design matrix is provided, `ROLCH` will model the parameter using an intercept.

        !!! note
            The provision of bounds for the coefficient vectors is only possible for LASSO/coordinate descent estimation.

        Args:
            y (np.ndarray): Response variable $Y$.
            x0 (np.ndarray): Design matrix for the 1st distribution parameter $X_\\mu$
            x1 (Optional[np.ndarray], optional): Design matrix for the 2nd distribution parameter $X_\\sigma$. Defaults to None.
            x2 (Optional[np.ndarray], optional): Design matrix for the 3rd distribution parameter $X_\\nu$. Defaults to None.
            x3 (Optional[np.ndarray], optional): Design matrix for the 4th distribution parameter $X_\\tau$. Defaults to None.
            sample_weights (Optional[np.ndarray], optional): User-defined sample weights. Defaults to None.
            beta_bounds (Dict[int, Tuple], optional): Bounds for the $\beta$ in the coordinate descent algorithm. The user needs to provide a `dict` with a mapping of tuples to distribution parameters 0, 1, 2, and 3 potentially. Defaults to None.
        """
        self.n_obs = y.shape[0]
        self.n_training = calculate_effective_training_length(self.forget, self.n_obs)

        if sample_weights is not None:
            w = sample_weights  # Align to sklearn API
        else:
            w = np.ones(y.shape[0])

        fv = np.stack(
            [
                self.distribution.initial_values(y, param=i)
                for i in range(self.distribution.n_params)
            ],
            axis=1,
        )
        X = self._make_input_array_list(x0=x0, x1=x1, x2=x2, x3=x3)
        self._scaler_train(X)
        # self.foo = self._scaler_transform(X)
        X_scaled = self._scaler_transform(X)
        x_gram = self._make_x_gram_list(X=X_scaled, w=w)
        y_gram = [np.empty((x.shape[1], 1)) for x in X]
        beta_path = {i: np.empty((self.lambda_n, x.shape[1])) for i, x in enumerate(X)}
        rss = {i: 0 for i in range(self.distribution.n_params)}

        J = {p: x.shape[1] for p, x in enumerate(X)}

        # Intercept only
        self.intercept_only = {
            p: self.intercept[p] == J[p] for p in range(self.distribution.n_params)
        }

        ## Handle parameter bounds
        if beta_bounds is None:
            self.beta_bounds = {
                p: (np.repeat(-np.inf, J[p]), np.repeat(np.inf, J[p]))
                for p in range(self.distribution.n_params)
            }
        else:
            for p in set(range(self.distribution.n_params)).difference(
                set(beta_bounds.keys())
            ):
                beta_bounds[p] = (np.repeat(-np.inf, J[p]), np.repeat(np.inf, J[p]))
            self.beta_bounds = beta_bounds

        if self.method == "lasso":
            for param in range(self.distribution.n_params):
                is_regularized = np.repeat(True, X[param].shape[1])
                is_regularized[0] = False
                self.is_regularized[param] = is_regularized

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
            X=X_scaled,
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

    def update_not_inverted_gram(self, gram, x_new, w_new=1):
        return (1 - self.forget) * gram + w_new * (x_new.T * x_new)

    def update_y_gram(self, gram, x_new, y_new, w_new=1):
        return (1 - self.forget) * gram + w_new * (x_new.T * y_new)

    def update_inverted_gram(self, inv_gram, x_new, w_new=1):
        gamma = 1 - self.forget
        new_inv_gram = (1 / (gamma)) * (
            inv_gram
            - (
                (w_new * inv_gram * x_new.T * x_new * inv_gram)
                / (gamma + w_new * x_new * inv_gram * x_new.T)
            )
        )
        return new_inv_gram

    def update_gram(self, gram, x_new, w_new=1):
        if self.method == "ols":
            return update_inverted_gram(gram, x_new, self.forget, w_new)
        else:
            return self.update_not_inverted_gram(gram, x_new, w_new)

    def make_x_gram(self, X, weights):
        if self.method == "ols":
            return init_inverted_gram(X, weights, self.forget)
        else:
            return init_gram(X, weights, self.forget)

    def update(
        self,
        y: np.ndarray,
        x0: np.ndarray,
        x1: Optional[np.ndarray] = None,
        x2: Optional[np.ndarray] = None,
        x3: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ):
        """Update the fit for the online GAMLSS Model.

        !!! warning
            Currently, the algorithm only takes single-step updates. Batch updates are planned for the first stable version.

        !!! note
            The `beta_bounds` from the initial fit are still valid for the update.

        Args:
            y (np.ndarray): Response variable $Y$.
            x0 (np.ndarray): Design matrix for the 1st distribution parameter $X_\\mu$
            x1 (Optional[np.ndarray], optional): Design matrix for the 2nd distribution parameter $X_\\sigma$. Defaults to None.
            x2 (Optional[np.ndarray], optional): Design matrix for the 3rd distribution parameter $X_\\nu$. Defaults to None.
            x3 (Optional[np.ndarray], optional): Design matrix for the 4th distribution parameter $X_\\tau$. Defaults to None.
            sample_weights (Optional[np.ndarray], optional): User-defined sample weights. Defaults to None.
        """
        if sample_weights is not None:
            w = sample_weights  # Align to sklearn API
        else:
            w = np.ones(y.shape[0])

        self.n_obs += y.shape[0]
        self.n_training = calculate_effective_training_length(self.forget, self.n_obs)

        # More efficient to do this?!
        # Since we get better start values
        fv = self.predict(x0=x0, x1=x1, x2=x2, x3=x3, what="response")

        X = self._make_input_array_list(x0=x0, x1=x1, x2=x2, x3=x3)

        self._scaler_update(X=X)
        X_scaled = self._scaler_transform(X=X)

        ## Reset rss and iterations
        self.rss_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.ic_iterations_inner = {i: {} for i in range(self.distribution.n_params)}

        self.x_gram_inner = copy.copy(self.x_gram)
        self.y_gram_inner = copy.copy(self.y_gram)
        self.rss_inner = copy.copy(self.rss)
        self.sum_of_weights_inner = copy.copy(self.sum_of_weights)

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
            X=X_scaled,
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
        self.rss = copy.copy(self.rss_inner)

        self.lambda_max = copy.copy(self.lambda_max_inner)
        self.lambda_path = copy.copy(self.lambda_path_inner)

    def _outer_update(self, X, y, w, x_gram, y_gram, beta_path, rss, fv):
        ## for new observations:
        global_di = -2 * np.log(self.distribution.pdf(y, fv))
        global_dev = (1 - self.forget) * self.global_dev + global_di
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

        betas = [np.zeros((self.lambda_n, i.shape[1])) for i in X]

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

                # print("Start:", betas[param])

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

            if abs(olddv - dv) <= 1e-3:
                break

            if abs(olddv - dv) / abs(olddv) < self.rel_tol_inner:
                break

            iteration_inner += 1
            eta = self.distribution.link_function(fv[:, param], param=param)
            # if iteration == 1:
            dr = 1 / self.distribution.link_derivative(eta, param=param)
            # mu, sigma, nu vs. fv?
            dl1dp1 = self.distribution.dl1_dp1(y, fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)
            ## Update the X and Y Gramian and the weight
            x_gram[param] = self.make_x_gram(X[param], w * wt)
            y_gram[param] = init_y_gram(X[param], wv, w * wt, self.forget)
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

            # print(param, ":", rss_new)

            if iteration_inner > 1 or iteration_outer > 1:

                if self.method == "ols" or self.intercept_only[param]:
                    if rss_new > (self.rss_tol_inner * rss):
                        # print("Rss break")
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

            self.beta_iterations_inner[param][iteration_outer][iteration_inner] = beta
            self.beta_path_iterations_inner[param][iteration_outer][
                iteration_inner
            ] = beta_path

        # print("End:", beta)

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

        di = -2 * np.log(self.distribution.pdf(y, fv))
        dv = (1 - self.forget) * self.global_dev + np.sum(di * w)
        olddv = dv + 1

        # Use this for the while loop
        iteration_inner = 0
        while True:
            if iteration_inner >= self.max_it_inner:
                break
            if abs(olddv - dv) <= self.abs_tol_inner:
                break
            if abs(olddv - dv) / abs(olddv) < self.rel_tol_inner:
                break

            iteration_inner += 1
            eta = self.distribution.link_function(fv[:, param], param=param)
            dr = 1 / self.distribution.link_derivative(eta, param=param)
            # mu, sigma, nu vs. fv?
            dl1dp1 = self.distribution.dl1_dp1(y, fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, -1e10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            x_gram_it = self.update_gram(x_gram[param], X[param], float(w * wt))
            y_gram_it = self.update_y_gram(
                y_gram[param],
                X[param],
                wv,
                float(w * wt),
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

            if self.method == "ols" or self.intercept_only[param]:
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
                np.sum(w * wt) + (1 - self.forget) * self.sum_of_weights[param]
            )

            olddv = dv

            di = -2 * np.log(self.distribution.pdf(y, fv))
            dv = np.sum(di * w) + (1 - self.forget) * self.global_dev

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
        x0: np.ndarray,
        x1: Optional[np.ndarray] = None,
        x2: Optional[np.ndarray] = None,
        x3: Optional[np.ndarray] = None,
        what: str = "response",
        return_contributions: bool = False,
    ) -> np.ndarray:
        """Predict the distibution parameters given input data.

        Args:
            x0 (np.ndarray): Design matrix for the 1st distribution parameter $X_\\mu$
            x1 (Optional[np.ndarray], optional): Design matrix for the 2nd distribution parameter $X_\\sigma$. Defaults to None.
            x2 (Optional[np.ndarray], optional): Design matrix for the 3rd distribution parameter $X_\\nu$. Defaults to None.
            x3 (Optional[np.ndarray], optional): Design matrix for the 4th distribution parameter $X_\\tau$. Defaults to None.
            what (str, optional): Predict the response or the link. Defaults to "response".
            return_contributions (bool, optional): Whether to return a `Tuple[prediction, contributions]` where the contributions of the individual covariates for each distribution parameter's predicted value is specified. Defaults to False.

        Raises:
            ValueError: Raises if `what` is not in `["link", "response"]`.

        Returns:
            np.ndarray: Predicted values for the distribution.
        """
        X = self._make_input_array_list(x0=x0, x1=x1, x2=x2, x3=x3)
        X_scaled = self._scaler_transform(X=X)
        prediction = [x @ b.T for x, b in zip(X_scaled, self.betas)]

        if return_contributions:
            contribution = [x * b.T for x, b in zip(X_scaled, self.betas)]

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
