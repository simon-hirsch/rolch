import copy
import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

from .. import HAS_PANDAS, HAS_POLARS
from ..base import Distribution, EstimationMethod, Estimator
from ..error import OutOfSupportError
from ..gram import init_forget_vector
from ..information_criteria import InformationCriterion
from ..methods import get_estimation_method
from ..scaler import OnlineScaler
from ..utils import calculate_effective_training_length, online_mean_update

if HAS_PANDAS:
    import pandas as pd
if HAS_POLARS:
    import polars as pl  # noqa


class OnlineGamlss(Estimator):
    """The online/incremental GAMLSS class."""

    def __init__(
        self,
        distribution: Distribution,
        equation: Dict,
        forget: float | Dict[int, float] = 0.0,
        method: Union[
            str, EstimationMethod, Dict[int, str], Dict[int, EstimationMethod]
        ] = "ols",
        scale_inputs: bool | np.ndarray = True,
        fit_intercept: Union[bool, Dict[int, bool]] = True,
        regularize_intercept: Union[bool, Dict[int, bool]] = False,
        ic: Union[str, Dict] = "aic",
        model_selection: Literal["local_rss", "global_ll"] = "local_rss",
        prefit_initial: bool | int = False,
        prefit_update: bool | int = False,
        cautious_updates: bool = False,
        max_it_outer: int = 30,
        max_it_inner: int = 30,
        abs_tol_outer: float = 1e-3,
        abs_tol_inner: float = 1e-3,
        rel_tol_outer: float = 1e-5,
        rel_tol_inner: float = 1e-5,
        rss_tol_inner: float = 1.5,
        step_size: float | Dict[int, float] = 1.0,
        verbose: int = 0,
        debug: bool = False,
        param_order: np.ndarray | None = None,
    ):
        """The `OnlineGamlss()` provides the fit, update and predict methods for linear parametric GAMLSS models.

        For a response variable $Y$ which is distributed according to the distribution $\mathcal{F}(\\theta)$
        with the distribution parameters $\\theta$, we model:

        $$g_k(\\theta_k) = \\eta_k = X_k\\beta_k$$

        where $g_k(\cdot)$ is a link function, which ensures that the predicted distribution parameters are in a
        sensible range (we don't want, e.g. negative standard deviations), and $\eta_k$ is the predictor (on the
        space of the link function). The model is fitted using iterative re-weighted least squares (IRLS).


        !!! note Tips and Tricks
            If you're facing issues with non-convergence and/or matrix inversion problems, please enable the `debug` mode and increase the
            logging level by increasing `verbose`.
            In debug mode, the estimator will save the weights, working vectors, derivatives each iteration in a
            according dictionary, i.e. self._debug_weights.
            The keys are composed of a tuple of ints of `(parameter, outer_iteration, inner_iteration)`.
            Very small and/or very large weights (implicitly second derivatives) can be a sign that either start values are not chosen appropriately or
            that the distributional assumption does not fit the data well.

        !!! warning Debug Mode
            Please don't use debug more for production models since it saves the `X` matrix and its scaled counterpart, so you will get large
            estimator objects.

        Args:
            distribution (rolch.Distribution): The parametric distribution.
            equation (Dict): The modelling equation. Follows the schema `{parameter[int]: column_identifier}`, where column_identifier can be either the strings `'all'`, `'intercept'` or a np.array of ints indicating the columns.
            forget (Union[float, Dict[int, float]], optional): The forget factor. Defaults to 0.0.
            method (Union[str, EstimationMethod, Dict[int, str], Dict[int, EstimationMethod]], optional): The estimation method. Defaults to "ols".
            scale_inputs (bool, optional): Whether to scale the input matrices. Defaults to True.
            fit_intercept (Union[bool, Dict[int, bool]], optional): Whether to fit an intercept. Defaults to True.
            regularize_intercept (Union[bool, Dict[int, bool]], optional): Whether to regularize the intercept. Defaults to False.
            ic (Union[str, Dict], optional): Information criterion for model selection. Defaults to "aic".
            max_it_outer (int, optional): Maximum outer iterations for the RS algorithm. Defaults to 30.
            max_it_inner (int, optional): Maximum inner iterations for the RS algorithm. Defaults to 30.
            abs_tol_outer (float, optional): Absolute tolerance on the deviance in the outer fit. Defaults to 1e-3.
            abs_tol_inner (float, optional): Absolute tolerance on the deviance in the inner fit. Defaults to 1e-3.
            rel_tol_outer (float, optional): Relative tolerance on the deviance in the outer fit. Defaults to 1e-5.
            rel_tol_inner (float, optional): Relative tolerance on the deviance in the inner fit. Defaults to 1e-5.
            rss_tol_inner (float, optional): Tolerance for increasing RSS in the inner fit. Defaults to 1.5.
            verbose (int, optional): Verbosity level. Level 0 will print no messages. Level 1 will print messages according to the start and end of each fit / update call and on finished outer iterations. Level 2 will print messages on each parameter fit in each outer iteration. Level 3 will print messages on each inner iteration. Defaults to 0.
            debug (bool, optional): Enable debug mode. Debug mode will save additional data to the estimator object.
                Currently, we save

                    * self._debug_X_dict
                    * self._debug_X_scaled
                    * self._debug_weights
                    * self._debug_working_vectors
                    * self._debug_dl1dlp1
                    * self._debug_dl2dlp2
                    * self._debug_eta

                to the the estimator. Debug mode works in batch and online settings. Note that debug mode is not recommended for production use. Defaults to False.
        """
        self.distribution = distribution
        self.equation = self._process_equation(equation)
        self._process_attribute(fit_intercept, True, "fit_intercept")
        self._process_attribute(regularize_intercept, False, "regularize_intercept")
        self._process_attribute(forget, default=0.0, name="forget")
        self._process_attribute(ic, "aic", name="ic")
        self.param_order = param_order

        if self.param_order is None:
            self._param_order = np.arange(self.distribution.n_params)
        else:
            if all(i in self.param_order for i in range(self.distribution.n_params)):
                self._param_order = self.param_order
            else:
                raise ValueError("All parameters should be in the param_order.")

        # Get the estimation method
        self._process_attribute(method, default="ols", name="method")
        self._method = {p: get_estimation_method(m) for p, m in self.method.items()}
        self.model_selection = model_selection

        self.scaler = OnlineScaler(to_scale=scale_inputs)
        self.do_scale = scale_inputs

        # These are global for all distribution parameters
        self.max_it_outer = max_it_outer
        self.max_it_inner = max_it_inner
        self.min_it_outer = 1
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner
        self.rss_tol_inner = rss_tol_inner
        self._process_attribute(step_size, default=1.0, name="step_size")
        self.cautious_updates = cautious_updates
        self.cond_start_val = False
        self.debug = debug
        self.verbose = verbose

        self.is_regularized = {}

        self.prefit_initial = int(prefit_initial)
        self.prefit_update = int(prefit_update)

    @property
    def betas(self):
        warnings.warn(
            "OnlineGamlss.betas is depreciated in favour of OnlineGamlss.beta to be consistent with beta_path. "
            "Alternatively, use OnlineGamlss.coef_ as in sklearn.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.beta

    def _print_message(self, message, level=0):
        if level <= self.verbose:
            print(f"[{self.__class__.__name__}]", message)

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

    def _count_nonzero_coef(self, exclude: int | np.ndarray | None = None) -> int:
        if exclude is None:
            gen = range(self.distribution.n_params)
        else:
            gen = np.delete(np.arange(self.distribution.n_params), exclude)
        return sum(np.count_nonzero(self.beta[p]) for p in gen)

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
            elif isinstance(self.equation[p], np.ndarray):
                if np.issubdtype(self.equation[p].dtype, bool):
                    if self.equation[p].shape[0] != X.shape[1]:
                        raise ValueError(f"Shape does not match for param {p}.")
                    J[p] = np.sum(self.equation[p]) + int(self.fit_intercept[p])
                elif np.issubdtype(self.equation[p].dtype, np.integer):
                    if self.equation[p].max() >= X.shape[1]:
                        raise ValueError(f"Shape does not match for param {p}.")
                    J[p] = self.equation[p].shape[0] + int(self.fit_intercept[p])
                else:
                    raise ValueError(
                        "If you pass a np.ndarray in the equation, "
                        "please make sure it is of dtype bool or int."
                    )
            elif isinstance(self.equation[p], list):
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

    def fit_select_model(
        self,
        X,
        y,
        wv,
        wt,  # do we need this?!
        w,
        beta_path,
        param,
    ) -> np.ndarray:
        # TODO Here:
        # - We should save the information in a named tuple
        # - We should keep only the information we need
        # - Consider the interation with the local RSS criterion?!
        f = init_forget_vector(self.forget[param], self.n_observations)
        n_nonzero_coef = np.count_nonzero(beta_path, axis=1)
        n_nonzero_coef_other = self._count_nonzero_coef(exclude=param)

        prediction_path = X @ beta_path.T

        if self.model_selection == "local_rss":
            residuals = wv[:, None] - prediction_path
            rss = np.sum(residuals**2 * w[:, None] * wt[:, None] * f[:, None], axis=0)
            rss = rss / np.mean(wt * w * f)
            ic = InformationCriterion(
                n_observations=self.n_training[param],
                n_parameters=n_nonzero_coef + n_nonzero_coef_other,
                criterion=self.ic[param],
            ).from_rss(rss=rss)
            best_ic = np.argmin(ic)
            beta = beta_path[best_ic, :]
            model_selection_data = rss

        elif self.model_selection == "global_ll":
            ll = np.zeros(self._method[param]._path_length)
            theta = np.copy(self.fv)
            for i in range(self._method[param]._path_length):
                theta[:, param] = self.distribution.link_inverse(
                    prediction_path[:, i], param=param
                )
                ll[i] = np.sum(f * self.distribution.logpdf(y, theta))

            ic = InformationCriterion(
                n_observations=self.n_training[param],
                n_parameters=n_nonzero_coef + n_nonzero_coef_other,
                criterion=self.ic[param],
            ).from_ll(log_likelihood=ll)
            best_ic = np.argmin(ic)
            beta = beta_path[best_ic, :]
            model_selection_data = ll

        return beta, model_selection_data, best_ic

    def update_select_model(
        self,
        X: np.ndarray,
        y: np.ndarray,  # observations / response
        wv: np.ndarray,  # working vector
        wt: np.ndarray,  # working weights
        w: np.ndarray,  # sample weights
        beta_path: np.ndarray,
        model_selection_data: Any,
        param: int,
    ):
        f = init_forget_vector(self.forget[param], y.shape[0])
        n_nonzero_coef = np.count_nonzero(beta_path, axis=1)
        prediction_path = X @ beta_path.T
        if self.model_selection == "local_rss":
            # Denominator
            # TODO: This should go in the online mean update function.
            denom = online_mean_update(
                self.mean_of_weights[param],
                wt,
                self.forget[param],
                self.n_observations,
            )
            residuals = wv[:, None] - prediction_path
            rss = (
                np.sum((residuals**2) * w[:, None] * wt[:, None] * f[:, None], axis=0)
                + (1 - self.forget[param]) ** y.shape[0]
                * (model_selection_data * self.mean_of_weights[param])
            ) / denom
            n_nonzero_coef_other = self._count_nonzero_coef(exclude=param)
            ic = InformationCriterion(
                n_observations=self.n_training[param],
                n_parameters=n_nonzero_coef + n_nonzero_coef_other,
                criterion=self.ic[param],
            ).from_rss(rss=rss)
            best_ic = np.argmin(ic)
            model_selection_data_new = rss
        elif self.model_selection == "global_ll":
            ll = np.zeros(self._method[param]._path_length)

            theta = np.copy(self.fv)
            for i in range(self._method[param]._path_length):
                theta[:, param] = self.distribution.link_inverse(
                    prediction_path[:, i], param=param
                )
                ll[i] = np.sum(w * f * self.distribution.logpdf(y, theta))
            ll = ll + (1 - self.forget[param]) ** y.shape[0] * model_selection_data

            ic = InformationCriterion(
                n_observations=self.n_training[param],
                n_parameters=n_nonzero_coef,
                criterion=self.ic[param],
            ).from_ll(log_likelihood=ll)
            best_ic = np.argmin(ic)
            model_selection_data_new = ll

        beta = beta_path[best_ic, :]
        return beta, model_selection_data_new, best_ic

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate the input matrices X and y.

        Args:
            X (np.ndarray): Input matrix $X$
            y (np.ndarray): Response vector $y$.

        Raises:
            OutOfSupportError: If the values of $y$ are below the range of the distribution.
            OutOfSupportError: If the values of $y$ are beyond the range of the distribution.
            ValueError: If `X` and `y` are not the same length.
            ValueError: If `X` or `y` contain NaN values.
            ValueError: If `X` or `y` contain infinite values.
        """
        if np.any(y < self.distribution.distribution_support[0]):
            raise OutOfSupportError(
                message=(
                    "y contains values below the distribution's support. "
                    f"The smallest value in y is {np.min(y)}. "
                    f"The support of the distribution is {str(self.distribution.distribution_support)}."
                )
            )
        if np.any(y > self.distribution.distribution_support[1]):
            raise OutOfSupportError(
                message=(
                    "Y contains values larger than the distribution's support. "
                    f"The smallest value in y is {np.max(y)}. "
                    f"The support of the distribution is {str(self.distribution.distribution_support)}."
                ),
            )

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y should have the same length.")

        if np.any(np.isnan(y)) or np.any(np.isnan(X)):
            raise ValueError("X and y should not contain Nan.")

        if not (np.all(np.isfinite(y)) & np.all(np.isfinite(X))):
            raise ValueError("X and y should contain only finite values.")

    def _make_initial_fitted_values(self, y: np.ndarray) -> np.ndarray:
        out = np.stack(
            [
                self.distribution.initial_values(y, param=i)
                for i in range(self.distribution.n_params)
            ],
            axis=1,
        )
        return out

    def _make_iter_schedule(self):
        return np.repeat(self.max_it_inner, repeats=self.max_it_outer)

    def _make_step_size_schedule(self):
        return np.tile(
            A=list(self.step_size.values()),
            reps=(self.max_it_outer, self.max_it_inner, 1),
        ).astype(float)

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

        self._validate_inputs(X, y)
        self.n_observations = y.shape[0]
        self.n_training = {
            p: calculate_effective_training_length(self.forget[p], self.n_observations)
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

        for p, x in X_dict.items():
            cond_num = np.linalg.cond(x)
            if cond_num > 100:
                message = (
                    f"Condition number of X for after scaling (and adding an intercept) for param {p} is {cond_num}. "
                    "This might lead to numerical issues. Consider using a regularized estimation method."
                )
                self._print_message(message=message, level=1)

        if self.debug:
            self._debug_X_dict = X_dict
            self._debug_X_scaled = X_scaled
            self._debug_weights = {}
            self._debug_working_vectors = {}
            self._debug_dl1dlp1 = {}
            self._debug_dl2dlp2 = {}
            self._debug_eta = {}
            self._debug_fv = {}
            self._debug_dv = {}

        self.x_gram = {}
        self.y_gram = {}
        self.residuals = np.zeros((self.n_observations, self.distribution.n_params))
        self.rss = np.zeros((self.distribution.n_params))
        self.rss_iterations = np.zeros(
            (self.distribution.n_params, self.max_it_outer, self.max_it_inner)
        )

        # For regularized estimation, we have the necessary data to do model selection!
        self.model_selection_data = {}
        self.best_ic = np.zeros(self.distribution.n_params)
        self.best_ic_iterations = np.full(
            (self.distribution.n_params, self.max_it_outer, self.max_it_inner),
            np.nan,
        )

        for p in range(self.distribution.n_params):
            is_regularized = np.repeat(True, self.J[p])
            if self.fit_intercept[p] and not (
                self.regularize_intercept[p] | self._is_intercept_only(p)
            ):
                is_regularized[0] = False
            self.is_regularized[p] = is_regularized

        # Betas might be different across distribution parameters
        # So this is a dict of dicts
        self.beta_iterations = {i: {} for i in range(self.distribution.n_params)}
        self.beta_iterations_inner = {i: {} for i in range(self.distribution.n_params)}

        self.beta_path = {p: None for p in range(self.distribution.n_params)}
        self.beta = {p: np.zeros(self.J[p]) for p in range(self.distribution.n_params)}

        self.beta_path_iterations_inner = {
            i: {} for i in range(self.distribution.n_params)
        }
        self.beta_path_iterations = {i: {} for i in range(self.distribution.n_params)}

        # We need to track the sum of weights for each
        # distribution parameter for online model selection
        self.sum_of_weights = {}
        self.mean_of_weights = {}

        self.schedule_iteration = self._make_iter_schedule()
        self.schedule_step_size = self._make_step_size_schedule()

        if self.prefit_initial > 0:
            message = (
                f"Setting max_it_inner to {self.prefit_initial} for first iteration"
            )
            self._print_message(message=message, level=1)
            self.schedule_iteration[: self.prefit_initial] = 1
            self.current_min_it_outer = int(self.prefit_initial)
        else:
            self.current_min_it_outer = int(self.min_it_outer)

        message = "Starting fit call"
        self._print_message(message=message, level=1)
        (
            self.global_dev,
            self.it_outer,
        ) = self._outer_fit(
            X=X_dict,
            y=y,
            w=w,
        )
        message = "Finished fit call"
        self._print_message(message=message, level=1)

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

        self._validate_inputs(X, y)
        if sample_weight is not None:
            w = sample_weight  # Align to sklearn API
        else:
            w = np.ones(y.shape[0])

        self.n_observations += y.shape[0]
        self.n_training = {
            p: calculate_effective_training_length(self.forget[p], self.n_observations)
            for p in range(self.distribution.n_params)
        }

        self.fv = self.predict(X, what="response")

        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }

        if self.debug:
            self._debug_X_dict = X_dict
            self._debug_X_scaled = X_scaled
            self._debug_weights = {}
            self._debug_working_vectors = {}
            self._debug_dl1dlp1 = {}
            self._debug_dl2dlp2 = {}
            self._debug_eta = {}

        ## Reset rss and ic to avoid confusion
        ## These are only for viewing, not read!!
        self.best_ic[:] = 0
        self.best_ic_iterations[:] = 0
        self.rss_iterations[:] = 0

        ## Copy old values for updating
        self.x_gram_inner = copy.copy(self.x_gram)
        self.y_gram_inner = copy.copy(self.y_gram)
        self.rss_old = copy.copy(self.rss)
        self.sum_of_weights_inner = copy.copy(self.sum_of_weights)
        self.mean_of_weights_inner = copy.copy(self.mean_of_weights)
        self.model_selection_data_old = copy.copy(self.model_selection_data)

        # Clean schedules for iterations and step size
        self.schedule_iteration = self._make_iter_schedule()
        self.schedule_step_size = self._make_step_size_schedule()

        if self.prefit_update > 0:
            message = (
                f"Setting max_it_inner to {self.prefit_update} for first iteration"
            )
            self._print_message(message=message, level=1)
            self.schedule_iteration[: self.prefit_update] = 1
            self.current_min_it_outer = int(self.prefit_update)

        if self.cautious_updates:
            lower = self.distribution.quantile(0.005, self.fv).item()
            upper = self.distribution.quantile(0.995, self.fv).item()

            if np.any(y < lower) or np.any(y > upper):
                message = (
                    f"New observations are outliers given current estimates. "
                    f"y: {y}, Lower bound: {lower}, upper bound: {upper}"
                )
                self._print_message(message=message, level=1)
                self.schedule_iteration[:4] = 1
                self.schedule_step_size[:4, :, :] = 0.25
                self.current_min_it_outer = 4
            else:
                self.current_min_it_outer = int(self.min_it_outer)

        message = "Starting update call"
        self._print_message(message=message, level=1)
        (
            self.global_dev,
            self.it_outer,
        ) = self._outer_update(
            X=X_dict,
            y=y,
            w=w,
        )

        self.x_gram = copy.copy(self.x_gram_inner)
        self.y_gram = copy.copy(self.y_gram_inner)
        self.sum_of_weights = copy.copy(self.sum_of_weights_inner)
        self.mean_of_weights = copy.copy(self.mean_of_weights_inner)
        message = "Finished update call"
        self._print_message(message=message, level=1)

    def _outer_update(self, X, y, w):
        ## for new observations:
        global_di = np.sum(-2 * self.distribution.logpdf(y, self.fv) * w)
        global_dev = (1 - self.forget[0]) ** y.shape[0] * self.global_dev + global_di
        global_dev_old = global_dev + 1000
        it_outer = 0

        while True:
            # Check relative congergence
            if it_outer >= self.current_min_it_outer:
                if (
                    np.abs(global_dev_old - global_dev) / np.abs(global_dev_old)
                    < self.rel_tol_outer
                ):
                    break

                if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                    break

                # if global_dev > global_dev_old:
                #     message = (
                #         f"Outer iteration {it_outer}: Global deviance increased. Breaking."
                #         f"Current LL {global_dev}, old LL {global_dev_old}"
                #     )
                #     self._print_message(message=message, level=1)
                #     break

                if it_outer >= self.max_it_outer:
                    break

            global_dev_old = global_dev
            it_outer += 1

            for param in self._param_order:

                self.beta_iterations_inner[param][it_outer] = {}
                self.beta_path_iterations_inner[param][it_outer] = {}

                global_dev = self._inner_update(
                    X=X,
                    y=y,
                    w=w,
                    it_outer=it_outer,
                    param=param,
                    dv=global_dev,
                )
                message = f"Outer iteration {it_outer}: Fitted param {param}: Current LL {global_dev}"
                self._print_message(message=message, level=2)

            message = f"Outer iteration {it_outer}: Finished: current LL {global_dev}"
            self._print_message(message=message, level=1)

        return global_dev, it_outer

    def _outer_fit(self, X, y, w):

        global_di = -2 * self.distribution.logpdf(y, self.fv)
        global_dev = np.sum(w * global_di)
        global_dev_old = global_dev + 1000
        it_outer = 0

        while True:
            # Check relative congergence
            if it_outer >= self.current_min_it_outer:
                if (
                    np.abs(global_dev_old - global_dev) / np.abs(global_dev_old)
                    < self.rel_tol_outer
                ):
                    break

                if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                    break

                if it_outer >= self.max_it_outer:
                    break

                if global_dev > global_dev_old:
                    message = (
                        f"Outer iteration {it_outer}: Global deviance increased. Breaking."
                        f"Current LL {global_dev}, old LL {global_dev_old}"
                    )
                    self._print_message(message=message, level=0)
                    break

            global_dev_old = float(global_dev)
            it_outer += 1

            for param in self._param_order:

                self.beta_iterations_inner[param][it_outer] = {}
                self.beta_path_iterations_inner[param][it_outer] = {}

                global_dev = self._inner_fit(
                    X=X,
                    y=y,
                    w=w,
                    param=param,
                    it_outer=it_outer,
                    dv=global_dev,
                )

                self.beta_iterations[param][it_outer] = self.beta[param]
                self.beta_path_iterations[param][it_outer] = self.beta_path[param]

                message = f"Outer iteration {it_outer}: Fitted param {param}: current LL {global_dev}"
                self._print_message(message=message, level=2)

            message = f"Outer iteration {it_outer}: Finished. Current LL {global_dev}, old LL {global_dev_old}"
            self._print_message(message=message, level=1)

        return (global_dev, it_outer)

    def _inner_fit(
        self,
        X,
        y,
        w,
        it_outer,
        param,
        dv,
    ):

        dv_start = np.sum(-2 * self.distribution.logpdf(y, self.fv) * w)
        dv_iterations = np.repeat(dv_start, self.max_it_inner + 1)
        fv_it = copy.copy(self.fv)
        fv_it_new = copy.copy(self.fv)
        step_decrease_counter = 0
        terminate = False
        bad_state = False

        message = f"Starting inner iteration param {param}, outer iteration {it_outer}, start DV {dv_start}"
        self._print_message(message=message, level=2)

        for it_inner in range(self.schedule_iteration[it_outer - 1]):
            # We can improve the fit by taking the conditional
            # start values for the first outer iteration and the first inner iteration
            # as soon the first parameter is fitted.
            step_it = self.schedule_step_size[it_outer - 1, it_inner, param]

            if (
                (it_inner == 0)
                and (it_outer == 1)
                and (param >= 1)
                and self.cond_start_val
            ):
                fv_it = self.distribution.calculate_conditional_initial_values(
                    y=y,
                    theta=fv_it,
                    param=param,
                )
                message = f"Outer iteration {it_outer}: Fitting Parameter {param}: Using conditional start value {fv_it[0, param]}."
                self._print_message(message=message, level=3)

            eta = self.distribution.link_function(fv_it[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, fv_it, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv_it, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            if self.debug:
                key = (param, it_outer, it_inner)
                self._debug_weights[key] = copy.copy(wt)
                self._debug_working_vectors[key] = copy.copy(wv)
                self._debug_dl1dlp1[key] = copy.copy(dl1dp1)
                self._debug_dl2dlp2[key] = copy.copy(dl2dp2)
                self._debug_eta[key] = copy.copy(eta)
                self._debug_fv[key] = copy.copy(fv_it)
                self._debug_dv[key] = float(dv_iterations[it_inner])

            ## Create the X and Y Gramian and the weight
            x_gram_it = self._method[param].init_x_gram(
                X=X[param],
                weights=(w * wt),
                forget=self.forget[param],
            )
            y_gram_it = self._method[param].init_y_gram(
                X=X[param],
                y=wv,
                weights=(w * wt),
                forget=self.forget[param],
            )
            # Select the model if we have a path-based method
            if self._method[param]._path_based_method:
                if (it_inner == 0) & (it_outer == 1):
                    beta_path_it = self._method[param].fit_beta_path(
                        x_gram=x_gram_it,
                        y_gram=y_gram_it,
                        is_regularized=self.is_regularized[param],
                    )
                elif it_inner > 0:
                    beta_path_it = self._method[param].update_beta_path(
                        x_gram=x_gram_it,
                        y_gram=y_gram_it,
                        is_regularized=self.is_regularized[param],
                        beta_path=beta_path_it,
                    )
                elif (it_outer > 1) & (it_inner == 0):
                    beta_path_it = self._method[param].update_beta_path(
                        x_gram=x_gram_it,
                        y_gram=y_gram_it,
                        is_regularized=self.is_regularized[param],
                        beta_path=self.beta_path[param],
                    )
                else:
                    print("This should not happen")

                beta_it, model_selection_data_it, best_ic_it = self.fit_select_model(
                    X=X[param],
                    y=y,
                    w=w,
                    wv=wv,
                    wt=wt,
                    beta_path=beta_path_it,
                    param=param,
                )
                self.best_ic_iterations[param, it_outer - 1, it_inner] = best_ic_it

            else:
                beta_path_it = None
                beta_it = self._method[param].fit_beta(
                    x_gram=x_gram_it,
                    y_gram=y_gram_it,
                    is_regularized=self.is_regularized[param],
                )

            # Calculate the prediction, residuals and RSS
            f = init_forget_vector(self.forget[param], self.n_observations)
            prediction_it = step_it * (X[param] @ beta_it.T) + (1 - step_it) * eta
            residuals_it = wv - prediction_it
            rss_it = np.sum(residuals_it**2 * wt * w * f) / np.mean(wt * w * f)

            # Calculate the fitted values and the deviance
            fv_it_new[:, param] = self.distribution.link_inverse(
                prediction_it, param=param
            )
            dv_it = np.sum(-2 * self.distribution.logpdf(y, fv_it_new) * w)
            dv_old = dv_iterations[it_inner]
            dv_increasing = dv_it > dv_old

            # This should really not happen unless your start values are
            # way to good and the model cannot reach these
            bad_state = dv_it > dv_iterations[0]

            # print(dv_it, dv_old, dv_increasing)
            message = f"Outer iteration {it_outer}: Fitting Parameter {param}: Inner iteration {it_inner}: Current Deviance {dv_it}"
            self._print_message(message=message, level=3)

            if dv_increasing and it_inner < (self.schedule_iteration[it_outer - 1] - 1):
                # print("Blabal")
                step_decrease_counter += 1
                self.schedule_step_size[it_outer - 1, it_inner + 1, param] = step_it / 2
                self._print_message(
                    f"Deviance increasing, step size halved. {step_decrease_counter}",
                    level=1,
                )
                if step_decrease_counter > 5:
                    message = f"Step size too small. Parameter {param}, Outer iteration {it_outer}, Inner iteration {it_inner}."
                    self._print_message(message=message, level=1)
                    terminate = True

            if (it_outer == 1) & (it_inner >= 1) | (it_outer >= 2):
                # Allow to break in principle.
                if abs(dv_old - dv_it) <= self.abs_tol_inner:
                    terminate = True
                if abs(dv_old - dv_it) / abs(dv_old) < self.rel_tol_inner:
                    terminate = True
                if it_inner == (self.max_it_inner - 1):
                    message = f"Reached max inner iteration in inner fit. Parameter:{param}, Outer iteration: {it_outer}, Inner iteration: {it_inner}."
                    self._print_message(message=message, level=3)
                    terminate = True

            if terminate and (it_outer == 1) and bad_state:
                message = (
                    f"The model ended in a bad state in the first outer iteration of param {param}. This is not a good sign.  \n"
                    f"The deviance increased from the start values to the current fit in inner iteration {it_inner}. \n"
                    f"The deviance is {dv_it} and the starting deviance for this inner iterations is {dv_iterations[0]}. \n"
                    "Please check your data and model. \n"
                    "Please turn on logging (verbose=3, debug=True) and check the debug information. \n"
                    "Consider using a pre-fit via the iteration_schedule and set the inner iterations to 1-2 for the first outer iteration."
                )
                self._print_message(message=message, level=0)

            if terminate:
                break
            if not dv_increasing:
                # print("deviance decreasing, write to fv_it")
                fv_it[:, param] = fv_it_new[:, param]
                # Set the deviance for the next inner iteration
                dv_iterations[(it_inner + 1) :] = dv_it

        if (not bad_state) or (it_outer == 1):
            # Write everything to the class
            self.x_gram[param] = x_gram_it
            self.y_gram[param] = y_gram_it
            self.beta[param] = beta_it
            self.beta_path[param] = beta_path_it
            self.fv[:, param] = fv_it_new[:, param]

            # Sum and mean of the weights
            self.sum_of_weights[param] = np.sum(w * wt)
            self.mean_of_weights[param] = np.mean(w * wt)

            # RSS
            self.rss[param] = rss_it
            self.rss_iterations[param, it_outer - 1, it_inner - 1] = rss_it

            # TODO: Think where this should go (at all?)
            # # Check if the local RSS are decreasing
            # if (it_inner > 1) or (it_outer > 1):
            #     if rss_it > (self.rss_tol_inner * self.rss[param]):
            #         message = f"Inner iteration {it_inner}: Fitting Parameter {param}: Current RSS {rss_it} > {self.rss_tol_inner} * {self.rss[param]}"
            #         self._print_message(message=message, level=3)
            #         break

            if self._method[param]._path_based_method:
                self.model_selection_data[param] = model_selection_data_it
                self.best_ic[param] = best_ic_it

        return dv_it

    def _inner_update(
        self,
        X,
        y,
        w,
        it_outer,
        dv,
        param,
    ):
        # di = -2 * self.distribution.logpdf(y, self.fv)
        # dv = (1 - self.forget[0]) * self.global_dev + np.sum(di * w)
        # olddv = dv + 1

        dv_start = (
            np.sum(-2 * self.distribution.logpdf(y, self.fv) * w)
            + (1 - self.forget[0]) ** y.shape[0]
            * self.global_dev  # global dev is previous observation / fit
        )
        dv_iterations = np.repeat(dv_start, self.max_it_inner + 1)
        fv_it = copy.copy(self.fv)
        fv_it_new = copy.copy(self.fv)
        step_decrease_counter = 0
        terminate = False

        for it_inner in range(self.schedule_iteration[it_outer - 1]):
            step_it = self.schedule_step_size[it_outer - 1, it_inner, param]

            eta = self.distribution.link_function(fv_it[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, fv_it, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, fv_it, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, -1e10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            if self.debug:
                key = (param, it_outer, it_inner)
                self._debug_weights[key] = copy.copy(wt)
                self._debug_working_vectors[key] = copy.copy(wv)
                self._debug_dl1dlp1[key] = copy.copy(dl1dp1)
                self._debug_dl2dlp2[key] = copy.copy(dl2dp2)
                self._debug_eta[key] = copy.copy(eta)
                # self._debug_x[key] = copy.copy(X[param])

            x_gram_it = self._method[param].update_x_gram(
                gram=self.x_gram[param],
                X=X[param],
                weights=(w * wt),
                forget=self.forget[param],
            )
            y_gram_it = self._method[param].update_y_gram(
                gram=self.y_gram[param],
                X=X[param],
                y=wv,
                weights=(w * wt),
                forget=self.forget[param],
            )
            # Select the model if we have a path-based method
            if self._method[param]._path_based_method:
                beta_path_it = self._method[param].update_beta_path(
                    x_gram=x_gram_it,
                    y_gram=y_gram_it,
                    beta_path=self.beta_path[param],
                    is_regularized=self.is_regularized[param],
                )

                beta_it, model_selection_data_it, best_ic_it = self.update_select_model(
                    X=X[param],
                    y=y,
                    w=w,
                    wv=wv,
                    wt=wt,
                    beta_path=beta_path_it,
                    model_selection_data=self.model_selection_data_old[param],
                    param=param,
                )
                self.best_ic[param] = best_ic_it
                self.best_ic_iterations[param, it_outer - 1, it_inner] = best_ic_it
            else:
                beta_it = self._method[param].update_beta(
                    x_gram=x_gram_it,
                    y_gram=y_gram_it,
                    beta=self.beta[param],
                    is_regularized=self.is_regularized[param],
                )
                beta_path_it = None
                model_selection_data_it = None

            # Calculate the prediction, residuals and RSS
            f = init_forget_vector(self.forget[param], y.shape[0])
            prediction_it = X[param] @ beta_it.T
            residuals_it = wv - prediction_it
            rss_it = np.sum(residuals_it**2 * wt * w * f) / np.mean(wt * w * f)

            denom = online_mean_update(
                self.mean_of_weights[param],
                wt,
                self.forget[param],
                self.n_observations,
            )
            sum_of_rss_it = (
                rss_it
                + (1 - self.forget[param]) ** y.shape[0]
                * (self.rss_old[param] * self.mean_of_weights[param])
            ) / denom

            # TODO: Do we need the rss_tol_inner here?
            # if (it_inner > 1) or (it_outer > 1):
            #     if sum_of_rss_it > (self.rss_tol_inner * self.rss[param]):
            #         print("Breaking RSS in Update step.")
            #         break

            # Calculate the fitted values and the deviance
            fv_it_new[:, param] = self.distribution.link_inverse(
                prediction_it, param=param
            )
            dv_it = (
                np.sum(-2 * self.distribution.logpdf(y, fv_it_new) * w)
                + (1 - self.forget[0]) ** y.shape[0] * self.global_dev
            )
            dv_old = dv_iterations[it_inner]
            dv_increasing = dv_it > dv_old

            if dv_increasing:
                step_decrease_counter += 1
                self.schedule_step_size[it_outer - 1, it_inner, param] = step_it / 2
                self._print_message(
                    f"Deviance increasing, step size halved. {step_decrease_counter}",
                    level=1,
                )
                if step_decrease_counter > 5:
                    message = f"Step size too small. Parameter {param}, Outer iteration {it_outer}, Inner iteration {it_inner}."
                    self._print_message(message=message, level=1)
                    terminate = True

            if (not dv_increasing) | (it_inner < self.max_it_inner - 1):
                # Allow to break in principle.
                if abs(dv_old - dv_it) <= self.abs_tol_inner:
                    terminate = True
                if abs(dv_old - dv_it) / abs(dv_old) < self.rel_tol_inner:
                    terminate = True
                if it_inner == (self.max_it_inner - 1):
                    message = f"Reached max inner iteration in inner fit. Parameter:{param}, Outer iteration: {it_outer}, Inner iteration: {it_inner}."
                    self._print_message(message=message, level=3)
                    terminate = True

            message = f"Outer iteration {it_outer}: Fitting Parameter {param}: Inner iteration {it_inner}: Current DV {dv_it}, previous DV {dv_old}, step size {step_it}"
            self._print_message(message=message, level=3)

            if terminate:
                break
            if not dv_increasing:
                # print("deviance decreasing, write to fv_it")
                fv_it[:, param] = fv_it_new[:, param]
                # Set the deviance for the next inner iteration
                dv_iterations[(it_inner + 1) :] = dv_it

            # olddv = dv
            # di = -2 * self.distribution.logpdf(y, self.fv)
            # dv = np.sum(di * w)

        # Assign to class variables
        self.x_gram_inner[param] = x_gram_it
        self.y_gram_inner[param] = y_gram_it
        self.fv[:, param] = fv_it_new[:, param]

        self.beta[param] = beta_it
        self.beta_path[param] = beta_path_it
        self.rss[param] = sum_of_rss_it
        self.rss_iterations[param, it_outer - 1, it_inner] = sum_of_rss_it
        self.model_selection_data[param] = model_selection_data_it

        # Update the weights
        self.sum_of_weights_inner[param] = (
            np.sum(w * wt) + (1 - self.forget[param]) * self.sum_of_weights[param]
        )
        self.mean_of_weights_inner[param] = (
            self.sum_of_weights_inner[param] / self.n_training[param]
        )
        return dv_it

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
        X_scaled = self.scaler.transform(X=X)
        X_dict = {
            p: self.make_model_array(X_scaled, p)
            for p in range(self.distribution.n_params)
        }
        prediction = [x @ b.T for x, b in zip(X_dict.values(), self.beta.values())]

        if return_contributions:
            contribution = [
                x * b.T for x, b in zip(X_dict.values(), self.beta.values())
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

    def predict_quantile(
        self,
        X: np.ndarray,
        quantile: float | np.ndarray,
    ) -> np.ndarray:
        """Predict the quantile(s) of the distribution.

        Args:
            X (np.ndarray): Covariate matrix $X$. Shape should be (n_samples, n_features).
            quantile (float | np.ndarray): Quantile(s) to predict.

        Returns:
            np.ndarray: Predicted quantile(s) of the distribution. Shape will be (n_samples, n_quantiles).
        """
        theta = self.predict(X, what="response")
        if isinstance(quantile, np.ndarray):
            quantile_pred = self.distribution.ppf(quantile[:, None], theta).T
        else:
            quantile_pred = self.distribution.ppf(quantile, theta).reshape(-1, 1)

        return quantile_pred
