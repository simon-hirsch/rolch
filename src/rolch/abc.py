import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rolch import HAS_PANDAS, HAS_POLARS
from rolch.gram import (
    init_gram,
    init_inverted_gram,
    init_y_gram,
    update_gram,
    update_inverted_gram,
    update_y_gram,
)
from rolch.utils import handle_param_dict

if HAS_PANDAS:
    import pandas as pd
if HAS_POLARS:
    import polars as pl


class LinkFunction(ABC):
    """The base class for the link functions."""

    @abstractmethod
    def link(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Link"""

    @abstractmethod
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Calculate the inverse of the link function"""

    @abstractmethod
    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate the first derivative of the link function"""
        raise NotImplementedError("Currently not implemented. Will be needed for GLMs")

    @abstractmethod
    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate the first derivative for the inverse link function"""


class Distribution(ABC):

    @abstractmethod
    def theta_to_params(self, theta: np.ndarray) -> Tuple:
        """Take the fitted values and return tuple of vectors for distribution parameters."""

    @abstractmethod
    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int) -> np.ndarray:
        """Take the first derivative of the likelihood function with respect to the param."""

    @abstractmethod
    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int) -> np.ndarray:
        """Take the second derivative of the likelihood function with respect to the param."""

    @abstractmethod
    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int]
    ) -> np.ndarray:
        """Take the first derivative of the likelihood function with respect to both parameters."""

    @abstractmethod
    def link_function(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        """Apply the link function for param on y."""

    @abstractmethod
    def link_inverse(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        """Apply the inverse of the link function for param on y."""

    @abstractmethod
    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        """Apply the derivative of the link function for param on y."""

    @abstractmethod
    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        """Apply the derivative of the inverse link function for param on y."""

    @abstractmethod
    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: int = None
    ) -> np.ndarray:
        """Calculate the initial values for the GAMLSS fit."""

    @abstractmethod
    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """The cumulative density function."""

    @abstractmethod
    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """The density function."""

    @abstractmethod
    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """The percentage point or quantile function of the distribution."""

    @abstractmethod
    def rvs(self, size: Union[int, Tuple], theta: np.ndarray) -> np.ndarray:
        """Draw random samples of shape size."""


class Estimator(ABC):
    """
    Base class for estimators.
    """

    def __init__(
        self,
        distribution: Distribution,
        equation: Optional[Dict] = None,
        fit_intercept: Union[bool, Dict] = True,
        method: str = "lasso",
        forget: float = 0.0,
    ):
        self.method = method
        self.distribution = distribution
        self.equation = self.process_equation(equation)
        self.process_attributes(fit_intercept, True, "fit_intercept")
        self.process_attributes(forget, 0.0, "forget")

    def process_attributes(self, attribute: Any, default: Any, name: str) -> None:
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

    def process_equation(self, equation):
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

    def _is_intercept_only(self, param):
        if isinstance(self.equation[param], str):
            return self.equation[param] == "intercept"
        else:
            return False

    def _make_x_gram(self, x: np.ndarray, w: np.ndarray, param: int) -> np.ndarray:
        """
        Make the Gram matrix.

        Args:
            x (np.ndarray): Covariate matrix.
            w (np.ndarray): Weight matrix.
            param (int): Parameter index.

        Returns:
            np.ndarray: Gram matrix.
        """
        if self.method == "ols":
            return init_inverted_gram(X=x, w=w, forget=self.forget[param])
        elif self.method == "lasso":
            return init_gram(X=x, w=w, forget=self.forget[param])

    def _make_y_gram(
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

    def _update_x_gram(
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
        if self.method == "ols":
            return update_inverted_gram(gram, X=x, w=w, forget=self.forget[param])
        if self.method == "lasso":
            return update_gram(gram, X=x, w=w, forget=self.forget[param])

    def _update_y_gram(
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
