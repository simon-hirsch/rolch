import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

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


class LinkFunction(ABC):
    """The base class for the link functions."""

    @abstractmethod
    def link(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Link"""

    @abstractmethod
    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Calculate the inverse of the link function"""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate the first derivative of the link function"""


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
    def link_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        """Apply the derivative of the link function for param on y."""

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
        method: Union[str, Dict] = "lasso",
        forget: float = 0.0,
    ):
        self.method = method
        self.forget = forget
        self.distribution = distribution
        self.fit_intercept = fit_intercept
        self.equation = self.preprocess_equation(equation)

    def preprocess_equation(self, equation):
        if equation is None:
            warnings.warn(
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

    def make_model_array(self, X: Union[np.ndarray], param: int):
        eq = self.equation[param]

        # TODO: Add intercept here?
        # TODO: Check difference between np.array and list more explicitly?
        if eq == "all":
            if isinstance(X, np.ndarray):
                return X
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                return X.to_numpy()
            if HAS_POLARS and isinstance(X, pl.DataFrame):
                return X.to_numpy()
        elif eq == "intercept":
            return np.ones((X.shape[0], 1))
        elif isinstance(eq, np.ndarray) | isinstance(eq, list):
            if isinstance(X, np.ndarray):
                return X[:, eq]
            if HAS_PANDAS and isinstance(pd.DataFrame):
                return X.loc[:, eq]
            if HAS_POLARS and isinstance(pl.DataFrame):
                return X.select(eq).to_numpy()

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
