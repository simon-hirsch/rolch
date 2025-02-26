from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


class Distribution(ABC):

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Each subclass must define 'n_params'."""
        pass

    @property
    @abstractmethod
    def distribution_support(self) -> Tuple[float, float]:
        """The support of the distribution."""
        pass

    @property
    @abstractmethod
    def parameter_support(self) -> dict:
        """The support of each parameter of the distribution."""
        pass

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

    def _validate_dln_dpn_inputs(
        self, y: np.ndarray, theta: np.ndarray, param: int
    ) -> None:
        if param >= self.n_params:
            raise ValueError(
                f"{self.__class__.__name__} has only {self.n_params} distribution parameters.\nYou have passed {param}. Please remember we start counting at 0."
            )

    def _validate_dl2_dpp_inputs(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int]
    ) -> None:
        if max(params) >= self.n_params:
            raise ValueError(
                f"{self.__class__.__name__} has only {self.n_params} distribution parameters.\nYou have passed {params}. Please remember we start counting at 0."
            )
        if params[0] == params[1]:
            raise ValueError("Cross derivatives must use different parameters.")

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
