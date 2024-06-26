from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


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
