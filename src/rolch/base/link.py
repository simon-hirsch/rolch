from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class LinkFunction(ABC):
    """The base class for the link functions."""

    @property
    @abstractmethod
    def link_support(self) -> Tuple[float, float]:
        """The support of the distribution."""
        pass

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
    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate the second derivative for the link function"""
        raise NotImplementedError("Currently not implemented.")

    @abstractmethod
    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate the first derivative for the inverse link function"""
        raise NotImplementedError("Currently not implemented.")
