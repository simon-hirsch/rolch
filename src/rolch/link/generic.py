from typing import Tuple

import numpy as np

from rolch.base import LinkFunction


class GenericInverseLink(LinkFunction):
    """
    This link function maps an arbitrary link function to its inverse by swapping the links and derivatives.
    You need to provide the link function and the support.
    This can be used to quickly create inverse links or as base class if you want to implement a custom link function.
    The default will not provide the second derivative, which you need to implement in the subclass.
    """

    def __init__(
        self,
        link_function: LinkFunction,
        link_support: Tuple,
    ) -> None:
        """Initializes the GenericInverseLink class.

        Args:
            link_function (LinkFunction): The Link function to invert
            link_support (Tuple): The support of the link function
        """
        self.link_function = link_function
        self.link_support = link_support

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.link_support[0], self.link_support[1])

    def link(self, x: np.ndarray) -> np.ndarray:
        return self.link_function.inverse(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return self.link_function.link(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return self.link_function.inverse_derivative(x)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return self.link_function.link_derivative(x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return super().link_second_derivative(x)
