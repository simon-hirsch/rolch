from typing import Tuple

import numpy as np

from ..base import LinkFunction
from .robust_math import SMALL_NUMBER, robust_exp, robust_log


class LogitLink(LinkFunction):
    """The Logit Link function.

    The logit-link function is defined as \(g(x) = \log (x/ (1-x))\).
    """
    link_support = (np.nextafter(0, 1), np.nextafter(1, 0))
    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return robust_log(x/(1-x))
        #return np.log(np.fmax( x/(1-x), LOG_LOWER_BOUND) )

    def inverse(self, x: np.ndarray) -> np.ndarray:
        #exp_x_adj = np.exp( np.fmax ( np.fmin ( x, EXP_UPPER_BOUND) ,np.log(LOG_LOWER_BOUND)))
        #return exp_x_adj / (exp_x_adj + 1)
    
        return robust_exp/(robust_exp + 1)

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        #exp_adj = np.exp( np.fmin (x, EXP_UPPER_BOUND) )
        #return exp_adj / ( (exp_adj + 1) ** 2 )

        return robust_exp/ ( (robust_exp + 1) ** 2 )

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / ( x (1-x) )

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return - (1- 2*x) / ( (x**2) * ( (1 - x)**2 ))
