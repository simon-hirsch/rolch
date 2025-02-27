import copy

from ..base import EstimationMethod
from .lasso_path import LassoPathMethod
from .recursive_least_squares import OrdinaryLeastSquaresMethod


class EstimationMethodFactory:

    def __init__(self):
        pass

    @staticmethod
    def from_string(method: str):
        if method == "lasso":
            return LassoPathMethod()
        elif method == "ols":
            return OrdinaryLeastSquaresMethod()
        else:
            return ValueError("Did not recognize method. Please provide [ols, lasso].")


def get_estimation_method(method: EstimationMethod | str):
    if isinstance(method, str):
        out = EstimationMethodFactory().from_string(method=method)
    elif isinstance(method, EstimationMethod):
        # This is a safeguard. The user might initialise only one concrete class
        # and pass it to all estimation methods in multi-method estimators.
        # We want each parameter to have its individual estimation method.
        out = copy.copy(method)
    else:
        raise ValueError("Method not recognized")
    return out
