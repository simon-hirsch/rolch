import sys
import warnings
from typing import Any, Dict, Union

import numpy as np


def handle_param_dict(
    self: Any,
    param: Union[Any, Dict],
    default: Union[Any, Dict],
    n_params: int,
    name: str,
) -> None:
    """This function takes the potentially incomplete parameter dict given by the user
    and the default value for a GAMLSS param and will fill it for all missing distribution
    parameters with the default. If the user passes just a bool, string, int, it will
    create a matching dictionary.

    This is necessary since ROLCH expects at many places as many user given parameters
    as distribution parameters, but we don't know the number of distribution
    parameters a-priori at the estimator initialization.

    This function proceeds to set the default value!

    Args:
        self (Any): The self
        param (Union[Any, Dict]): The param given by the user
        default (Any): The default value
        n_params (int): The number of distribution parameters.
        name (str): The name of the parameter.

    Warns:
        Raises a warning if the user sets an incomplete default dict.

    """
    # Handle (incomplete) dictionaries of parameters.
    if isinstance(param, dict):
        for p in range(n_params):
            if p not in param.keys():
                warnings.warn(
                    f"[{self.__class__.__name__}] "
                    f"No value given for parameter {name} for distribution "
                    f"parameter {p}. Setting default value {default}.",
                    RuntimeWarning,
                    stacklevel=1,
                )
                if isinstance(default, dict):
                    param[p] = default[p]
                else:
                    param[p] = default
    else:
        # No warning since we expect that floats/strings/ints are either the defaults
        # Or given on purpose for all params the ame
        param = {p: param for p in range(n_params)}

    setattr(self, name, param)


def calculate_asymptotic_training_length(forget: float):
    if forget == 0:
        # Return the highest possible value that is no infinity
        return sys.maxsize
    else:
        return 1 / forget


def calculate_effective_training_length(forget: float, n_obs: int):
    if forget == 0:
        return n_obs
    else:
        return (1 - (1 - forget) ** n_obs) / forget


def online_mean_update(avg: float, value: float, forget: float, n_seen: int):

    n_asymmptotic = calculate_asymptotic_training_length(forget)
    n_eff = calculate_effective_training_length(forget, n_seen)

    forget_scaled = forget * np.maximum(n_asymmptotic / n_eff, 1.0)

    diff = value - avg
    incr = forget_scaled * diff

    if forget_scaled > 0:
        new_avg = avg + incr
    else:
        new_avg = avg + diff / n_seen

    return new_avg
