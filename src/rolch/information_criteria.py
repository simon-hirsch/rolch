from typing import Literal, Union

import numpy as np


def information_criterion(
    n_observations: Union[int, np.ndarray],
    n_parameters: Union[int, np.ndarray],
    rss: Union[float, np.ndarray],
    criterion: Literal["aic", "bic", "hqc"] = "aic",
) -> Union[float, np.ndarray]:
    """Calcuate the information criteria.

    The information criteria are calculated from the Resdiual Sum of Squares $RSS$. The function provides the
    calculation of Akaikes IC, the bayesian IC and the Harman-Quinn IC.

    Args:
        n_observations (Union[int, np.ndarray]): Number of observations
        n_parameters (Union[int, np.ndarray]): Number of parameters
        rss (Union[float, np.ndarray]): Residual sum of squares
        criterion (Literal["aic", "bic", "hqc"], optional): Information criteria to calculate. Defaults to "aic".

    Raises:
        ValueError: Raises if the criterion is not one of `aic`, `bic` or `hqc`.

    Returns:
        Union[float, np.ndarray]: The value of the IC for given inputs.
    """
    constant_term = -n_observations / 2 * (1 + np.log(2 * np.pi))

    if criterion == "aic":
        ic = (
            2 * n_parameters
            + n_observations * np.log(rss / n_observations)
            + constant_term
        )
    elif criterion == "bic":
        ic = (
            n_parameters * np.log(n_observations)
            + n_observations * np.log(rss / n_observations)
            + constant_term
        )
    elif criterion == "hqc":
        ic = (
            2 * n_parameters * np.log(np.log(n_observations))
            + n_observations * np.log(rss / n_observations)
            + constant_term
        )
    else:
        raise ValueError("criterion not recognized.")
    return ic


def select_best_model_by_information_criterion(
    n_observations: float,
    n_parameters: np.ndarray,
    rss: np.ndarray,
    criterion: Literal["aic", "bic", "hqc", "max"],
) -> int:
    """Calculates the information criterion and returns the model with the best (lowest) IC.

    !!! note
        The information criterion `max` will always return the largest model.

    Args:
        n_observations (float): Number of observations
        n_parameters (np.ndarray): Number of parameters per model
        rss (np.ndarray): Residual sum of squares per model
        criterion (Literal["aic", "bic", "hqc", "max"]): Information criterion.

    Raises:
        ValueError: Raises if the criterion is not one of `aic`, `bic`, `hqc` or `max`.

    Returns:
        int: Index of the model with the best (lowest) IC.
    """
    if criterion == "max":
        best = n_parameters.shape[0] - 1
    else:
        ic = information_criterion(
            n_observations=n_observations,
            n_parameters=n_parameters,
            rss=rss,
            criterion=criterion,
        )
        best = np.argmin(ic)
    return best
