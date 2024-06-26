from typing import Literal

import numpy as np


def information_criterion(
    n_observations, n_parameters, rss, criterion: Literal["aic", "bic", "hqc"] = "aic"
):
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
):
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
