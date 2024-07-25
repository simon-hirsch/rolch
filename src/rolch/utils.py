import sys
import numpy as np


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
