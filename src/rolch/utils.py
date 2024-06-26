import sys


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
