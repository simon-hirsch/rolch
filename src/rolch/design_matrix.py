import numpy as np


def add_intercept(X: np.ndarray):
    return np.hstack((np.ones((X.shape[0], 1)), X))
