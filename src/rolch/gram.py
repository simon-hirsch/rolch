import numpy as np
import numba as nb


@nb.njit()
def init_forget_vector(forget: float, size: int) -> np.ndarray:
    gamma = 1 - forget
    if gamma == 1:
        out = np.ones(size)
    else:
        out = np.empty(size)
        out[-1] = 1
        for i in range(size - 1, 0, -1):
            out[i - 1] = out[i] * gamma
    return out


@nb.njit()
def init_gram(X: np.ndarray, w: np.ndarray, forget: float = 0) -> np.ndarray:
    """Initialise the Gramian Matrix."""
    f = init_forget_vector(forget, X.shape[0])
    gram = (X * np.expand_dims((w * f) ** 0.5, -1)).T @ (
        X * np.expand_dims((w * f) ** 0.5, -1)
    )
    return gram


@nb.njit()
def init_y_gram(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, forget: float = 0
) -> np.ndarray:
    """Initialise the 'Y-Gram'-Matrix X.T @ Y."""
    f = init_forget_vector(forget, X.shape[0])
    gram = np.expand_dims(
        (X * np.expand_dims((w * f) ** 0.5, -1)).T @ (y * (w * f) ** 0.5), axis=-1
    )
    return gram


@nb.njit()
def init_inverted_gram(X: np.ndarray, w: np.ndarray, forget: float = 0) -> np.ndarray:
    """Initialise the inverted Gramian"""
    inv_gram = np.linalg.inv(init_gram(X=X, w=w, forget=forget))
    return inv_gram


# TODO (SH): For some reason we switched the syntax in C++ here
# We should change that


@nb.njit()
def update_gram(
    gram: np.ndarray, X: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    new_gram = (1 - forget) * gram + w * np.outer(X, X)
    return new_gram


@nb.njit()
def update_y_gram(
    gram: np.ndarray, X: np.ndarray, y: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    new_gram = (1 - forget) * gram + w * np.outer(X, y)
    return new_gram


@nb.njit()
def update_inverted_gram(
    gram: np.ndarray, X: np.ndarray, forget: float = 0, w: float = 1
) -> np.ndarray:
    gamma = 1 - forget
    new_gram = (1 / gamma) * (
        gram - ((w * gram @ np.outer(X, X) @ gram) / (gamma + w * X @ gram @ X.T))
    )
    return new_gram
