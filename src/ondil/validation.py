import numpy as np


def ensure_atleast_bivariate(y: np.ndarray) -> None:
    """Validates that the response variable is a multivariate array with at least two dimensions.

    Args:
        y (np.ndarray): The response variable to validate.
    Raises:
        ValueError: If `y` does not have at least two dimensions and has less than two columns.
    """
    if y.shape[1] < 2:
        raise ValueError(
            "Multivariate response must have at least two dimensions. "
            f"Got {y.shape[1]} dimensions."
        )
