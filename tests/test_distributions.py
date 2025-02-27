from itertools import product

import numpy as np
import pytest

import rolch

DISTRIBUTIONS = [
    getattr(rolch.distributions, name)() for name in rolch.distributions.__all__
]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_raise_error_cross_derivative(distribution):
    n_params = distribution.n_params
    # We just want some values here on a reasonable space to ensure that
    # We can test the raise of the derivative
    lower = np.clip(distribution.distribution_support[0], -1e3, 1e3)
    upper = np.clip(distribution.distribution_support[1], -1e3, 1e3)
    y = np.random.uniform(low=lower, high=upper, size=1000)
    theta = np.hstack(
        [distribution.initial_values(y, param=p)[:, None] for p in range(n_params)]
    )
    for a, b in product(range(n_params), range(n_params)):
        if a == b:
            with pytest.raises(
                ValueError, match="Cross derivatives must use different parameters."
            ):
                distribution.dl2_dpp(y, theta, (a, b))
        else:
            deriv = distribution.dl2_dpp(y, theta, (a, b))
            assert y.shape == deriv.shape, "Derivative shape should match y.shape"
