import numpy as np
import pytest
from sklearn.datasets import make_regression

from rolch.distributions import DistributionJSU
from rolch.estimators import OnlineGamlss

FIT_INTERCEPT = [True, False]
N_FEATURES = np.round(np.geomspace(1, 100, 20)).astype(int)


@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("fit_intercept", FIT_INTERCEPT)
def test_get_J_from_equation(n_features, fit_intercept):

    # For param 3:
    # We need at least the 10 first ones to check whether we fail on wrong dims
    # For everything else, we want to have the correct dims for the boolean array.

    if n_features <= 10:
        bool_array = np.repeat(True, 10)
    else:
        bool_array = np.concatenate(
            (np.repeat(True, 10), np.repeat(False, n_features - 10))
        )

    equation = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, 4),
        3: bool_array,
    }

    EXPECTED = {
        0: {True: n_features + 1, False: n_features},
        1: {True: 1, False: 1},
        2: {True: 5, False: 4},
        3: {True: 11, False: 10},
    }

    X, _ = make_regression(n_samples=100, n_features=n_features)
    distribution = DistributionJSU()

    estimator = OnlineGamlss(
        distribution=distribution,
        equation=equation,
        fit_intercept=fit_intercept,
    )

    if n_features < 4:
        with pytest.raises(ValueError):
            # We want to complain about parameter 2 because the max value in the int is
            print(n_features)
            J = estimator.get_J_from_equation(X)
    if (n_features >= 4) and (n_features <= 10):
        # Then we will fail on the boolean.
        # We check the integer array before, so this will still fail before
        with pytest.raises(ValueError):
            J = estimator.get_J_from_equation(X)
    if n_features > 10:
        J = estimator.get_J_from_equation(X)
        assert J[0] == EXPECTED[0][fit_intercept], "Wrong J for param == 0"
        assert J[1] == EXPECTED[1][fit_intercept], "Wrong J for param == 1"
        assert J[2] == EXPECTED[2][fit_intercept], "Wrong J for param == 2"
        assert J[3] == EXPECTED[3][fit_intercept], "Wrong J for param == 3"
