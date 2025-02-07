import numpy as np
import pytest
from sklearn.datasets import make_regression

from rolch.distributions import DistributionJSU
from rolch.estimators import OnlineGamlss

FIT_INTERCEPT = [True, False]
N_FEATURES = np.round(np.geomspace(11, 100, 5)).astype(int)


@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("fit_intercept", FIT_INTERCEPT)
def test_get_J_from_equation(n_features, fit_intercept):

    equation = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, 4),
        3: np.array([True] * 10 + [False] * (n_features - 10)).astype(bool),
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

    J = estimator.get_J_from_equation(X)
    assert J[0] == EXPECTED[0][fit_intercept], "Wrong J for param == 0"
    assert J[1] == EXPECTED[1][fit_intercept], "Wrong J for param == 1"
    assert J[2] == EXPECTED[2][fit_intercept], "Wrong J for param == 2"
    assert J[3] == EXPECTED[3][fit_intercept], "Wrong J for param == 3"


def test_get_J_from_equation_warnings():

    n_features = 10
    fit_intercept = True

    equation_fail_2 = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, 20),
        3: np.array([True] * n_features).astype(bool),
    }

    X, _ = make_regression(n_samples=100, n_features=n_features)
    distribution = DistributionJSU()

    estimator = OnlineGamlss(
        distribution=distribution,
        equation=equation_fail_2,
        fit_intercept=fit_intercept,
    )
    with pytest.raises(ValueError, match="Shape does not match for param 2."):
        J = estimator.get_J_from_equation(X)

    # Test for parameter three
    equation_fail_3 = {
        0: "all",  # should adjust to n_features
        1: "intercept",
        2: np.arange(0, n_features),
        3: np.array([True, False] * 10).astype(bool),
    }

    X, _ = make_regression(n_samples=100, n_features=10)
    distribution = DistributionJSU()
    estimator = OnlineGamlss(
        distribution=distribution,
        equation=equation_fail_3,
        fit_intercept=fit_intercept,
    )

    with pytest.raises(
        ValueError,
        match="Shape does not match for param 3.",
    ):
        J = estimator.get_J_from_equation(X)
