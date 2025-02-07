import numpy as np
import pytest
from sklearn.datasets import make_regression

from rolch import OnlineScaler

N_FEATURES = [10, 100, 1000]
N_SAMPLES = [100, 1000, 10000]
HORIZON = [1, 10, 50, 90]
DTYPES = [bool, int]

# We don't jet test online updates with forget


@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
def test_standard_scaling(n_features, n_samples):
    X, _ = make_regression(n_features=n_features, n_samples=n_samples)
    expected_out = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    scaler = OnlineScaler(forget=0, to_scale=True)
    scaler.fit(X=X)
    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)


@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
@pytest.mark.parametrize("selection_dtype", DTYPES)
def test_standard_scaling_feature_selection(n_features, n_samples, selection_dtype):

    if selection_dtype == bool:
        to_scale = np.random.choice([True, False], n_features)
    if selection_dtype == int:
        to_scale = np.random.choice(np.arange(n_features), n_features // 2)

    X, _ = make_regression(n_features=n_features, n_samples=n_samples)
    scaled_out = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    expected_out = np.copy(X)
    expected_out[:, to_scale] = scaled_out[:, to_scale]

    scaler = OnlineScaler(forget=0, to_scale=to_scale)
    scaler.fit(X=X)
    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)


@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
def test_standard_scaling_dont_scale(n_features, n_samples):
    X, _ = make_regression(n_features=n_features, n_samples=n_samples)
    expected_out = X

    scaler = OnlineScaler(forget=0, to_scale=False)
    scaler.fit(X=X)
    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)


@pytest.mark.parametrize("horizon", HORIZON)
@pytest.mark.parametrize("n_features", N_FEATURES)
@pytest.mark.parametrize("n_samples", N_SAMPLES)
def test_online_update_batch(n_features, n_samples, horizon):
    X, _ = make_regression(n_features=n_features, n_samples=n_samples)
    expected_out = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    index_initial = np.arange(0, n_samples - horizon)
    index_online = np.arange(n_samples - horizon, n_samples)
    scaler = OnlineScaler(forget=0, to_scale=True)
    scaler.fit(X=X[index_initial])

    for i in index_online:
        scaler.partial_fit(X[[i], :])

    out = scaler.transform(X=X)
    np.testing.assert_array_almost_equal(expected_out, out)
