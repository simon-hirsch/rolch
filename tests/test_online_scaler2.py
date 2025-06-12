import numpy as np
import pytest
from ondil.scaler import OnlineScaler
from ondil.gram import init_forget_vector


@pytest.mark.parametrize(
    "N", [10, 100, 1000], ids=["samples_10", "samples_100", "samples_1000"]
)
@pytest.mark.parametrize("N_init", [1, 10, 100], ids=["init_1", "init_10", "init_100"])
@pytest.mark.parametrize("D", [1, 10], ids=["features_1", "features_10"])
@pytest.mark.parametrize(
    "forget",
    [0, 0.001, 0.01, 0.1],
    ids=["no_forgetting", "forget_0.001", "forget_0.01", "forget_0.1"],
)
def test_online_scaler(N, N_init, D, forget):
    """Test OnlineScaler with various combinations of parameters."""
    # Skip invalid combinations
    if N_init >= N:
        pytest.skip("N_init must be less than N")

    np.random.seed(42)  # For reproducible tests

    X = np.random.uniform(0, 10, N * D).reshape(-1, D)
    W_sample = np.random.uniform(0.1, 1.0, (N, 1)).reshape(-1, 1)

    # Initialize and fit the scaler
    os = OnlineScaler(forget=forget)
    os.fit(X[0:N_init,], sample_weight=W_sample[0:N_init,].flatten())

    # Calculate effective weights for comparison
    W_forget = init_forget_vector(forget, N).reshape(-1, 1)
    W_effective = W_forget * W_sample

    # Calculate true mean and variance for all columns after initial fit
    true_mean_init = np.array(
        [
            np.average(X[0:N_init, d], weights=W_effective[0:N_init].flatten())
            for d in range(D)
        ]
    )
    true_var_init = np.array(
        [
            np.average(
                (X[0:N_init, d] - true_mean_init[d]) ** 2,
                weights=W_effective[0:N_init].flatten(),
            )
            for d in range(D)
        ]
    )

    # Assert initial fit is correct
    assert np.allclose(os.m, true_mean_init, rtol=1e-10), (
        f"Initial mean mismatch: {os.m} vs {true_mean_init}"
    )
    assert np.allclose(os.v, true_var_init, rtol=1e-10), (
        f"Initial variance mismatch: {os.v} vs {true_var_init}"
    )

    # Test partial fit updates
    for i in range(N_init, N):
        os.partial_fit(X[i : i + 1,], sample_weight=W_sample[i,])

        # Calculate true mean and variance for all columns
        true_mean = np.array(
            [
                np.average(
                    X[0 : (i + 1), d], weights=W_effective[0 : (i + 1)].flatten()
                )
                for d in range(D)
            ]
        )
        true_var = np.array(
            [
                np.average(
                    (X[0 : (i + 1), d] - true_mean[d]) ** 2,
                    weights=W_effective[0 : (i + 1)].flatten(),
                )
                for d in range(D)
            ]
        )

        assert np.allclose(os.m, true_mean, rtol=1e-10), (
            f"Step {i - N_init + 1}: Mean mismatch: {os.m} vs {true_mean}"
        )
        assert np.allclose(os.v, true_var, rtol=1e-10), (
            f"Step {i - N_init + 1}: Variance mismatch: {os.v} vs {true_var}"
        )
