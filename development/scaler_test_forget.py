def calculate_weighted_statistics(X: np.ndarray, forget: float) -> tuple:
    """Calculate weighted mean and variance with exponential forgetting.

    Args:
        X: Data array
        forget: Forget factor

    Returns:
        Tuple of (weighted_mean, weighted_variance)
    """
    if forget == 0:
        return np.mean(X, axis=0), np.var(X, axis=0)

    n_obs = len(X)
    # Create exponential weights (most recent observation has weight 1)
    indices = np.arange(n_obs)
    weights = (1 - forget) ** (n_obs - 1 - indices)

    # Normalize weights
    weights = weights / np.sum(weights)

    # Calculate weighted mean
    weighted_mean = np.average(X, weights=weights, axis=0)

    # Calculate weighted variance
    weighted_variance = np.average((X - weighted_mean) ** 2, weights=weights, axis=0)

    return weighted_mean, weighted_variance


# %%
np.random.seed(42)

N = 110  # Number of samples
N_init = 1
assert N > N_init, "N must be greater than N_init for this test."

X = np.random.uniform(
    0,
    10,
    N,
).reshape(-1, 1)

# %%
# Test with different forget values
forget_values = [0.0, 0.01]

for forget in forget_values:
    print(f"\n=== Testing with forget = {forget} ===")

    os = OnlineScaler(forget=forget)
    os.fit(X[0:N_init,])

    # Initial comparison
    true_mean, true_var = calculate_weighted_statistics(X[0:N_init,], forget)
    print(f"Initial - OnlineScaler mean: {os.m[0]:.8f}, True mean: {true_mean[0]:.8f}")
    print(f"Initial - OnlineScaler var: {os.v[0]:.8f}, True var: {true_var[0]:.8f}")

    # Test the update method
    for i in range(N_init, N):
        os.partial_fit(X[i,].reshape(-1, 1))

        # Calculate true weighted statistics for all data seen so far
        true_mean, true_var = calculate_weighted_statistics(X[0 : (i + 1),], forget)

        mean_diff = round(abs(true_mean[0] - os.m[0]), 8)
        var_diff = round(abs(true_var[0] - os.v[0]), 8)

        print(f"{i - N_init + 1:03d}: Mean diff = {mean_diff}, Var diff = {var_diff}")
