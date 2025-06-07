import numpy as np
import pytest
import rpy2.robjects as robjects
import ondil.distributions
import inspect
from ondil import OnlineGamlss


def get_distributions_with_gamlss():
    """Get all distribution classes that have a corresponding_gamlss attribute."""
    distributions = []
    for name, obj in inspect.getmembers(ondil.distributions):
        if hasattr(obj, "corresponding_gamlss"):
            instance = obj()
            if instance.corresponding_gamlss is not None:
                distributions.append(instance)
    return distributions


@pytest.mark.parametrize(
    "distribution",
    get_distributions_with_gamlss(),
    ids=lambda dist: dist.__class__.__name__,
)
def test_distribution_coefficients(distribution):
    """Test that OnlineGamlss coefficients match R gamlss for different distributions."""
    N = 1000
    CLIP_BOUNDS = (-1e5, 1e5)

    theta = np.array(
        [
            [
                np.random.uniform(
                    np.clip(distribution.parameter_support[i][0], *CLIP_BOUNDS),
                    np.clip(distribution.parameter_support[i][1], *CLIP_BOUNDS),
                )
                for i in range(distribution.n_params)
            ]
        ]
    )

    y = distribution.rvs(size=N, theta=theta)
    robjects.globalenv["y"] = robjects.FloatVector(y[0, :])

    code = f"""
      # Create data frame
      df <- data.frame(y = y)
      model <- gamlss::gamlss(
        y ~ 1,
        family = gamlss.dist::{distribution.corresponding_gamlss}(),
        data = df
      )
      list(
      "mu" = model$mu.coefficients,
      "sigma" = model$sigma.coefficients,
      "nu" = model$nu.coefficients,
      "tau" = model$tau.coefficients
      )
      """

    # Obtain coefficients from R
    R_list = robjects.r(code)

    estimator = OnlineGamlss(
        distribution=distribution,
        equation={0: "intercept"},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
        rss_tol_inner=10,
    )

    estimator.fit(X=y.T, y=y[0, :])

    # Compare all parameter estimates
    for beta_idx, r_param_name in distribution.parameter_names.items():
        assert np.allclose(estimator.beta[beta_idx], R_list.rx2(r_param_name)), (
            f"{r_param_name} coefficients don't match"
        )
