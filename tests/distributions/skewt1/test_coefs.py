import numpy as np
import rpy2.robjects as robjects

from ondil.distributions import SkewT1
from ondil.estimators import OnlineDistributionalRegression

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]

def test_skewt1_distribution():
    dist = SkewT1()

    code = f"""
    library(gamlss)
    library(gamlss.dist)
    data(mtcars)

    model <- gamlss(
        mpg ~ cyl + hp,
        sigma.formula = ~cyl + hp,
        # nu and tau intercept-only (default)
        family = ST1(),
        data = as.data.frame(mtcars)
    )

    list(
        "mu" = coef(model, "mu"),
        "sigma" = coef(model, "sigma"),
        "nu" = coef(model, "nu"),
        "tau" = coef(model, "tau")
    )
    """

    # Obtain coefficients from R
    R_list = robjects.r(code)
    coef_R_mu = np.asarray(R_list.rx2("mu"))
    coef_R_sg = np.asarray(R_list.rx2("sigma"))
    coef_R_nu = float(R_list.rx2("nu")[0])
    coef_R_tau = float(R_list.rx2("tau")[0])

    estimator = OnlineDistributionalRegression(
        distribution=dist,
        equation={0: np.array([0, 2]), 1: np.array([0, 2])},  # Only mu and sigma depend on predictors
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
    )

    estimator.fit(X=X, y=y)

    assert np.allclose(estimator.beta[0], coef_R_mu, atol=0.03), (
        "Location coefficients don't match"
    )
    assert np.allclose(estimator.beta[1], coef_R_sg, atol=0.03), (
        "Scale coefficients don't match"
    )
    assert np.allclose(estimator.beta[2], coef_R_nu, atol=0.01), (
        "Skew coefficients don't match"
    )
    assert np.allclose(estimator.beta[3], coef_R_tau, atol=7.01), (
        "Kurtosis coefficients don't match"
    )
