import numpy as np
from rolch import OnlineGamlss, DistributionLogistic

file = "tests/data/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_logistic_distribution():
    # Run R code to get coefficients
    # library("gamlss")
    # data(mtcars)

    # model = gamlss(
    #     mpg ~ cyl + hp,
    #     sigma.formula = ~cyl + hp,
    #     family=LO(),
    #     data=as.data.frame(mtcars)
    # )

    # coef(model, "mu")
    # coef(model, "sigma")

    # To get these coefficients
    coef_R_mu = np.array([35.48583210, -2.26935567, -0.01036068])
    coef_R_sg = np.array([1.434803735, -0.120435484, -0.001492783])

    estimator = OnlineGamlss(
        distribution=DistributionLogistic(),
        equation={0: np.array([0, 2]), 1: np.array([0, 2])},
        method="ols",
        scale_inputs=False,
        fit_intercept=True,
        rss_tol_inner=10,
    )

    estimator.fit(X=X, y=y)

    assert np.allclose(
        estimator.beta[0], coef_R_mu, atol=0.01
    ), "Location coefficients don't match"
    assert np.allclose(
        estimator.beta[1], coef_R_sg, atol=0.01
    ), "Scale coefficients don't match"
