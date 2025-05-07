import numpy as np

from rolch import DistributionNormal, OnlineGamlss, DistributionLogNormalMedian

file = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, 1:]


def test_normal_distribution():
    # Run the following R code
    # library("gamlss")
    # data(mtcars)

    # model = gamlss(
    #     mpg ~ cyl + hp,
    #     sigma.formula = ~cyl + hp,
    #     family=NO(),
    #     data=as.data.frame(mtcars)
    # )

    # coef(model, "mu")
    # coef(model, "sigma")

    # To get these coefficients
    coef_R_mu = np.array([36.51776626, -2.32470221, -0.01421071])
    coef_R_sg = np.array([1.8782995906, -0.1262290913, -0.0003943062])

    estimator = OnlineGamlss(
        distribution=DistributionNormal(),
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


def test_LogNormalMedian_distribution():
    # Run R code to get coefficients
    # library("gamlss")
    # data(mtcars)

    # model <- gamlss(
    #   mpg ~ cyl + hp,
    #   sigma.formula = ~ cyl + hp,
    #   family = LOGNO2(),
    #   data = as.data.frame(mtcars)
    # )

    # coef(model, "mu")
    # coef(model, "sigma")

    coef_R_mu = np.array([3.779333731, -0.089852349, -0.001845681])
    coef_R_sg = np.array([-2.01801755, -0.10430409, 0.00492528])

    # Fit Python rolch model (LogNormal2)
    estimator = OnlineGamlss(
        distribution=DistributionLogNormalMedian(),
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
