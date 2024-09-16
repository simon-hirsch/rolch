import numpy as np

import rolch

file = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
mtcars = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:]

y = mtcars[:, 0]
X = mtcars[:, [1, 3]]
X = np.hstack((np.ones((y.shape[0], 1)), X))


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

    estimator = rolch.OnlineGamlss(
        distribution=rolch.DistributionNormal(),
        method="ols",
        do_scale=False,
        expect_intercept=True,
        rss_tol_inner=10,
    )
    estimator.fit(y=y, x0=X, x1=X)

    assert np.allclose(
        estimator.betas[0], coef_R_mu, atol=0.01
    ), "Location coefficients don't match"
    assert np.allclose(
        estimator.betas[1], coef_R_sg, atol=0.01
    ), "Scale coefficients don't match"
