import numpy as np
import rpy2.robjects as robjects

from ondil import DistributionInverseGamma


def test_inverse_gamma_distribution():
    dist = DistributionInverseGamma()

    code = f"""
    library(gamlss.dist)
    set.seed(1)
    y <- rep(0.3, 3)
    mu <- runif(3, 0.1, 5)
    sigma <- rexp(3)
    dist <- gamlss.dist::{dist.corresponding_gamlss}()

    list(
      "y" = y,
      "mu" = mu,
      "sigma" = sigma,
      "cdf" = pIGAMMA(y, mu, sigma),
      "pdf" = dIGAMMA(y, mu, sigma),
      "ppf" = qIGAMMA(y, mu, sigma)
    )
    """

    # Obtain data and values from R
    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array([
        R_list.rx2("mu"),     # mu
        R_list.rx2("sigma"),  # sigma
    ])

    # Compute values using Python
    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(y, theta=theta.T)

    # Assertions for test validation
    assert np.allclose(cdf, R_list.rx2("cdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
    assert np.allclose(pdf, R_list.rx2("pdf")), (
        "Densities don't match, inspect the pdf function"
    )
    assert np.allclose(ppf, R_list.rx2("ppf")), (
        "Quantiles don't match, inspect the ppf function"
    )