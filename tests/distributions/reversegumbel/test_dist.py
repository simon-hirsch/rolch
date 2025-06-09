import numpy as np
import rpy2.robjects as robjects

from ondil import DistributionReverseGumbel


def test_reversegumbel_distribution():
    dist = DistributionReverseGumbel()

    code = f"""
    library(gamlss.dist)
    set.seed(1)
    y <- rep(0.3, 3)
    mu <- rnorm(3)
    sigma <- rexp(3)
    dist <- gamlss.dist::{dist.corresponding_gamlss}()

    list(
      "y" = y,
      "mu" = mu,
      "sigma" = sigma,
      "cdf" = pRG(y, mu, sigma),
      "pdf" = dRG(y, mu, sigma),
      "ppf" = qRG(pRG(y, mu, sigma), mu, sigma)
    )
    """

    # Obtain data and values from R
    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array([
        R_list.rx2("mu"),
        R_list.rx2("sigma"),
    ])

    # Compute values using Python
    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(cdf, theta=theta.T)

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
