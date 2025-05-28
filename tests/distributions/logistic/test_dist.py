import numpy as np
import rpy2.robjects as robjects
from ondil import DistributionLogistic


def test_logistic_distribution():
    dist = DistributionLogistic()

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
      "cdf" = pLO(y, mu, sigma),
      "pdf" = dLO(y, mu, sigma),
      "ppf" = qLO(y, mu, sigma)
    )
    """

    # Obtain data and derivatives from R
    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array(
        [
            R_list.rx2("mu"),  # mu
            R_list.rx2("sigma"),  # sigma
        ]
    )

    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(y, theta=theta.T)

    assert np.allclose(cdf, R_list.rx2("cdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
    assert np.allclose(pdf, R_list.rx2("pdf")), (
        "Densities don't match, inspect the pdf function"
    )
    assert np.allclose(ppf, R_list.rx2("ppf")), (
        "Quantiles don't match, inspect the ppf function"
    )
