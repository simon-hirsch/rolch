import numpy as np
import rpy2.robjects as robjects
from ondil import DistributionExponential


def test_exponential_distribution():
    dist = DistributionExponential()

    code = f"""
    library(gamlss.dist)
    set.seed(1)
    y <- rexp(3, rate = 1)  # generate y values > 0
    mu <- rexp(3, rate = 1) + 0.1  # ensure mu > 0
    dist <- gamlss.dist::{dist.corresponding_gamlss}()

    list(
      "y" = y,
      "mu" = mu,
      "cdf" = pEXP(y, mu = mu),
      "pdf" = dEXP(y, mu = mu),
      "ppf" = qEXP(p = pEXP(y, mu = mu), mu = mu)
    )
    """

    # Run the R code and extract values
    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array([
        R_list.rx2("mu"),
    ])

    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(cdf, theta=theta.T)

    assert np.allclose(cdf, R_list.rx2("cdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
    assert np.allclose(pdf, R_list.rx2("pdf")), (
        "Densities don't match, inspect the pdf function"
    )
    assert np.allclose(ppf, R_list.rx2("ppf")), (
        "Quantiles don't match, inspect the ppf function"
    )
