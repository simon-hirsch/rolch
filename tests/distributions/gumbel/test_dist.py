import numpy as np
import rpy2.robjects as robjects

from ondil import DistributionGumbel


def test_gumbel_distribution():
    dist = DistributionGumbel()
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
      "cdf" = pGU(y, mu, sigma),
      "pdf" = dGU(y, mu, sigma),
      "ppf" = qGU(pGU(y, mu, sigma), mu, sigma)
    )
    """
    R_list = robjects.r(code)
    y = np.array(R_list.rx2("y"))
    theta = np.array([R_list.rx2("mu"), R_list.rx2("sigma")])

    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(cdf, theta=theta.T)

    assert np.allclose(cdf, R_list.rx2("cdf"))
    assert np.allclose(pdf, R_list.rx2("pdf"))
    assert np.allclose(ppf, R_list.rx2("ppf"))
