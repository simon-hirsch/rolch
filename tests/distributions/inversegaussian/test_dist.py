import numpy as np
import rpy2.robjects as robjects

from ondil import DistributionInverseGaussian


def test_inversegaussian_distribution():
    dist = DistributionInverseGaussian()

    code = """
    library(gamlss.dist)
    set.seed(1)
    y <- rIG(3, mu = 2, sigma = 1)  # generate y values > 0
    mu <- rexp(3, rate = 1) + 0.1   # ensure mu > 0
    sigma <- rexp(3, rate = 1) + 0.1  # ensure sigma > 0

    list(
      "y" = y,
      "mu" = mu,
      "sigma" = sigma,
      "cdf" = pIG(y, mu = mu, sigma = sigma),
      "pdf" = dIG(y, mu = mu, sigma = sigma),
      "ppf" = qIG(p = pIG(y, mu = mu, sigma = sigma), mu = mu, sigma = sigma)
    )
    """

    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    theta = np.array(
        [
            R_list.rx2("mu"),
            R_list.rx2("sigma"),
        ]
    )

    cdf = dist.cdf(y, theta=theta.T)
    pdf = dist.pdf(y, theta=theta.T)
    ppf = dist.ppf(cdf, theta=theta.T)

    assert np.allclose(cdf, R_list.rx2("cdf")), (
        "Probabilities don't match, inspect the cdf function"
    )
    assert np.allclose(pdf, R_list.rx2("pdf")), (
        "Densities don't match, inspect the pdf function"
    )
    # Note the discussion in the PR here: There is a numerical precision missmatch in 
    # the implementations provided by scipy and R::GAMLSS
    # https://github.com/simon-hirsch/ondil/pull/118#issuecomment-2942992914
    assert np.allclose(ppf, R_list.rx2("ppf"), atol=1e-3, rtol=1e-3), (
        "Quantiles don't match, inspect the ppf function"
    )
