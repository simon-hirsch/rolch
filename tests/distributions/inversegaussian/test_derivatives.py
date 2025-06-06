import numpy as np
import rpy2.robjects as robjects

from ondil import DistributionInverseGaussian


def test_inversegaussian_derivatives():
    code = """
    library(gamlss.dist)
    set.seed(1)
    y <- rep(1.5, 3)
    mu <- rexp(3, rate = 1) + 0.1
    sigma <- rexp(3, rate = 1) + 0.1
    dist <- gamlss.dist::IG()
    list(
      "y" = y,
      "mu" = mu,
      "sigma" = sigma,
      "dldm" = dist$dldm(y, mu, sigma),
      "d2ldm2" = dist$d2ldm2(mu, sigma),
      "dldd" = dist$dldd(y, mu, sigma),
      "d2ldd2" = dist$d2ldd2(sigma),
      "d2ldmdd" = dist$d2ldmdd(y)
    )
    """

    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    mu = np.array(R_list.rx2("mu"))
    sigma = np.array(R_list.rx2("sigma"))
    theta = np.array([mu, sigma])

    dist = DistributionInverseGaussian()

    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta.T, param=0)
    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta.T, param=0)
    dl1_dp1_1 = dist.dl1_dp1(y, theta=theta.T, param=1)
    dl2_dp2_1 = dist.dl2_dp2(y, theta=theta.T, param=1)
    dl2_dpp_01 = dist.dl2_dpp(y, theta=theta.T, params=(0, 1))

    assert np.allclose(
        dl1_dp1_0, R_list.rx2("dldm")
    ), "First derivative wrt mu doesn't match"
    assert np.allclose(
        dl2_dp2_0, R_list.rx2("d2ldm2")
    ), "Second derivative wrt mu doesn't match"
    assert np.allclose(
        dl1_dp1_1, R_list.rx2("dldd")
    ), "First derivative wrt sigma doesn't match"
    assert np.allclose(
        dl2_dp2_1, R_list.rx2("d2ldd2")
    ), "Second derivative wrt sigma doesn't match"
    assert np.allclose(
        dl2_dpp_01, R_list.rx2("d2ldmdd")
    ), "Cross derivative doesn't match"
