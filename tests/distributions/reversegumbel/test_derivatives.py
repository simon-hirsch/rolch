import numpy as np
import rpy2.robjects as robjects

from ondil import DistributionReverseGumbel


def test_reversegumbel_distribution():

    dist = DistributionReverseGumbel()
    code = f"""
      set.seed(1)
      y <- rep(0.3, 3)
      mu <- rnorm(3)
      sigma <- rexp(3)
      dist <- gamlss.dist::{dist.corresponding_gamlss}()
      list(
        "y" = y,
        "mu" = mu,
        "sigma" = sigma,
        "dldm" = dist$dldm(y, mu, sigma),
        "dldd" = dist$dldd(y, mu, sigma),
        "d2ldm2" = dist$d2ldm2(sigma),
        "d2ldd2" = dist$d2ldd2(sigma),
        "d2ldmdd" = dist$d2ldmdd(sigma)
      )
    """

    # Evaluate R code and get results
    R_list = robjects.r(code)

    # Extract data from R
    y = np.array(R_list.rx2("y"))
    theta = np.array([
        R_list.rx2("mu"),  # mu (location)
        R_list.rx2("sigma"),  # sigma (scale)
    ])

    # Compute Python derivatives
    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta.T, param=0)
    dl1_dp1_1 = dist.dl1_dp1(y, theta=theta.T, param=1)
    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta.T, param=0)
    dl2_dp2_1 = dist.dl2_dp2(y, theta=theta.T, param=1)
    dl2_dpp = dist.dl2_dpp(y, theta=theta.T, params=(0, 1))

    # Assertions
    assert np.allclose(dl1_dp1_0, R_list.rx2("dldm")), (
        "First derivative wrt mu doesn't match"
    )
    assert np.allclose(dl1_dp1_1, R_list.rx2("dldd")), (
        "First derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dp2_0, R_list.rx2("d2ldm2")), (
        "Second derivative wrt mu doesn't match"
    )
    assert np.allclose(dl2_dp2_1, R_list.rx2("d2ldd2")), (
        "Second derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dpp, R_list.rx2("d2ldmdd")), (
        "Cross derivative wrt mu and sigma doesn't match"
    )