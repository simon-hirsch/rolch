import numpy as np
import rpy2.robjects as robjects
from ondil import DistributionExponential

def test_exponential_distribution():
    code = """
    set.seed(1)
    y <- rep(0.3, 3)
    mu <- rnorm(3)
    dist <- gamlss.dist::EXP()
    list(
      "y" = y,
      "mu" = mu,
      "dldm" = dist$dldm(y, mu),
      "d2ldm2" = dist$d2ldm2(mu)
    )
    """

    R_list = robjects.r(code)

    y = np.array(R_list.rx2("y"))
    mu = np.array(R_list.rx2("mu"))
    theta = np.array([mu])

    dist = DistributionExponential()

    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta.T, param=0)
    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta.T, param=0)

    assert np.allclose(dl1_dp1_0, R_list.rx2("dldm")), (
        "First derivative wrt mu doesn't match"
    )
    assert np.allclose(dl2_dp2_0, R_list.rx2("d2ldm2")), (
        "Second derivative wrt mu doesn't match"
    )
