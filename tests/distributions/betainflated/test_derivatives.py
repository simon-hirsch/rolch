# %%
import numpy as np
import rpy2.robjects as robjects
from ondil import DistributionBetaInflated


def test_betainflated_distribution():
    code = """
    set.seed(1)
    y <- rep(0.3, 3)
    mu <- runif(3)
    sigma <- runif(3)
    nu <- runif(3)
    tau <- runif(3)
    dist <- gamlss.dist::BEINF()
    list(
    "y" = y,
    "mu" = mu,
    "sigma" = sigma,
    "nu" = nu,
    "tau" = tau,
    "dldm" = dist$dldm(y, mu, sigma),
    "dldd" = dist$dldd(y, mu, sigma),
    "dldv" = dist$dldv(y, nu, tau),
    "dldt" = dist$dldt(y, nu, tau),
    "d2ldm2" = dist$d2ldm2(y, mu, sigma),
    "d2ldd2" = dist$d2ldd2(y, mu, sigma, nu, tau),
    "d2ldv2" = dist$d2ldv2(nu, tau), 
    "d2ldt2" = dist$d2ldt2(nu, tau),
    "d2ldmdd" = dist$d2ldmdd(y, mu, sigma),
    "d2ldmdv" = dist$d2ldmdv(y),
    "d2ldmdt" = dist$d2ldmdt(y), 
    "d2ldddv" = dist$d2ldddv(y),
    "d2ldddt" = dist$d2ldddt(y),
    "d2ldvdt" = dist$d2ldvdt(nu, tau)
    )
    """

    # Obtain data and derivatives from R
    R_list = robjects.r(code)

    # We take the data from R
    y = np.array(R_list.rx2("y"))
    theta = np.array(
        [
            R_list.rx2("mu"),  # mu
            R_list.rx2("sigma"),  # sigma
            R_list.rx2("nu"),  #nu 
            R_list.rx2("tau"),  #tau
        ]
    )

    # Obtain the derivatives from Python
    dist = DistributionBetaInflated()
    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta.T, param=0)
    dl1_dp1_1 = dist.dl1_dp1(y, theta=theta.T, param=1)
    dl1_dp1_2 = dist.dl1_dp1(y, theta=theta.T, param=2)
    dl1_dp1_3 = dist.dl1_dp1(y, theta=theta.T, param=3)
    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta.T, param=0)
    dl2_dp2_1 = dist.dl2_dp2(y, theta=theta.T, param=1)
    dl2_dp2_2 = dist.dl2_dp2(y, theta=theta.T, param=2)
    dl2_dp2_3 = dist.dl2_dp2(y, theta=theta.T, param=3)
    dl2_dpp_01 = dist.dl2_dpp(y, theta=theta.T, params=(0, 1))
    dl2_dpp_02 = dist.dl2_dpp(y, theta=theta.T, params=(0, 2))
    dl2_dpp_03 = dist.dl2_dpp(y, theta=theta.T, params=(0, 3))
    dl2_dpp_12 = dist.dl2_dpp(y, theta=theta.T, params=(1, 2))
    dl2_dpp_13 = dist.dl2_dpp(y, theta=theta.T, params=(1, 3))
    dl2_dpp_23 = dist.dl2_dpp(y, theta=theta.T, params=(2, 3))

    # Compare with R results
    assert np.allclose(dl1_dp1_0, R_list.rx2("dldm")), (
        "First derivative wrt mu doesn't match"
    )
    assert np.allclose(dl1_dp1_1, R_list.rx2("dldd")), (
        "First derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl1_dp1_2, R_list.rx2("dldv")), (
        "First derivative wrt nu doesn't match"
    )
    assert np.allclose(dl1_dp1_3, R_list.rx2("dldt")), (
        "First derivative wrt tau doesn't match"
    )

    assert np.allclose(dl2_dp2_0, R_list.rx2("d2ldm2")), (
        "Second derivative wrt mu doesn't match"
    )
    assert np.allclose(dl2_dp2_1, R_list.rx2("d2ldd2")), (
        "Second derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dp2_2, R_list.rx2("d2ldv2")), (
        "Second derivative wrt nu doesn't match"
    )
    assert np.allclose(dl2_dp2_3, R_list.rx2("d2ldt2")), (
        "Second derivative wrt tau doesn't match"
    )

    assert np.allclose(dl2_dpp_01, R_list.rx2("d2ldmdd")), (
        "Second derivative wrt mu and sigma doesn't match"
    )
    assert np.allclose(dl2_dpp_02, R_list.rx2("d2ldmdv")), (
        "Second derivative wrt mu and nu doesn't match"
    )
    assert np.allclose(dl2_dpp_03, R_list.rx2("d2ldmdt")), (
        "Second derivative wrt mu and tau doesn't match"
    )
    assert np.allclose(dl2_dpp_12, R_list.rx2("d2ldddv")), (
        "Second derivative wrt sigma and nu doesn't match"
    )
    assert np.allclose(dl2_dpp_13, R_list.rx2("d2ldddt")), (
        "Second derivative wrt sigma and tau doesn't match"
    )
    print(dl2_dpp_23 - R_list.rx2("d2ldvdt"))
    assert np.allclose(dl2_dpp_23, R_list.rx2("d2ldvdt")), (
        "Second derivative wrt nu and tau doesn't match"
    )
