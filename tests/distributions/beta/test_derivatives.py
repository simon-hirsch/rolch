import numpy as np
from ondil import DistributionBeta


def test_beta_distribution():
    # Run R code to get coefficients
    # set.seed(1)
    # y <- rep(0.3, 3)
    # mu <- rnorm(3)
    # sigma <- rexp(3)
    # dist <- gamlss.dist::BE()
    # dist$dldm(y, mu, sigma)
    # dist$dldd(y, mu, sigma)
    # dist$d2ldm2(mu, sigma)
    # dist$d2ldd2(mu, sigma)
    # dist$d2ldmdd(mu, sigma)

    y = np.array([0.3, 0.3, 0.3])
    theta = np.array(
        [
            [-0.6264538, 0.1836433, -0.8356286],  # mu
            [0.4360686, 2.8949685, 1.2295621],  # sigma
        ]
    )
    dist = DistributionBeta()

    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta.T, param=0)
    dl1_dp1_0_r = np.array([7.1889027, 7.4931310, -0.5959113])

    dl1_dp1_1 = dist.dl1_dp1(y, theta=theta.T, param=1)
    dl1_dp1_1_r = np.array([46.9985453, 0.5937514, -0.9769637])

    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta.T, param=0)
    dl2_dp2_0_r = np.array([-236.956381, -43.529086, -2.781071])

    dl2_dp2_1 = dist.dl2_dp2(y, theta=theta.T, param=1)
    dl2_dp2_1_r = np.array([-3032.2383259, 0.4149559, -38.7447110])

    dl2_dpp = dist.dl2_dpp(y, theta=theta.T, params=(0, 1))
    dl2_dpp_r = np.array([-856.6320126, 0.3953103, 11.2519330])

    assert np.allclose(dl1_dp1_0, dl1_dp1_0_r), "First derivative wrt mu doesn't match"
    assert np.allclose(dl1_dp1_1, dl1_dp1_1_r), (
        "First derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dp2_0, dl2_dp2_0_r), "Second derivative wrt mu doesn't match"
    assert np.allclose(dl2_dp2_1, dl2_dp2_1_r), (
        "Second derivative wrt sigma doesn't match"
    )
    assert np.allclose(dl2_dpp, dl2_dpp_r), (
        "Second derivative wrt mu and sigma doesn't match"
    )
