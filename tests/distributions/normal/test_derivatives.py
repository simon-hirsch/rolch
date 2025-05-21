# %%
import numpy as np
from rolch import DistributionNormal


def test_normal_distribution():
    # Run R code to get coefficients
    #
    # set.seed(1)
    # y <- rep(0.3, 3)
    # mu <- rnorm(3)
    # sigma <- rexp(3)
    # dist <- gamlss.dist::NO()
    # dist$dldm(y, mu, sigma)
    # dist$dldd(y, mu, sigma)
    # dist$d2ldm2(sigma)
    # dist$d2ldd2(sigma)
    # dist$d2ldmdd(y)

    print("Testing normal distribution derivatives")
    y = np.array([0.3, 0.3, 0.3])
    theta = np.array(
        [
            [-0.6264538, 0.1836433, -0.8356286],  # mu
            [0.4360686, 2.8949685, 1.2295621],  # sigma
        ]
    )
    dist = DistributionNormal()

    dl1_dp1_0 = dist.dl1_dp1(y, theta=theta.T, param=0)
    dl1_dp1_0_r = np.array([4.87207640, 0.01388365, 0.75116514])

    dl1_dp1_1 = dist.dl1_dp1(y, theta=theta.T, param=1)
    dl1_dp1_1_r = np.array([8.0577999, -0.3448689, -0.1195185])

    dl2_dp2_0 = dist.dl2_dp2(y, theta=theta.T, param=0)
    dl2_dp2_0_r = np.array([-5.2588444, -0.1193197, -0.6614532])

    dl2_dp2_1 = dist.dl2_dp2(y, theta=theta.T, param=1)
    dl2_dp2_1_r = np.array([-10.5176887, -0.2386395, -1.3229063])

    dl2_dpp = dist.dl2_dpp(y, theta=theta.T, params=(0, 1))
    dl2_dpp_r = np.array([0.0, 0.0, 0.0])

    assert np.allclose(dl1_dp1_0, dl1_dp1_0_r), "First derivative wrt mu doesn't match"
    assert np.allclose(
        dl1_dp1_1, dl1_dp1_1_r
    ), "First derivative wrt sigma doesn't match"
    assert np.allclose(dl2_dp2_0, dl2_dp2_0_r), "Second derivative wrt mu doesn't match"
    assert np.allclose(
        dl2_dp2_1, dl2_dp2_1_r
    ), "Second derivative wrt sigma doesn't match"
    assert np.allclose(
        dl2_dpp, dl2_dpp_r + 1
    ), "Second derivative wrt mu and sigma doesn't match"
