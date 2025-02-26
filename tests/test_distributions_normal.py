import pytest
import numpy as np
from rolch.distributions.normal import DistributionNormal

PARAM = np.arange(DistributionNormal().n_params)


@pytest.mark.parametrize("param", PARAM)
def test_dl2_dpp_raises_value_error(param):
    dist = DistributionNormal()
    y = np.array([1, 2, 3])
    theta = np.array([[0, 1], [1, 2], [2, 3]])
    with pytest.raises(
        ValueError, match="Cross derivatives must use different parameters."
    ):
        dist.dl2_dpp(y, theta, (param, param))
