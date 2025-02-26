import pytest
import numpy as np
from rolch.distributions.johnsonsu import DistributionJSU

PARAM = np.arange(DistributionJSU().n_params)


@pytest.mark.parametrize("param", PARAM)
def test_dl2_dpp_raises_value_error(param):
    dist = DistributionJSU()
    y = np.array([1, 2, 3])
    theta = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
    with pytest.raises(
        ValueError, match="Cross derivatives must use different parameters."
    ):
        dist.dl2_dpp(y, theta, (param, param))
