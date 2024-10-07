# Author: Simon Hirsch
# Licence: MIT

import numpy as np
import pytest

import rolch

REAL_LINE_LINKS = [rolch.IdentityLink]
POSITIVE_LINE_LINKS = [rolch.LogLink, rolch.SqrtLink, rolch.LogIdentLink]
SHIFTED_LINKS = [rolch.LogShiftValueLink, rolch.SqrtShiftValueLink]
VALUES = np.array([2, 5, 10, 25, 100])
M = 10000


@pytest.mark.parametrize("linkfun", REAL_LINE_LINKS)
def test_link_real_line(linkfun):
    """Test links that are defined on the real line (-inf, inf)"""
    instance = linkfun()
    x = np.linspace(-100, 100, M)
    y = instance.inverse(instance.link(x))
    print(y)
    assert np.allclose(x, y), "Links don't match"


@pytest.mark.parametrize("linkfun", POSITIVE_LINE_LINKS)
def test_link_positive_line(linkfun):
    """Test links that are defined on the positive real line (eps, inf)"""
    instance = linkfun()
    x = np.linspace(1e-10, 100, M)
    y = instance.inverse(instance.link(x))
    assert np.allclose(x, y), "Links don't match"


@pytest.mark.parametrize("linkfun", SHIFTED_LINKS)
@pytest.mark.parametrize("value", VALUES)
def test_link_positive_shifted_line(linkfun, value):
    """Test links that are shifted. This changes the domain of the links."""
    instance = linkfun(value)
    x = np.linspace(value, 100 + value, M)
    y = instance.inverse(instance.link(x))
    assert np.allclose(x, y), "Links don't match"
