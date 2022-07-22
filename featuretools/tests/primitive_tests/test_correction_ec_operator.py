import numpy as np
from numpy.random import seed
from numpy.random import randn

from featuretools.primitives import (
    ECIsUnique,
    ECIsNorm
)


def test_ECIsUnique():
    is_ECunique = ECIsUnique()
    tolerance_percent = 50
    assert is_ECunique([3, 1, 2, 3, 5, np.NaN], tolerance_percent) == [True, False, True, True, False, False]


def test_ECIsNorm():
    is_ECnorm = ECIsNorm()
    seed(0)
    norm_col = randn(10)
    assert (is_ECnorm(norm_col) == [True, True, True, True, True, True, True, True, True, True])
