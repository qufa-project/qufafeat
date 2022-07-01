import numpy as np

from featuretools.primitives import (
    ECIsUnique
)


def test_ECIsUnique():
    is_ECunique = ECIsUnique()
    tolerance_percent = 50
    assert is_ECunique([3, 1, 2, 3, 5, np.NaN], tolerance_percent) == [True, False, True, True, False, False]
