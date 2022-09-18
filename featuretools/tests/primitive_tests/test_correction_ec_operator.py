import numpy as np
from numpy import random
from numpy.random import seed
from numpy.random import randn

from featuretools.primitives import (
    ECIsUnique,
    ECIsNorm,
    ECHasInclude
)


def test_ECIsUnique():
    is_ECunique = ECIsUnique()
    tolerance_percent = 50
    assert is_ECunique([3, 1, 2, 3, 5, np.NaN], tolerance_percent) == [True, False, True, True, False, False]


def test_ECIsNorm_1():
    is_ECnorm = ECIsNorm()
    np.random.seed(seed=100)
    data = random.normal(loc=10, scale=5, size=100)  # (m = 10, s = 5, size = 100) normal distribution
    data = data.tolist()
    data.insert(0, 30)

    expected_result = [True for _ in range(len(data))]
    expected_result[0] = False

    assert (is_ECnorm(data) == expected_result)


def test_ECIsNorm_2():
    is_ECnorm = ECIsNorm()
    np.random.seed(seed=100)
    data = random.normal(loc=10, scale=5, size=100)  # (m = 10, s = 5, size = 100) normal distribution
    data = data.tolist()
    data.insert(0, 30)
    data.insert(80, 100)

    expected_result = [False for _ in range(len(data))]

    assert (is_ECnorm(data) == expected_result)


def test_ECHasInclude_1():
    has_ECInclude = ECHasInclude()

    candidate_col = [1, 2, 2, 4, 6, 8, 10]
    criteria_col = [1, 2, 4, 4, 6, 8, 8, 8, 10, 12]

    assert (has_ECInclude(candidate_col, criteria_col) == [True for i in range(len(candidate_col))])

def test_ECHasInclude_2():
    has_ECInclude = ECHasInclude()

    candidate_col = [1, 2, 2, 4, 6, 6, 8, 10]
    criteria_col = [1, 2, 3, 4, 5, 7, 8, 10]

    expected_result = [True for i in range(len(candidate_col))]
    expected_result[4] = False
    expected_result[5] = False

    assert (has_ECInclude(candidate_col, criteria_col) == expected_result)
