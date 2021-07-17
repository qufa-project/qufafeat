import featuretools as ft
import numpy as np

import pytest

from featuretools.primitives import (
    AbsoluteDiff
)


def test_AbsoluteDiff():
    absdiff = AbsoluteDiff()
    res = absdiff([3.0, -5.0, -2.4]).tolist()
    assert np.isnan(res[0])
    assert res[1:] == [ 8.0, 2.6]
