import featuretools as ft

import pytest

from featuretools.primitives import (
    Autocorrelation,
    Correlation,
    CountAboveMean,
    CountBelowMean,
    CountGreaterThan
)

def test_AutoCorrelation():
    autocorr = Autocorrelation()
    assert round(autocorr([1, 2, 3, 1, 3, 2]), 3) == -0.598


def test_Correlation():
    corr = Correlation(method='kendall')
    array_1 = [1, 4, 6, 7]
    array_2 = [1, 5, 9, 7]
    assert round(corr(array_1, array_2), 5) == 0.66667


def test_CountAboveMean():
    count_above_mean = CountAboveMean()
    assert count_above_mean([1, 2, 3, 4, 5])


def test_CountBelowMean():
    count_below_mean = CountBelowMean()
    assert count_below_mean([1, 2, 3, 4, 10])


def test_CountGreaterThan():
    count_greater_than = CountGreaterThan()
    assert count_greater_than([1, 2, 3, 4, 5])