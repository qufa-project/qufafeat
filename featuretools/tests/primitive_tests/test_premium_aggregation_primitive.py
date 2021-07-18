import featuretools as ft

import pytest

from datetime import datetime
from featuretools.primitives import (
    Autocorrelation,
    Correlation,
    CountAboveMean,
    CountBelowMean,
    CountGreaterThan,
    TimeSinceLastFalse,
    TimeSinceLastMax,
    TimeSinceLastMin,
    TimeSinceLastTrue,
    Variance
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


def test_TimeSinceLastFalse():
    time_since_last_false = TimeSinceLastFalse()
    cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
    times = [datetime(2010, 1, 1, 11, 45, 0), datetime(2010, 1, 1, 11, 55, 15), datetime(2010, 1, 1, 11, 57, 30)]
    booleans = [True, False, True]
    assert time_since_last_false(times, booleans, time=cutoff_time) == 285.0


def test_TimeSinceLastMax():
    time_since_last_max = TimeSinceLastMax()
    cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
    times = [datetime(2010, 1, 1, 11, 45, 0), datetime(2010, 1, 1, 11, 55, 15), datetime(2010, 1, 1, 11, 57, 30)]
    assert time_since_last_max(times, [1, 3, 2], time=cutoff_time) == 285.0


def test_TimeSinceLastMin():
    time_since_last_min = TimeSinceLastMin()
    cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
    times = [datetime(2010, 1, 1, 11, 45, 0), datetime(2010, 1, 1, 11, 55, 15), datetime(2010, 1, 1, 11, 57, 30)]
    assert time_since_last_min(times, [1, 3, 2], time=cutoff_time) == 900.0


def test_TimeSinceLastTrue():
    time_since_last_true = TimeSinceLastTrue()
    cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
    times = [datetime(2010, 1, 1, 11, 45, 0), datetime(2010, 1, 1, 11, 55, 15), datetime(2010, 1, 1, 11, 57, 30)]
    booleans = [True, True, False]
    assert time_since_last_true(times, booleans, time=cutoff_time) == 285.0


def test_Variance():
    variance = Variance()
    assert variance([0, 3, 4, 3, None]) == 2.25

