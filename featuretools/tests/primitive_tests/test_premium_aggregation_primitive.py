import featuretools as ft

import pytest

from datetime import datetime
import numpy as np
from featuretools.primitives import (
    Autocorrelation,
    Correlation,
    CountAboveMean,
    CountBelowMean,
    CountGreaterThan,
    NMostCommonFrequency,
    NUniqueDays,
    NUniqueDaysOfCalendarYear,
    NUniqueDaysOfMonth,
    NUniqueMonths,
    NUniqueWeeks,
    NumConsecutiveGreaterMean,
    NumConsecutiveLessMean,
    NumFalseSinceLastTrue,
    NumPeaks,
    NumTrueSinceLastFalse,
    NumZeroCrossings,
    PathLength,
    PercentUnique,
    TimeSinceLastFalse,
    TimeSinceLastMax,
    TimeSinceLastMin,
    TimeSinceLastTrue,
    Variance,
    CountInsideRange,
    CountLessThan,
    CountOutsideRange,
    CountInsideNthSTD,
    CountOutsideNthSTD,
    DateFirstEvent,
    HasNoDuplicates,
    IsMonotonicallyDecreasing,
    IsMonotonicallyIncreasing,
    IsUnique,
    Kurtosis,
    MaxConsecutiveFalse,
    MaxConsecutiveNegatives,
    MaxConsecutivePositives,
    MaxConsecutiveTrue,
    MaxConsecutiveZeros,
    MaxCount,
    MaxMinDelta,
    MedianCount,
    MinCount
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
    assert count_above_mean([1, 2, 3, 4, 5]) == 2


def test_CountBelowMean():
    count_below_mean = CountBelowMean()
    assert count_below_mean([1, 2, 3, 4, 10]) == 3


def test_CountGreaterThan():
    count_greater_than = CountGreaterThan()
    assert count_greater_than([1, 2, 3, 4, 5]) == 2


def test_NMostCommonFrequency():
    n_most_common_frequency = NMostCommonFrequency(4)
    assert n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4]).to_list() == [3, 2, 2, 1]


def test_NUniqueDays():
    n_unique_days = NUniqueDays()
    times = [datetime(2019, 2, 1), datetime(2019, 2, 1), datetime(2018, 2, 1), datetime(2019, 1, 1)]
    assert n_unique_days(times) == 3


def test_NUniqueDaysOfCalendarYear():
    n_unique_days_of_calendar_year = NUniqueDaysOfCalendarYear()
    times = [datetime(2019, 2, 1), datetime(2019, 2, 1), datetime(2018, 2, 1), datetime(2019, 1, 1)]
    assert n_unique_days_of_calendar_year(times) == 2


def test_NUniqueDaysOfMonth():
    n_unique_days_of_month = NUniqueDaysOfMonth()
    times = [datetime(2019, 1, 1), datetime(2019, 2, 1), datetime(2018, 2, 1), datetime(2019, 1, 2), datetime(2019, 1, 3)]
    assert n_unique_days_of_month(times) == 3


def test_NUniqueMonths():
    n_unique_months = NUniqueMonths()
    times = [datetime(2019, 1, 1), datetime(2019, 1, 2), datetime(2019, 1, 3), datetime(2019, 2, 1), datetime(2018, 2, 1)]
    assert n_unique_months(times) == 3


def test_NUniqueWeeks():
    n_unique_weeks = NUniqueWeeks()
    times = [datetime(2018, 2, 2), datetime(2019, 1, 1), datetime(2019, 2, 1), datetime(2019, 2, 1), datetime(2019, 2, 3), datetime(2019, 2, 21)]
    assert n_unique_weeks(times) == 4


def test_NumConsecutiveGreaterMean():
    num_consecutive_greater_mean = NumConsecutiveGreaterMean(skipna=False)
    assert np.isnan(num_consecutive_greater_mean([1, 2, 3, 4, 5, 6, None]))


def test_NumConsecutiveLessMean():
    num_consecutive_less_mean = NumConsecutiveLessMean()
    assert num_consecutive_less_mean([1, 2, 3, 4, 5, 6]) == 3.0


def test_NumFalseSinceLastTrue():
    false_since_last_true = NumFalseSinceLastTrue()
    assert false_since_last_true([True, False, True, False, False]) == 2


def test_NumPeaks():
    num_peaks = NumPeaks()
    assert num_peaks([-5, 0, 10, 0, 10, -5, -4, -5, 10, 0]) == 4


def test_NumTrueSinceLastFalse():
    true_since_last_false = NumTrueSinceLastFalse()
    assert true_since_last_false([False, True, False, True, True]) == 2


def test_NumZeroCrossings():
    zero_crossings = NumZeroCrossings()
    assert zero_crossings([1, -1, 2, -2, 3, -3]) == 5


def test_PathLength():
    path_len = PathLength(unit='kilometers')
    assert path_len([(41.881832, -87.623177), (38.6270, -90.1994), (39.0997, -94.5786)]) == 805.5203180792812


def test_PercentUnique():
    percent_uniq = PercentUnique()
    assert percent_uniq([1, 1, 2, 2, 3, 4, 5, 6, 7, 8]) == 0.8


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


def test_CountInsideRange():
    count_inside_range = CountInsideRange(lower=1.5, upper=3.6)
    assert count_inside_range([1, 2, 3, 4, 5]) == 2


def test_CountLessThan():
    count_less_than = CountLessThan(threshold=3.5)
    assert count_less_than([1, 2, 3, 4, 5]) == 3


def test_CountOutsideRange():
    count_outside_range = CountOutsideRange(lower=1.5, upper=3.6)
    assert count_outside_range([1, 2, 3, 4, 5]) == 3


def test_CountInsideNthSTD():
    count_inside_nth_std = CountInsideNthSTD(n=1.5)
    assert count_inside_nth_std([1, 10, 15, 20, 100]) == 4


def test_CountOutsideNthSTD():
    count_outside_nth_std = CountOutsideNthSTD(n=1.5)
    assert count_outside_nth_std([1, 10, 15, 20, 100]) == 1


def test_DateFirstEvent():
    date_first_event = DateFirstEvent()
    first_event = date_first_event([
        datetime(2011, 4, 9, 10, 30, 10),
        datetime(2011, 4, 9, 10, 30, 20),
        datetime(2011, 4, 9, 10, 30, 30)])
    assert first_event == datetime(2011, 4, 9, 10, 30, 10)


def test_HasNoDuplicates():
    has_no_duplicates = HasNoDuplicates()
    assert has_no_duplicates([1, 1, 2]) == False


def test_IsMonotonicallyDecreasing():
    is_monotonically_decreasing = IsMonotonicallyDecreasing()
    assert is_monotonically_decreasing([9, 5, 3, 1]) == True


def test_IsMonotonicallyIncreasing():
    is_monotonically_increasing = IsMonotonicallyIncreasing()
    assert is_monotonically_increasing([1, 3, 5, 9]) == True


def test_IsUnique():
    is_unique = IsUnique()
    assert is_unique(['red', 'blue', 'green', 'blue']) == False


def test_Kurtosis():
    kurtosis = Kurtosis()
    assert kurtosis([1, 2, 3, 4, 5]) == -1.3


def test_MaxConsecutiveFalse():
    max_consecutive_false = MaxConsecutiveFalse()
    assert max_consecutive_false([True, False, False, True, True, False]) == 2


def test_MaxConsecutiveNegatives():
    max_consecutive_negatives = MaxConsecutiveNegatives()
    assert max_consecutive_negatives([1.0, -1.4, -2.4, -5.4, 2.9, -4.3]) == 3


def test_MaxConsecutivePositives():
    max_consecutive_positives = MaxConsecutivePositives()
    assert max_consecutive_positives([1.0, -1.4, 2.4, 5.4, 2.9, -4.3]) == 3


def test_MaxConsecutiveTrue():
    max_consecutive_true = MaxConsecutiveTrue()
    assert max_consecutive_true([True, False, True, True, True, False]) == 3


def test_MaxConsecutiveZeros():
    max_consecutive_zeros = MaxConsecutiveZeros()
    assert max_consecutive_zeros([1.0, -1.4, 0, 0.0, 0, -4.3]) == 3


def test_MaxCount():
    max_count = MaxCount()
    assert max_count([1, 2, 5, 1, 5, 3, 5]) == 3


def test_MaxMinDelta():
    max_min_delta = MaxMinDelta()
    assert max_min_delta([7, 2, 5, 3, 10]) == 8


def test_MedianCount():
    median_count = MedianCount()
    assert median_count([1, 2, 3, 1, 5, 3, 5]) == 2


def test_MinCount():
    min_count = MinCount()
    assert min_count([1, 2, 5, 1, 5, 3, 5]) == 2

#okay
def test_IsUnique1():
    is_unique = IsUnique()
    tolerance_percent = 100
    IsUnique([3, 1, 2, 3, 4, None], tolerance_percent)
    assert IsUnique([3, 1, 2, 3, 4, None], tolerance_percent) == [True, False, True, True, True, False]

 #okay
def test_IsUnique2():
    is_unique_col = IsUnique(skipna=False)
    tolerance_percent = 100
    IsUnique([3, 1, 2, 3, 4, None], tolerance_percent)
    assert IsUnique([3, 1, 2, 3, 4, None], tolerance_percent) == [True, False, True, True, True]

print("END")