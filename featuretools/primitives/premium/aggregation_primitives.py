from datetime import datetime, timedelta

import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from dask import dataframe as dd
from scipy import stats
import scipy
from scipy.signal import find_peaks
from haversine import haversine
from collections import Counter
import math
import statistics

from featuretools.primitives.base.aggregation_primitive_base import (
    AggregationPrimitive
)
from featuretools.utils import convert_time_units
from featuretools.utils.gen_utils import Library
from featuretools.variable_types import (
    Boolean,
    Categorical,
    DatetimeTimeIndex,
    Discrete,
    Index,
    LatLong,
    Numeric,
    Variable,
    Datetime
)


class Autocorrelation(AggregationPrimitive):
    """Determines the Pearson correlation between a series and a shifted version of the series.

    Examples:
    >>> autocorrelation = Autocorrelation()
    >>> round(autocorrelation([1, 2, 3, 1, 3, 2]), 3)
    -0.598
    """
    name = "autocorrelation"
    input_types = [[Numeric]]
    return_type = Numeric
    stack_on_self = False
    default_value = 0
    compatibility = [Library.PANDAS]
    description_template = "autocorrelation"

    def __init__(self, lag=1):
        self.lag = lag

    def get_function(self):
        def autocorr(values):
            return pd.Series.autocorr(values.astype(float), lag=self.lag)

        return autocorr


class Correlation(AggregationPrimitive):
    """Computes the correlation between two columns of values.

    Examples:
    >>> correlation = Correlation()
    >>> array_1 = [1, 4, 6, 7]
    >>> array_2 = [1, 5, 9, 7]
    >>> correlation(array_1, array_2)
    0.9221388919541468
    """
    name = "correlation"
    input_types = [[Numeric, Numeric]]
    return_type = Numeric
    stack_on_self = False
    default_value = 0
    compatibility = [Library.PANDAS]
    description_template = "correlation"

    def __init__(self, method='pearson'):
        self.method = method

    def get_function(self):
        def corr(values1, values2):
            return pd.Series.corr(values1.astype(float), values2.astype(float), method=self.method)

        return corr


class NMostCommonFrequency(AggregationPrimitive):
    """Determines the frequency of the n most common items.

    Description:
        Given a list, find the n most common items, and return a series showing the frequency of each item.
        If the list has less than n unique values, the resulting series will be padded with nan.

    Args:
        n (int) : defines "n" in "n most common".
            Defaults to 3.
        skipna (bool) : Determines if to use NA/null values.
            Defaults to True to skip NA/null.

    Examples:
        >>> n_most_common_frequency = NMostCommonFrequency()
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4]).to_list()
        [3, 2, 2]

        We can increase n to include more items.

        >>> n_most_common_frequency = NMostCommonFrequency(4)
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4]).to_list()
        [3, 2, 2, 1]

        `NaN`s are skipped by default.

        >>> n_most_common_frequency = NMostCommonFrequency(3)
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4, None, None, None]).to_list()
        [3, 2, 2]

        The way `NaN`s are treated can be controlled.

        >>> n_most_common_frequency = NMostCommonFrequency(3, skipna=False)
        >>> n_most_common_frequency([1, 1, 1, 2, 2, 3, 4, 4, None, None, None]).to_list()
        [3, 3, 2]
    """
    name = "n_most_common_frequency"
    input_types = [Discrete]
    return_type = Discrete
    description_template = "the frequency of the n most common items in {}"

    def __init__(self, n=3, skipna=True):
        self.n = n
        self.skipna = skipna

    def get_function(self):
        def most_common_freq(values):
            not_nan_list = []
            nan_cnt = 0
            for value in values:
                if value:
                    not_nan_list.append(value)
                else:
                    nan_cnt += 1
            freq_list = list(Counter(not_nan_list).values())
            if not self.skipna:
                freq_list.append(nan_cnt)

            freq_list.sort(reverse=True)
            if len(freq_list) >= self.n:
                result = freq_list[:self.n]
            else:
                result = freq_list + [np.nan] * (self.n-len(freq_list))
            return result

        return most_common_freq


class NUniqueDays(AggregationPrimitive):
    """Determines the number of unique days.

    Description:
        Given a list of datetimes, return the number of unique days.
        The same day in two different years is treated as different.
        So Feb 21, 2017 is different than Feb 21, 2019, even though they are both the 21st of February.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_days = NUniqueDays()
        >>> times = [datetime(2019, 2, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1),
        ...          datetime(2019, 1, 1)]
        >>> n_unique_days(times)
        3
    """
    name = "n_unique_days"
    input_types = [Datetime]
    return_type = Numeric
    description_template = "the number of unique days in {}"

    def get_function(self):
        def uniq_days(dates):
            uniq = {(date.year, date.month, date.day) for date in dates}
            return len(uniq)

        return uniq_days


class NUniqueDaysOfCalendarYear(AggregationPrimitive):
    """Determines the number of unique calendar days.

    Description:
        Given a list of datetimes, return the number of unique calendar days.
        The same date in two different years is counted as one.
        So Feb 21, 2017 is not unique from Feb 21, 2019.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_days_of_calendar_year = NUniqueDaysOfCalendarYear()
        >>> times = [datetime(2019, 2, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1),
        ...          datetime(2019, 1, 1)]
        >>> n_unique_days_of_calendar_year(times)
        2
    """
    name = "n_unique_days_of_calendar_year"
    input_types = [Datetime]
    return_type = Numeric
    description_template = "the number of unique calendar days in {}"

    def get_function(self):
        def uniq_cal_year(dates):
            uniq = {(date.month, date.day) for date in dates}
            return len(uniq)

        return uniq_cal_year


class NUniqueDaysOfMonth(AggregationPrimitive):
    """Determines the number of unique days of month.

    Description:
        Given a list of datetimes, return the number of unique days of month.
        The maximum value is 31.
        2018-01-01 and 2018-02-01 will be counted as 1 unique day.
        2019-01-01 and 2018-01-01 will also be counted as 1.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_days_of_month = NUniqueDaysOfMonth()
        >>> times = [datetime(2019, 1, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1),
        ...          datetime(2019, 1, 2),
        ...          datetime(2019, 1, 3)]
        >>> n_unique_days_of_month(times)
        3
    """
    name = "n_unique_days_of_month"
    input_types = [Datetime]
    return_type = Numeric
    description_template = "the number of unique days of month in {}"

    def get_function(self):
        def uniq_days_of_month(dates):
            uniq = {date.day for date in dates}
            return len(uniq)

        return uniq_days_of_month


class NUniqueMonths(AggregationPrimitive):
    """Determines the number of unique months.

    Description:
        Given a list of datetimes, return the number of unique months.
        NUniqueMonths counts absolute month, not month of year, so the same month in two different years is treated as different.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_months = NUniqueMonths()
        >>> times = [datetime(2019, 1, 1),
        ...          datetime(2019, 1, 2),
        ...          datetime(2019, 1, 3),
        ...          datetime(2019, 2, 1),
        ...          datetime(2018, 2, 1)]
        >>> n_unique_months(times)
        3
    """
    name = "n_unique_months"
    input_types = [Datetime]
    return_type = Numeric
    description_template = "the number of unique months in {}"

    def get_function(self):
        def uniq_months(dates):
            uniq = {(date.year, date.month) for date in dates}
            return len(uniq)

        return uniq_months


class NUniqueWeeks(AggregationPrimitive):
    """Determines the number of unique weeks.

    Description:
        Given a list of datetimes, return the number of unique weeks (Monday-Sunday).
        NUniqueWeeks counts by absolute week, not week of year, so the first week of 2018 and the first week of 2019 count as two unique values.

    Examples:
        >>> from datetime import datetime
        >>> n_unique_weeks = NUniqueWeeks()
        >>> times = [datetime(2018, 2, 2),
        ...          datetime(2019, 1, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2019, 2, 1),
        ...          datetime(2019, 2, 3),
        ...          datetime(2019, 2, 21)]
        >>> n_unique_weeks(times)
        4
    """
    name = "n_unique_weeks"
    input_types = [Datetime]
    return_type = Numeric
    description_template = "the number of unique weeks in {}"

    def get_function(self):
        def uniq_weeks(dates):
            date_arr = []
            for date in dates:
                date = datetime.date(date)
                date_arr.append(date)
            uniq = {(date.isocalendar()[0], date.isocalendar()[1]) for date in date_arr}
            return len(uniq)

        return uniq_weeks


class NumConsecutiveGreaterMean(AggregationPrimitive):
    """Determines the length of the longest subsequence above the mean.

    Description:
        Given a list of numbers, find the longest subsequence of numbers larger than the mean of the entire sequence.
        Return the length of the longest subsequence.

    Args:
        skipna (bool) : If this is False and any value in x is `NaN`, then the result will be `NaN`.
            If True, `NaN` values are skipped.
            Default is True.

    Examples:
        >>> num_consecutive_greater_mean = NumConsecutiveGreaterMean()
        >>> num_consecutive_greater_mean([1, 2, 3, 4, 5, 6])
        3.0

        We can control the way `NaN` values are handled.

        >>> num_consecutive_greater_mean = NumConsecutiveGreaterMean(skipna=False)
        >>> num_consecutive_greater_mean([1, 2, 3, 4, 5, 6, None])
        nan
    """
    name = "num_consecutive_greater_mean"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the length of the longest subsequence above the mean of {}"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def consecutive_greater_mean(values):
            not_non_arr = []
            for val in values:
                if val:
                    not_non_arr.append(val)
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            mean = np.mean(not_non_arr)
            max_len = 0
            cur_len = 0
            for value in not_non_arr:
                if value > mean:
                    cur_len += 1
                else:
                    cur_len = 0
                if max_len < cur_len:
                    max_len = cur_len
            return max_len

        return consecutive_greater_mean


class NumConsecutiveLessMean(AggregationPrimitive):
    """Determines the length of the longest subsequence below the mean.

    Description:
        Given a list of numbers, find the longest subsequence of numbers smaller than the mean of the entire sequence.
        Return the length of the longest subsequence.

    Args:
        skipna (bool) : If this is False and any value in x is `NaN`, then the result will be `NaN`.
            If True, `NaN` values are skipped.
            Default is True.

    Examples:
        >>> num_consecutive_less_mean = NumConsecutiveLessMean()
        >>> num_consecutive_less_mean([1, 2, 3, 4, 5, 6])
        3.0

        We can control the way `NaN` values are handled.

        >>> num_consecutive_less_mean = NumConsecutiveLessMean(skipna=False)
        >>> num_consecutive_less_mean([1, 2, 3, 4, 5, 6, None])
        nan
    """
    name = "num_consecutive_less_mean"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the length of the longest subsequence below the mean of {}"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def consecutive_less_mean(values):
            not_non_arr = []
            for val in values:
                if val:
                    not_non_arr.append(val)
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            mean = np.mean(not_non_arr)
            max_len = 0
            cur_len = 0
            for value in not_non_arr:
                if value < mean:
                    cur_len += 1
                else:
                    cur_len = 0
                if max_len < cur_len:
                    max_len = cur_len
            return max_len

        return consecutive_less_mean


class NumFalseSinceLastTrue(AggregationPrimitive):
    """Calculates the number of 'False' values since the last `True` value.

    Description:
        From a series of Booleans, find the last record with a `True` value.
        Return the count of 'False' values between that record and the end of the series.        Return nan if no values are `False`.
        Return nan if no values are `True`.
        Any nan values in the input are ignored.
        A 'True' value in the last row will result in a count of 0.
        Inputs are converted too booleans before calculating the result.

    Examples:
        >>> num_false_since_last_true = NumFalseSinceLastTrue()
        >>> num_false_since_last_true([True, False, True, False, False])
        2
    """
    name = "num_false_since_last_true"
    input_types = [Boolean]
    return_type = Numeric
    description_template = "the number of 'False' values since the last `True` value in {}"

    def get_function(self):
        def false_since_last_true(values):
            cnt = 0
            for i in range(len(values) - 1, -1, -1):
                if values[i] is None:
                    continue
                if not values[i]:
                    cnt += 1
                else:
                    return cnt
            return np.nan

        return false_since_last_true


class NumPeaks(AggregationPrimitive):
    """Determines the number of peaks in a list of numbers.

    Description:
        Given a list of numbers, count the number of local maxima.

    Examples:
        >>> num_peaks = NumPeaks()
        >>> num_peaks([-5, 0, 10, 0, 10, -5, -4, -5, 10, 0])
        4
    """
    name = "num_peaks"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the number of peaks in {}"

    def get_function(self):
        def num_peaks(values):
            return len(find_peaks(values)[0])

        return num_peaks


class NumTrueSinceLastFalse(AggregationPrimitive):
    """Calculates the number of 'True' values since the last `False` value.

    Description:
        From a series of Booleans, find the last record with a `False` value.
        Return the count of 'True' values between that record and the end of the series.
        Return nan if no values are `False`.
        Any nan values in the input are ignored.
        A 'False' value in the last row will result in a count of 0.

    Examples:
        >>> num_true_since_last_false = NumTrueSinceLastFalse()
        >>> num_true_since_last_false([False, True, False, True, True])
        2
    """
    name = "num_true_since_last_false"
    input_types = [Boolean]
    return_type = Numeric
    description_template = "the number of 'True' values since the last `False` value in {}"

    def get_function(self):
        def true_since_last_false(values):
            cnt = 0
            for i in range(len(values) - 1, -1, -1):
                if values[i] is None:
                    continue
                if values[i]:
                    cnt += 1
                else:
                    return cnt
            return np.nan

        return true_since_last_false


class NumZeroCrossings(AggregationPrimitive):
    """Determines the number of times a list crosses 0.

    Description:
        Given a list of numbers, return the number of times the value crosses 0.
        It is the number of times the value goes from a positive number to a negative number, or a negative number to a positive number.
        NaN values are ignored.

    Examples:
        >>> num_zero_crossings = NumZeroCrossings()
        >>> num_zero_crossings([1, -1, 2, -2, 3, -3])
        5
    """
    name = "num_zero_crossings"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the number of times {} crosses 0."

    def get_function(self):
        def zero_crossings(values):
            not_nan_list = []
            for value in values:
                if not (value is None):
                    not_nan_list.append(value)
            sum = 0
            for i in range(1, len(not_nan_list)):
                if (not_nan_list[i - 1] < 0 and not_nan_list[i] > 0) or (
                        not_nan_list[i - 1] > 0 and not_nan_list[i] < 0):
                    sum += 1
            return sum

        return zero_crossings


class PathLength(AggregationPrimitive):
    """Determines the length of a path defined by a series of coordinates.

    Description:
        Given a list of latitude and longitude points, sum the distance of between each subsequent point to get the total length of the path.
        Distance is calculated using the Haversine formula with a earth radius of 3958.7613 miles (6371.0088 kilometers).

    Args:
        unit (str) : Distance units used for the output.
            Defaults to miles.
            Should be one of ['miles', 'kilometers'].
        skipna (bool) : Determines whether to skip over missing latitude or longitude values.
            If True, all rows with missing values will be dropped before computing the total distance.
            If False, and missing values are present, PathLength will return np.nan.
            Defaults to True.

    Examples:
        >>> path_length = PathLength()
        >>> path_length([(41.881832, -87.623177), (38.6270, -90.1994), (39.0997, -94.5786)])
        500.52711614147347

        We can return the length in kilometers.

        >>> path_length_km = PathLength(unit='kilometers')
        >>> path_length_km([(41.881832, -87.623177), (38.6270, -90.1994), (39.0997, -94.5786)])
        805.5203180792812

        `NaN`s are skipped by default.

        >>> round(path_length([(41.881832, -87.623177), None, (39.0997, -94.5786)]), 3)
        412.765

        The way `NaN`s are treated can be controlled.

        >>> path_length_skipna = PathLength(skipna=False)
        >>> path_length_skipna([(41.881832, -87.623177), None, (39.0997, -94.5786)])
        nan
    """
    name = "path_length"
    input_types = [LatLong]
    return_type = Numeric
    description_template = "the length of {}"

    def __init__(self, unit='miles', skipna=True):
        if unit == 'miles':
            self.unit = 'mi'
        else:
            self.unit = 'km'
        self.skipna = skipna

    def get_function(self):
        def path_len(values):
            if self.skipna:
                values = list(filter(None, values))
            else:
                for value in values:
                    if not ('tuple' in str(type(value))):
                        return np.nan

            sum = 0
            for i in range(1, len(values)):
                sum += haversine(values[i-1], values[i], unit=self.unit)
            return sum

        return path_len


class PercentUnique(AggregationPrimitive):
    """Determines the percent of unique values.

    Description:
        Given a list of values, determine what percent of the list is made up of unique values.
        Multiple `NaN` values are treated as one unique value.

    Args:
        skipna (bool) : Determines whether to ignore `NaN` values.
            Defaults to True.

    Examples:
        >>> percent_unique = PercentUnique()
        >>> percent_unique([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
        0.8

        We can control whether or not `NaN` values are ignored.

        >>> percent_unique = PercentUnique()
        >>> percent_unique([1, 1, 2, None])
        0.5
        >>> percent_unique_skipna = PercentUnique(skipna=False)
        >>> percent_unique_skipna([1, 1, 2, None])
        0.75
    """
    name = "percent_unique"
    input_types = [Discrete]
    return_type = Numeric
    description_template = "the percent of unique values in {}"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def pct_uniq(values):
            return len(values.value_counts(dropna=self.skipna)) / len(values)

        return pct_uniq


class TimeSinceLastFalse(AggregationPrimitive):
    """Calculates the time since the last `False` value.

    description:
        Using a series of Datetimes and a series of Booleans, find the last record with a `False` value.
        Return the seconds elapsed between that record and the instance's cutoff time.
        Return nan if no values are `False`.

    Examples:
        >>> from datetime import datetime
        >>> time_since_last_false = TimeSinceLastFalse()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> booleans = [True, False, True]
        >>> time_since_last_false(times, booleans, time=cutoff_time)
        285.0
    """
    name = "time_since_last_false"
    input_types = [DatetimeTimeIndex, Boolean]
    return_type = Numeric
    description_template = "the time since the last `False` value in {}"

    def get_function(self):
        def time_since_last_false(values, boolean, time=None):
            index = -1
            for i in range(len(boolean)-1, -1, -1):
                if not boolean[i]:
                    index = i
                    break
            if index == -1:
                return np.nan
            time_since = time - values.iloc[index]
            return time_since.total_seconds()

        return time_since_last_false


class TimeSinceLastMax(AggregationPrimitive):
    """Calculates the time since the maximum value occurred.

    Description:
        Given a list of numbers, and a corresponding index of datetimes, find the time of the maximum value, and return the time elapsed since it occured.
        This calculation is done using an instance id's cutoff time.

    Examples:
        >>> time_since_last_max = TimeSinceLastMax()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last_max(times, [1, 3, 2], time=cutoff_time)
        285.0
    """
    name = "time_since_last_max"
    input_types = [DatetimeTimeIndex, Numeric]
    return_type = Numeric
    description_template = "the time since the maximum value occurred in {}"

    def get_function(self):
        def time_since_last_max(values, arr, time=None):
            maxIndex = arr.idxmax()
            time_since = time - values.iloc[maxIndex]
            return time_since.total_seconds()

        return time_since_last_max


class TimeSinceLastMin(AggregationPrimitive):
    """Calculates the time since the minimum value occurred.

    Description:
        Given a list of numbers, and a corresponding index of datetimes, find the time of the minimum value, and return the time elapsed since it occured.
        This calculation is done using an instance id's cutoff time.

    Examples:
        >>> time_since_last_min = TimeSinceLastMin()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last_min(times, [1, 3, 2], time=cutoff_time)
        900.0
    """
    name = "time_since_last_min"
    input_types = [DatetimeTimeIndex, Numeric]
    return_type = Numeric
    description_template = "the time since the minimum value occurred in {}"

    def get_function(self):
        def time_since_last_min(values, arr, time=None):
            minIndex = arr.idxmin()
            time_since = time - values.iloc[minIndex]
            return time_since.total_seconds()

        return time_since_last_min


class TimeSinceLastTrue(AggregationPrimitive):
    """Calculates the time since the last `True` value.

    description:
        Using a series of Datetimes and a series of Booleans, find the last record with a `True` value.
        Return the seconds elapsed between that record and the instance's cutoff time.
        Return nan if no values are `True`.

    Examples:
        >>> time_since_last_true = TimeSinceLastTrue()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> booleans = [True, True, False]
        >>> time_since_last_true(times, booleans, time=cutoff_time)
        285.0
    """
    name = "time_since_last_true"
    input_types = [DatetimeTimeIndex, Boolean]
    return_type = Numeric
    description_template = "the time since the last `True` value in {}"

    def get_function(self):
        def time_since_last_true(values, boolean, time=None):
            index = -1
            for i in range(len(boolean)-1, -1, -1):
                if boolean[i]:
                    index = i
                    break
            if index == -1:
                return np.nan
            time_since = time - values.iloc[index]
            return time_since.total_seconds()

        return time_since_last_true


class Variance(AggregationPrimitive):
    """Calculates the variance of a list of numbers.

    Description:
        Given a list of numbers, return the variance, using numpy's built-in variance function.
        Nan values in a series will be ignored.
        Return nan when the series is empty or entirely null.

    Examples:
        >>> variance = Variance()
        >>> variance([0, 3, 4, 3])
        2.25
    """
    name = "variance"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the variance of {}"

    def get_function(self):
        def var(values):
            return np.var(values)

        return var


class CountAboveMean(AggregationPrimitive):
    """Calculates the number of values that are above the mean.

    Examples:
    >>> count_above_mean = CountAboveMean()
    >>> count_above_mean([1, 2, 3, 4, 5])
    2
    """
    name = "count_above_mean"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_above_mean"
    stack_on_self = False

    def __init__(self, skipna=False):
        self.skipna = skipna

    def get_function(self):
        def count_above_mean(array):
            count = 0
            len = 0
            not_non_arr = []
            for val in array:
                if val:
                    not_non_arr.append(val)
                    len += 1
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            mean = sum(not_non_arr) / len
            for val in not_non_arr:
                if val > mean:
                    count += 1
            return count

        return count_above_mean


class CountBelowMean(AggregationPrimitive):
    """Determines the number of values that are below the mean.

    Examples:
    >>> count_below_mean = CountBelowMean()
    >>> count_below_mean([1, 2, 3, 4, 10])
    3
    """
    name = "count_below_mean"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_below_mean"
    stack_on_self = False

    def __init__(self, skipna=False):
        self.skipna = skipna

    def get_function(self):
        def count_below_mean(array):
            count = 0
            len = 0
            not_non_arr = []
            for val in array:
                if val:
                    not_non_arr.append(val)
                    len += 1
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            mean = sum(not_non_arr) / len
            for val in not_non_arr:
                if val < mean:
                    count += 1
            return count

        return count_below_mean


class CountGreaterThan(AggregationPrimitive):
    """Determines the number of values greater than a controllable threshold.

    Examples:
    >>> count_greater_than = CountGreaterThan(threshold=3)
    >>> count_greater_than([1, 2, 3, 4, 5])
    2
    """
    name = "count_greater_than"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_greater_than"
    stack_on_self = False

    def __init__(self, threshold=3):
        self.threshold = threshold

    def get_function(self):
        def count_greater_than(array):
            count = 0
            for val in array:
                if not val:
                    continue
                if val > self.threshold:
                    count += 1
            return count

        return count_greater_than


class CountInsideRange(AggregationPrimitive):
    """Determines the number of values that fall within a certain range.

    Examples:
    >>> count_inside_range = CountInsideRange(lower=1.5, upper=3.6)
    >>> count_inside_range([1, 2, 3, 4, 5])
    2
    """
    name = "count_inside_range"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_inside_range"
    stack_on_self = False

    def __init__(self, lower=0, upper=1, skipna=True):
        self.lower = lower
        self.upper = upper
        self.skipna = skipna

    def get_function(self):
        def count_inside_range(array):
            count = 0
            for val in array:
                if self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan
                elif val >= self.lower and val <= self.upper:
                    count += 1
            return count

        return count_inside_range


class CountLessThan(AggregationPrimitive):
    """Determines the number of values less than a controllable threshold.

    Examples:
    >>> count_less_than = CountLessThan(threshold=3.5)
    >>> count_less_than([1, 2, 3, 4, 5])
    3
    """
    name = "count_less_than"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_less_than"
    stack_on_self = False

    def __init__(self, threshold=3.5):
        self.threshold = threshold

    def get_function(self):
        def count_less_than(array):
            count = 0
            for val in array:
                if not val:
                    continue
                if val < self.threshold:
                    count += 1
            return count

        return count_less_than


class CountOutsideRange(AggregationPrimitive):
    """Determines the number of values that fall outside a certain range.

    Examples:
    >>> count_outside_range = CountOutsideRange(lower=1.5, upper=3.6)
    >>> count_outside_range([1, 2, 3, 4, 5])
    3
    """
    name = "count_outside_range"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_outside_range"
    stack_on_self = False

    def __init__(self, lower=0, upper=1, skipna=True):
        self.lower = lower
        self.upper = upper
        self.skipna = skipna

    def get_function(self):
        def count_outside_range(array):
            count = 0
            for val in array:
                if self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan
                elif val < self.lower or val > self.upper:
                    count += 1
            return count

        return count_outside_range


class CountInsideNthSTD(AggregationPrimitive):
    """Determines the count of observations that lie inside the first N standard deviations (inclusive).

    Examples:
    >>> count_inside_nth_std = CountInsideNthSTD(n=1.5)
    >>> count_inside_nth_std([1, 10, 15, 20, 100])
    4
    """
    name = "count_inside_nth_std"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_inside_nth_std"
    stack_on_self = False

    def __init__(self, n=1):
        self.n = n

    def get_function(self):
        def count_inside_nth_std(array):
            count = 0
            not_non_arr = []
            for val in array:
                if not val:
                    continue
                else:
                    not_non_arr.append(val)

            mean = sum(not_non_arr) / len(array)
            std = 0
            for val in not_non_arr:
                std += (val-mean) ** 2
            std_dev = std / len(array)
            std_dev = math.sqrt(std_dev)
            for val in not_non_arr:
                if val <= std_dev:
                    count += 1
            return count

        return count_inside_nth_std


class CountOutsideNthSTD(AggregationPrimitive):
    """Determines the number of observations that lie outside the first N standard deviations.

    Examples:
    >>> count_outside_nth_std = CountOutsideNthSTD(n=1.5)
    >>> count_outside_nth_std([1, 10, 15, 20, 100])
    1
    """
    name = "count_outside_nth_std"
    input_types = [Numeric]
    return_type = Numeric
    default_value = 0
    description_template = "count_outside_nth_std"
    stack_on_self = False

    def __init__(self, n=1):
        self.n = n

    def get_function(self):
        def count_outside_nth_std(array):
            count = 0
            not_non_arr = []
            for val in array:
                if not val:
                    continue
                else:
                    not_non_arr.append(val)

            mean = sum(not_non_arr) / len(array)
            std = 0
            for val in not_non_arr:
                std += (val-mean) ** 2
            std_dev = std / len(array)
            std_dev = math.sqrt(std_dev)
            for val in not_non_arr:
                if val > std_dev:
                    count += 1
            return count

        return count_outside_nth_std


class DateFirstEvent(AggregationPrimitive):
    """Determines the first datetime from a list of datetimes.

    Examples:
    >>> date_first_event = DateFirstEvent()
    >>> date_first_event([
    ...     datetime(2011, 4, 9, 10, 30, 10),
    ...     datetime(2011, 4, 9, 10, 30, 20),
    ...     datetime(2011, 4, 9, 10, 30, 30)])
    Timestamp('2011-04-09 10:30:10')
    """
    name = "date_first_event"
    input_types = [DatetimeTimeIndex]
    return_type = Datetime
    default_value = 0
    description_template = "date_first_event"
    stack_on_self = False

    def get_function(self):
        def date_first_event(datetimes):
            sorted_list = datetimes.sort_values()
            return sorted_list[0]

        return date_first_event


class HasNoDuplicates(AggregationPrimitive):
    """Determines if there are duplicates in the input.

    Examples:
    >>> has_no_duplicates = HasNoDuplicates()
    >>> has_no_duplicates([1, 1, 2])
    False
    >>> has_no_duplicates([1, 2, 3])
    True    
    """
    name = "has_no_duplicates"
    input_types = [Discrete]
    return_type = Boolean
    default_value = 0
    description_template = "has_no_duplicates"
    stack_on_self = False

    def get_function(self):
        def has_no_duplicates(numbers):
            no_duplicates = set(numbers)
            if len(numbers) != len(no_duplicates):
                return False
            else:
                return True

        return has_no_duplicates


class IsMonotonicallyDecreasing(AggregationPrimitive):
    """Determines if a series is monotonically decreasing.

    Description:
        Given a list of numeric values, return True if the values are strictly decreasing.
        If the series contains `NaN` values, they will be skipped.

    Examples:
        >>> is_monotonically_decreasing = IsMonotonicallyDecreasing()
        >>> is_monotonically_decreasing([9, 5, 3, 1])
        True
    """
    name = "is_monotonically_decreasing"
    input_types = [Numeric]
    return_type = Boolean

    def get_function(self):
        def is_monotonically_decreasing(numbers):
            decrease = numbers.sort_values(ascending=False)
            if decrease.equals(numbers):
                return True
            else: return False

        return is_monotonically_decreasing


class IsMonotonicallyIncreasing(AggregationPrimitive):
    """Determines if a series is monotonically increasing.

    Description:
        Given a list of numeric values, return True if the values are strictly increasing.
        If the series contains `NaN` values, they will be skipped.

    Examples:
        >>> is_monotonically_increasing = IsMonotonicallyIncreasing()
        >>> is_monotonically_increasing([1, 3, 5, 9])
        True
    """
    name = "is_monotonically_increasing"
    input_types = [Numeric]
    return_type = Boolean

    def get_function(self):
        def is_monotonically_increasing(numbers):
            decrease = numbers.sort_values()
            if decrease.equals(numbers):
                return True
            else: return False

        return is_monotonically_increasing


class IsUnique(AggregationPrimitive):
    """Determines whether or not a series of discrete is all unique.

    Description:
        Given a series of discrete values, return True if each value in the series is unique.
        If any value is repeated, return False.

    Examples:
        >>> is_unique = IsUnique()
        >>> is_unique(['red', 'blue', 'green', 'yellow'])
        True
    """
    name = "is_unique"
    input_types = [Discrete]
    return_type = Boolean

    def get_function(self):
        def is_unique(array):
            unique = set(array)
            if len(unique) == len(array):
                return True
            else: return False

        return is_unique


class Kurtosis(AggregationPrimitive):
    """Calculates the kurtosis for a list of numbers

    Examples:
        >>> kurtosis = Kurtosis()
        >>> kurtosis([1, 2, 3, 4, 5])
        -1.3
    """
    name = "kurtosis"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, fisher = True, bias = True, nan_policy = "propagate"):
        self.fisher = fisher
        self.bias = bias
        self.nan_policy = nan_policy

    def get_function(self):
        def kurtosis(array):
            array = list(filter(None, array))
            return scipy.stats.kurtosis(array, None, self.fisher, self.bias, self.nan_policy)

        return kurtosis


class MaxConsecutiveFalse(AggregationPrimitive):
    """Determines the maximum number of consecutive False values in the input

    Examples:
        >>> max_consecutive_false = MaxConsecutiveFalse()
        >>> max_consecutive_false([True, False, False, True, True, False])
        2
    """
    name = "max_consecutive_false"
    input_types = [Boolean]
    return_type = Numeric

    def __init__(self, skipna = True):
        self.skipna = skipna

    def get_function(self):
        def max_consecutive_false(array):
            max_count = 0
            count = 0
            for val in array:
                if self.skipna and val == None:
                    continue
                elif not self.skipna and val == None:
                    count = 0
                elif val == False:
                    count += 1
                else:
                    max_count = max(max_count, count)
                    count = 0
            return max_count

        return max_consecutive_false


class MaxConsecutiveNegatives(AggregationPrimitive):
    """Determines the maximum number of consecutive negative values in the input

    Examples:
        >>> max_consecutive_negatives = MaxConsecutiveNegatives()
        >>> max_consecutive_negatives([1.0, -1.4, -2.4, -5.4, 2.9, -4.3])
        3
    """
    name = "max_consecutive_negatives"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = True):
        self.skipna = skipna

    def get_function(self):
        def max_consecutive_negatives(array):
            max_count = 0
            count = 0
            for val in array:
                if self.skipna and val == None:
                    continue
                elif not self.skipna and val == None:
                    count = 0
                elif val < 0:
                    count += 1
                else:
                    max_count = max(max_count, count)
                    count = 0
            return max_count

        return max_consecutive_negatives


class MaxConsecutivePositives(AggregationPrimitive):
    """Determines the maximum number of consecutive positive values in the input

    Examples:
        >>> max_consecutive_positives = MaxConsecutivePositives()
        >>> max_consecutive_positives([1.0, -1.4, 2.4, 5.4, 2.9, -4.3])
        3
    """
    name = "max_consecutive_positives"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = True):
        self.skipna = skipna

    def get_function(self):
        def max_consecutive_positives(array):
            max_count = 0
            count = 0
            for val in array:
                if self.skipna and val == None:
                    continue
                elif not self.skipna and val == None:
                    count = 0
                elif val > 0:
                    count += 1
                else:
                    max_count = max(max_count, count)
                    count = 0
            return max_count

        return max_consecutive_positives


class MaxConsecutiveTrue(AggregationPrimitive):
    """Determines the maximum number of consecutive True values in the input

    Examples:
        >>> max_consecutive_true = MaxConsecutiveTrue()
        >>> max_consecutive_true([True, False, True, True, True, False])
        3
    """
    name = "max_consecutive_true"
    input_types = [Boolean]
    return_type = Numeric

    def __init__(self, skipna = True):
        self.skipna = skipna

    def get_function(self):
        def max_consecutive_true(array):
            max_count = 0
            count = 0
            for val in array:
                if self.skipna and val == None:
                    continue
                elif not self.skipna and val == None:
                    count = 0
                elif val == True:
                    count += 1
                else:
                    max_count = max(max_count, count)
                    count = 0
            return max_count

        return max_consecutive_true


class MaxConsecutiveZeros(AggregationPrimitive):
    """Determines the maximum number of consecutive zero values in the input

    Examples:
        >>> max_consecutive_zeros = MaxConsecutiveZeros()
        >>> max_consecutive_zeros([1.0, -1.4, 0, 0.0, 0, -4.3])
        3
    """
    name = "max_consecutive_zeros"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = True):
        self.skipna = skipna

    def get_function(self):
        def max_consecutive_zeros(array):
            max_count = 0
            count = 0
            for val in array:
                if self.skipna and val == None:
                    continue
                elif not self.skipna and val == None:
                    count = 0
                elif val == 0:
                    count += 1
                else:
                    max_count = max(max_count, count)
                    count = 0
            return max_count

        return max_consecutive_zeros


class MaxCount(AggregationPrimitive):
    """Calculates the number of occurrences of the max value in a list

    Examples:
        >>> max_count = MaxCount()
        >>> max_count([1, 2, 5, 1, 5, 3, 5])
        3
    """
    name = "max_count"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = False):
        self.skipna = skipna

    def get_function(self):
        def max_count(array):
            not_non_arr = []
            for val in array:
                if val:
                    not_non_arr.append(val)
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            max_value = max(not_non_arr)
            count = list(array).count(max_value)
            return count

        return max_count


class MaxMinDelta(AggregationPrimitive):
    """Determines the difference between the max and min value.

    Examples:
        >>> max_min_delta = MaxMinDelta()
        >>> max_min_delta([7, 2, 5, 3, 10])
        8
    """
    name = "max_min_delta"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = False):
        self.skipna = skipna

    def get_function(self):
        def max_min_delta(array):
            not_non_arr = []
            for val in array:
                if val:
                    not_non_arr.append(val)
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            max_value = max(not_non_arr)
            min_value = min(not_non_arr)
            return (max_value - min_value)

        return max_min_delta


class MedianCount(AggregationPrimitive):
    """Calculates the number of occurrences of the median value in a list

    Examples:
        >>> median_count = MedianCount()
        >>> median_count([1, 2, 3, 1, 5, 3, 5])
        2
    """
    name = "median_count"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = False):
        self.skipna = skipna

    def get_function(self):
        def median_count(numbers):
            median_value = statistics.median(numbers.astype(float))
            count = list(numbers).count(median_value)
            return count

        return median_count


class MinCount(AggregationPrimitive):
    """Calculates the number of occurrences of the min value in a list

    Examples:
        >>> min_count = MinCount()
        >>> min_count([1, 2, 5, 1, 5, 3, 5])
        2
    """
    name = "min_count"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna = False):
        self.skipna = skipna

    def get_function(self):
        def min_count(array):
            not_non_arr = []
            for val in array:
                if val:
                    not_non_arr.append(val)
                elif self.skipna and not val:
                    continue
                elif not self.skipna and not val:
                    return np.nan

            min_value = min(not_non_arr)
            count = list(array).count(min_value)
            return count

        return min_count

class IsUnique(AggregationPrimitive):
    """ Detect Values outside the allowed error range
        in the column of unique values.

        Description:
            Given a list of values, detect values in a range that are not allowed in a unique column.

        Args:
            skipna (bool): Determines whether to ignore thre rows have 'NaN' values.
                           Default to True. => the rows that have "NaN" are not removed.

        Examples:
            >>> is_unique_col = IsUnique()
            >>> tolerance_percent = 100
            >>> IsUnique([3, 1, 2, 3, 4], tolerance_percent)
                [True, False, True, True, True]

            We can remove the rows having 'NaN' values.

            >>> is_unique_col = IsUnique()
            >>> tolerance_percent = 100
            >>> IsUnique([3, 1, 2, 3, 4, None], tolerance_percent)
                [True, False, True, True, True, False]

            >>> is_unique_col = IsUnique(skipna=False)
            >>> tolerance_percent = 100
            >>> IsUnique([3, 1, 2, 3, 4, None], tolerance_percent)
               [ True, False, True, True, True]
    """

    name = "is_unique"
    input_types = [Discrete]
    return_type = [Discrete]
    description_template = "detect the values not allowed in unique columns"

    def __init__(self, skipna=True):
        self.skipna = skipna


    def get_function(self):
        def is_uniq(input_data, tolerance_percent):
            df1 = pd.DataFrame(data=input_data)
            df1.rename(columns={0: "value"}, inplace=True)
            if self.skipna == False:
                df1.dropna(inplace=True)

            ## Extract the reference value
            df2 = input_data.value_counts()
            df2 = pd.DataFrame(data=df2)
            df2.reset_index(inplace=True)
            df2.rename(columns={"index": "value", 0: "count"}, inplace=True)
            reference_value = df2['value'].values[0]

            ## Calculate allowed range
            upper_tolerance = reference_value * (1 + tolerance_percent/100)
            lower_tolerance = reference_value * (1 - tolerance_percent/100)

            ## Detect values in unacceptable ranges
            result = []
            for index, row in df1.iterrows():
                if row['value'] >= lower_tolerance and row['value'] <= upper_tolerance:
                    result.append(True)
                else:
                    result.append(False)
            return result
        return is_uniq
