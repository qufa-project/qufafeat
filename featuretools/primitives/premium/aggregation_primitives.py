from datetime import datetime, timedelta

import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from dask import dataframe as dd
from scipy import stats
from scipy.signal import find_peaks
from haversine import haversine

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
    Variable
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
            return pd.Series.autocorr(values, lag=self.lag)

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
            return pd.Series.corr(values1, values2, method=self.method)

        return corr


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
                if np.isnan(values[i]):
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
                if np.isnan(values[i]):
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
                if not np.isnan(value):
                    not_nan_list.append(value)
            return (np.diff(np.sign(not_nan_list)) != 0).sum()

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
            mean = sum(array) / len(array)
            for val in array:
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
            mean = sum(array) / len(array)
            for val in array:
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
                if self.skipna == True and not val:
                    return NaN
                if val >= self.lower and val <= self.upper:
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
                if self.skipna == True and not val:
                    return NaN
                if val < self.lower or val > self.upper:
                    count += 1
            return count

        return count_outside_range
