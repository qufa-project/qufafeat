from datetime import datetime, timedelta

import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from dask import dataframe as dd
from scipy import stats

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
