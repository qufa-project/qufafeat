from datetime import datetime, timedelta

import numpy as np
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
