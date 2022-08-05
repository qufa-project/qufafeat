import numpy as np
import pandas as pd
from scipy import stats
import scipy
from scipy.signal import find_peaks
from haversine import haversine
from collections import Counter
from scipy.stats import shapiro
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
    Datetime,
    Discrete,
)


class ECIsUnique(AggregationPrimitive):
    """ Detect Values outside the allowed error range
        in the column of unique values.

        Description:
            Given a list of values, detect values in a range that are not allowed in a unique column.

        Args:
            skipna (bool): Determines whether to ignore thre rows have 'NaN' values.
                           Default to True. => the rows that have "NaN" are not removed.

        Examples:
            >>> is_unique_col = ECIsUnique()
            >>> tolerance_percent = 50
            >>> is_unique_col([3, 1, 2, 3, 5], tolerance_percent)
                [True, False, True, True, False]
            We can remove the rows having 'NaN' values.

            >>> is_unique_col = ECIsUnique()
            >>> tolerance_percent = 50
            >>> is_unique_col([3, 1, 2, 3, 5, np.NaN], tolerance_percent)
                [True, False, True, True, False, False]

            >>> is_unique_col = ECIsUnique(skipna=False)
            >>> tolerance_percent = 50
            >>> is_unique_col([3, 1, 2, 3, 5, None], tolerance_percent)
               [ True, False, True, True, False]
    """
    name = "ec_is_unique"
    input_types = [[Numeric], Numeric]
    return_type = [Boolean]
    compatibility = [Library.PANDAS]
    description_template = "detect the values not allowed in unique columns"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def ec_is_unique(input_data, tolerance_percent):
            vals, counts = np.unique(input_data, return_counts=True)
            # Extract the position of reference value
            mode_index = np.argmax(counts)
            # Extract the reference value
            reference_value = vals[mode_index]

            # Calculate the upper and lower limit of the allowed range
            upper_limit = (reference_value * (1 + tolerance_percent / 100)).item()
            lower_limit = (reference_value * (1 - tolerance_percent / 100)).item()

            result = (lower_limit <= input_data) & (input_data <= upper_limit)
            return list(result)

        return ec_is_unique


class ECIsNorm(AggregationPrimitive):
    """ Determines whether a given value can be included
        in a normal distributed column.

        Description:
            Given a value, determine whether a given value can be included in a column
            that already forms a normal distribution.

        Examples:
            >>> is_norm = ECIsNorm()
            >>> norm_col = randn(10)
            >>> is_norm(norm_col)
                [True, True, True, True, True, True, True, True, True, True]

    """
    name = "ec_is_norm"
    input_types = [[Numeric], Numeric]
    return_type = [Boolean]
    description_template = "detect the values not allowed in norm columns"

    def get_function(self):
        def ec_is_norm(input_data):

            """
            [ Shapiro-Wilk Test ]
            It's the most powerful test to check the normality of a variable.
            IF the p-value <= 0.05 THEN we assume the distribution of our variable is not normal/gaussian.
            IF the p-value > 0.05 THEN we assume the distribution of our variable is normal/gaussian.
            """
            # do Shapiro-Wilk Test
            statistic, p_value = shapiro(input_data)

            if p_value > 0.05:
                df_data = pd.DataFrame(input_data, columns=['data'])
                df_data['result'] = [True for i in range(df_data.shape[0])]
                '''
                [ IQR method ] 
                It's the general method to detect outlier data
                '''
                level_q1 = df_data['data'].quantile(0.25)
                level_q3 = df_data['data'].quantile(0.75)
                iqr = level_q3 - level_q1

                df_data.loc[(df_data['data'] > level_q3 + (1.5 * iqr)) | (
                        df_data['data'] < level_q1 - (1.5 * iqr)), 'result'] = False
                return list(df_data['result'])

            else:  # fail
                return [False for i in range(len(input_data))]

        return ec_is_norm
