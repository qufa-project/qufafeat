import numpy as np
import pandas as pd
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
