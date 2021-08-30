from collections import deque
from featuretools.variable_types.variable import Discrete
import warnings

import numpy as np

import string
import pandas
import re
import math

from datetime import datetime, timedelta
from pyzipcode import ZipCodeDatabase
from pandas import Series
from scipy.signal import savgol_filter
from scipy.stats import stats
from phone_iso3166.country import phone_country
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.utils import convert_time_units
from featuretools.utils.entity_utils import replace_latlong_nan
from featuretools.utils.gen_utils import Library
from featuretools.variable_types import (
    Boolean,
    Categorical,
    DateOfBirth,
    Datetime,
    DatetimeTimeIndex,
    LatLong,
    NaturalLanguage,
    Numeric,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    URL,
    Variable,
    ZIPCode
)


class AbsoluteDiff(TransformPrimitive):
    """Computes the absolute diff of a number.

    Examples:
        >>> absdiff = AbsoluteDiff()
        >>> absdiff([3.0, -5.0, -2.4]).tolist()
        [nan, 8.0, 2.6]
    """
    name = "absolute_diff"
    input_types = [Numeric]
    return_type = Numeric
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the absolute diff of {}"

    def get_function(self):
        def func_absdiff(values):
            return np.absolute(np.diff(values, prepend=float('nan')))
        return func_absdiff


class AgeOverN(TransformPrimitive):
    input_types = [DateOfBirth]
    return_type = Boolean
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]

    def get_function_helper(self, overN):
        def age(x, time=None):
            return (time - x).dt.days / 365 > overN
        return age


class AgeOver18(AgeOverN):
    """Determines whether a person is over 18 years old given their date of birth.

    Description:
        Returns True if the person's age is greater than or equal to 18 years.
        Returns False if the age is less than 18 years of age.
        Returns nan if the age is not defined or doesn't exist..

    Examples:
        Determine whether someone born on Jan 1, 2000 is over 18 years old as of January 1, 2019.

        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age_over_18 = AgeOver18()
        >>> input_ages = [pd.to_datetime("01-01-2000"), pd.to_datetime("06-01-2010")]
        >>> age_over_18(input_ages, time=reference_date).tolist()
        [True, False]
    """
    name = "age_over_18"
    description_template = "the age over 18 from {}"

    def get_function(self):
        return AgeOverN.get_function_helper(self, 18)


class AgeOver25(AgeOverN):
    """Determines whether a person is over 25 years old given their date of birth.

    Description:
        Returns True if the person's age is greater than or equal to 25 years.
        Returns False if the age is less than 25 years of age.
        Returns nan if the age is not defined or doesn't exist..

    Examples:
        Determine whether someone born on Jan 1, 2000 is over 25 years old as of January 1, 2019.

        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age_over_25 = AgeOver25()
        >>> input_ages = [pd.to_datetime("01-01-2000"), pd.to_datetime("06-01-1990")]
        >>> age_over_25(input_ages, time=reference_date).tolist()
        [False, True]
    """
    name = "age_over_25"
    description_template = "the age over 25 from {}"

    def get_function(self):
        return AgeOverN.get_function_helper(self, 25)


class AgeOver65(AgeOverN):
    """Determines whether a person is over 65 years old given their date of birth.

    Description:
        Returns True if the person's age is greater than or equal to 65 years.
        Returns False if the age is less than 65 years of age.
        Returns nan if the age is not defined or doesn't exist..

    Examples:
        Determine whether someone born on Jan 1, 1950 is over 65 years old as of January 1, 2019.

        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age_over_65 = AgeOver65()
        >>> input_ages = [pd.to_datetime("01-01-1950"), pd.to_datetime("01-01-2000")]
        >>> age_over_65(input_ages, time=reference_date).tolist()
        [True, False]
    """
    name = "age_over_65"
    description_template = "the age over 65 from {}"

    def get_function(self):
        return AgeOverN.get_function_helper(self, 65)


class AgeUnderN(TransformPrimitive):
    input_types = [DateOfBirth]
    return_type = Boolean
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the age under 18 from {}"

    def get_function_helper(self, underN):
        def age(x, time=None):
            return (time - x).dt.days / 365 < underN
        return age


class AgeUnder18(AgeUnderN):
    """Determines whether a person is under 18 years old given their date of birth.

    Description:
        Returns True if the person's age is less than 18 years.
        Returns False if the age is more than or equal to 18 years.
        Returns np.nan if the age is not defined, or doesn't exist.

    Examples:
        Determine whether someone born on Jan 1, 2000 is under 18 years old as of January 1, 2019.

        >>>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age_under_18 = AgeUnder18()
        >>> input_ages = [pd.to_datetime("01-01-2000"), pd.to_datetime("06-01-2010")]
        >>> age_under_18(input_ages, time=reference_date).tolist()
        [False, True]
    """
    name = "age_under_18"
    description_template = "the age under 18 from {}"

    def get_function(self):
        return AgeUnderN.get_function_helper(self, 18)



class AgeUnder65(AgeUnderN):
    """Determines whether a person is under 65 years old given their date of birth.

    Description:
        Returns True if the person's age is less than 65 years.
        Returns False if the age is more than or equal to 65 years.
        Returns np.nan if the age is not defined, or doesn't exist.

    Examples:
        Determine whether two people are under age 65 as of January 1, 2019.

        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age_under_65 = AgeUnder65()
        >>> input_ages = [pd.to_datetime("01-01-1950"),
        ...               pd.to_datetime("06-01-2010")]
        >>> age_under_65(input_ages, time=reference_date).tolist()
        [False, True]
    """
    name = "age_under_65"
    description_template = "the age under 65 from {}"

    def get_function(self):
        return AgeUnderN.get_function_helper(self, 65)


class NaturalLanguageToYear(TransformPrimitive):
    """Extracts the year from a string

    Description:
        If a year is present in a string, extract the year.
        This will only match years between 1800 and 2199.
        Years will not be extracted if immediately preceeded or followed by another number or letter.
        If there are multiple years present in a string, only the first year will be returned.

    Examples:
        >>> text_to_year = NaturalLanguageToYear()
        >>> array = pd.Series(["The year was 1887.",
        ...                    "This string has no year",
        ...                    "Toy Story (1995)",
        ...                    "12451997abc"])
        >>> text_to_year(array).tolist()
        ['1887', nan, '1995', nan]
    """
    name = "natural_language_to_year"
    input_types = [NaturalLanguage]
    return_type = Ordinal
    description_template = "the year from {}"

    def get_function(self):
        def lang_to_year(values):
            result = []
            for value in values:
                numbers = re.findall('\d+', value)
                find = False
                for number in numbers:
                    if 1800 <= int(number) < 2200:
                        result.append(int(number))
                        find = True
                        break
                if not find:
                    result.append(np.nan)
            return np.array(result)

        return lang_to_year


class NthWeekOfMonth(TransformPrimitive):
    """Determines the nth week of the month from a given date.

    Description:
        Converts a datetime to an float representing the week of the month in which the date falls.
        The first day of the month starts week 1, and the week number is incremented each Sunday.

    Examples:
        >>> from datetime import datetime
        >>> nth_week_of_month = NthWeekOfMonth()
        >>> times = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3),
        ...          datetime(2019, 3, 31),
        ...          datetime(2019, 3, 30)]
        >>> nth_week_of_month(times).tolist()
        [1.0, 2.0, 6.0, 5.0]
    """
    name = "nth_week_of_month"
    input_types = [Datetime]
    return_type = Numeric
    description_template = "the nth week of the month from {}"

    def get_function(self):
        def nth_week(dates):
            result = []
            for date in dates:
                first_day = date.replace(day=1)
                if first_day.weekday() < 6:
                    first_day = first_day - timedelta(days=first_day.weekday()+1)
                result.append((date - first_day).days // 7 + 1)
            return np.array(result)

        return nth_week


class PartOfDay(TransformPrimitive):
    """Determines what part of the day a particular time value falls in.

    Description:
        Given a list of datetimes, determine part of day based on the hour.
        The options are: Morning (5am-11am), Afternoon (12pm-5pm), Evening (6pm-9pm), or Night (10pm-4am).
        If the date is missing, return `NaN`.

    Examples:
        >>> from datetime import datetime
        >>> part_of_day = PartOfDay()
        >>> times = [datetime(2010, 1, 1, 1, 45, 0),
        ...          datetime(2010, 1, 1, 8, 55, 15),
        ...          datetime(2010, 1, 1, 16, 55, 15),
        ...          datetime(2010, 1, 1, 23, 57, 30)]
        >>> part_of_day(times).tolist()
        ['Night', 'Morning', 'Afternoon', 'Night']
    """
    name = "part_of_day"
    input_types = [Datetime]
    return_type = Categorical
    description_template = "what part of the day {} falls in"

    def get_function(self):
        def part_of_day(values):
            result = []
            for value in values:
                hour = value.hour
                if 5 <= hour <= 11:
                    result.append('Morning')
                elif 12 <= hour <= 17:
                    result.append('Afternoon')
                elif 18 <= hour <= 21:
                    result.append('Evening')
                else:
                    result.append('Night')
            return np.array(result)

        return part_of_day


class PercentChange(TransformPrimitive):
    """Determines the percent difference between values in a list.

    Description:
        Given a list of numbers, return the percent difference between each subsequent number.
        Percentages are shown in decimal form (not multiplied by 100).

    Args:
        periods (int) : Periods to shift for calculating percent change.
            Default is 1.
        fill_method (str) : Method for filling gaps in reindexed Series.
            Valid options are `backfill`, `bfill`, `pad`, `ffill`.
            `pad / ffill`: fill gap with last valid observation.
            `backfill / bfill`: fill gap with next valid observation.
            Default is `pad`.
        limit (int) : The max number of consecutive NaN values in a gap that can be filled.
            Default is None.
        freq (DateOffset, timedelta, or str) : Instead of calcualting change between subsequent points, PercentChange will calculate change between points with a certain interval between their date indices.
            `freq` defines the desired interval.
            When freq is used, the resulting index will also be filled to include any missing dates from the specified interval.
            If the index is not date/datetime and freq is used, it will raise a NotImplementedError.
            If freq is None, no changes will be applied.
            Default is None

    Examples:
        >>> percent_change = PercentChange()
        >>> percent_change([2, 5, 15, 3, 3, 9, 4.5]).to_list()
        [nan, 1.5, 2.0, -0.8, 0.0, 2.0, -0.5]

        We can control the number of periods to return the percent difference between points further from one another.

        >>> percent_change_2 = PercentChange(periods=2)
        >>> percent_change_2([2, 5, 15, 3, 3, 9, 4.5]).to_list()
        [nan, nan, 6.5, -0.4, -0.8, 2.0, 0.5]

        We can control the method used to handle gaps in data.

        >>> percent_change = PercentChange()
        >>> percent_change([2, 4, 8, None, 16, None, 32, None]).to_list()
        [nan, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        >>> percent_change_backfill = PercentChange(fill_method='backfill')
        >>> percent_change_backfill([2, 4, 8, None, 16, None, 32, None]).to_list()
        [nan, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, nan]

        We can control the maximum number of NaN values to fill in a gap.

        >>> percent_change = PercentChange()
        >>> percent_change([2, None, None, None, 4]).to_list()
        [nan, 0.0, 0.0, 0.0, 1.0]
        >>> percent_change_limited = PercentChange(limit=2)
        >>> percent_change_limited([2, None, None, None, 4]).to_list()
        [nan, 0.0, 0.0, nan, nan]

        We can specify a date frequency on which to calculate percent change.

        >>> import pandas as pd
        >>> dates = pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-05'])
        >>> x_indexed = pd.Series([1, 2, 3, 4], index=dates)
        >>> percent_change = PercentChange()
        >>> percent_change(x_indexed).to_list()
        [nan, 1.0, 0.5, 0.33333333333333326]
        >>> date_offset = pd.tseries.offsets.DateOffset(1)
        >>> percent_change_freq = PercentChange(freq=date_offset)
        >>> percent_change_freq(x_indexed).to_list()
        [nan, 1.0, 0.5, nan]
    """
    name = "percent_change"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the percent difference between values in {}"

    def __init__(self, periods=1, fill_method='pad', limit=None, freq=None):
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        self.freq = freq

    def get_function(self):
        def pct_change(values):
            return values.pct_change(periods=self.periods, fill_method=self.fill_method, limit=self.limit, freq=self.freq)

        return pct_change


class PhoneNumberToCountry(TransformPrimitive):
    """Determines the country of a phone number.

    Description:
        Given a list of phone numbers, return the country of each one, based on the country code.
        If a phone number is missing or invalid, return np.nan.

    Examples:
        >>> phone_number_to_country = PhoneNumberToCountry()
        >>> phone_number_to_country(['+55 85 5555555', '+81 55-555-5555', '+1-541-754-3010',]).tolist()
        ['BR', 'JP', 'US']
    """
    name = "phone_number_to_country"
    input_types = [PhoneNumber]
    return_type = Categorical
    description_template = "the country of {}"

    def get_function(self):
        def phone_to_country(values):
            result = []
            for value in values:
                result.append(phone_country(value))
            return np.array(result)

        return phone_to_country


class PolarityScore(TransformPrimitive):
    """Calculates the polarity of a text on a scale from -1 (negative) to 1 (positive)

    Description:
        Given a list of strings assign a polarity score from -1 (negative text), to 0 (neutral text), to 1 (positive text).
        The function returns a score for every given piece of text.
        If a string is missing, return 'NaN'

    Examples:
        >>> x = ['He loves dogs', 'She hates cats', 'There is a dog', '']
        >>> polarity_score = PolarityScore()
        >>> polarity_score(x).tolist()
        [0.677, -0.649, 0.0, 0.0]
    """
    name = "polarity_score"
    input_types = [NaturalLanguage]
    return_type = Numeric
    description_template = "the polarity of {} on a scale from -1 to 1"

    def get_function(self):
        def polarity_score(values):
            result = []
            analyazer = SentimentIntensityAnalyzer()
            for value in values:
                result.append(analyazer.polarity_scores(value)['compound'])
            return np.array(result)

        return polarity_score


class PunctuationCount(TransformPrimitive):
    """Determines number of punctuation characters in a string.

    Description:
        Given list of strings, determine the number of punctuation characters in each string.
        Looks for any of the following: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    Examples:
        >>> x = ['This is a test file.', 'This is second line', 'third line: $1,000']
        >>> punctuation_count = PunctuationCount()
        >>> punctuation_count(x).tolist()
        [1.0, 0.0, 3.0]
    """
    name = "punctuation_count"
    input_types = [NaturalLanguage]
    return_type = Numeric
    description_template = "the number of punctuation characters in {}"

    def get_function(self):
        def punc_cnt(values):
            result = []
            for value in values:
                cnt = 0
                for punc in string.punctuation:
                    if punc in value:
                        cnt += 1
                result.append(cnt)
            return np.array(result)

        return punc_cnt


class Quarter(TransformPrimitive):
    """Determines the quarter of the year of a datetime

    Examples:
        >>> import pandas as pd
        >>> quarter = Quarter()
        >>> quarter([pd.to_datetime('2018-02-28'),
        ...          pd.to_datetime('2018-08-15'),
        ...          pd.to_datetime('2018-12-31'),
        ...          pd.to_datetime('2018-05-01')]).tolist()
        [1, 3, 4, 2]
    """
    name = "quarter"
    input_types = [Datetime]
    return_type = Ordinal
    description_template = "the quarter of the year of {}"

    def get_function(self):
        def quarter(values):
            result = []
            for value in values:
                month = value.month
                if 1 <= month <= 3:
                    result.append(1)
                elif 4 <= month <= 6:
                    result.append(2)
                elif 7 <= month <= 9:
                    result.append(3)
                else:
                    result.append(4)
            return np.array(result)

        return quarter


class SameAsPrevious(TransformPrimitive):
    """Determines if a value is equal to the previous value in a list.

    Description:
        Compares a value in a list to the previous value and returns True if the value is equal to the previous value or False otherwise.
        The first item in the output will always be False, since there is no previous element for the first element comparison.
        Any nan values in the input will be filled using either a forward-fill or backward-fill method, specified by the fill_method argument.
        The number of consecutive nan values that get filled can be limited with the limit argument.
        Any nan values left after filling will result in False being returned for any comparison involving the nan value.

    Args:
        fill_method (str) : Method for filling gaps in series.
            Validoptions are `backfill`, `bfill`, `pad`, `ffill`.
            `pad / ffill`: fill gap with last valid observation.
            `backfill / bfill`: fill gap with next valid observation.
            Default is `pad`.
        limit (int) : The max number of consecutive NaN values in a gap that can be filled.
            Default is None.

    Examples:
        >>> same_as_previous = SameAsPrevious()
        >>> same_as_previous([1, 2, 2, 4]).tolist()
        [False, False, True, False]

        The fill method for nan values can be specified

        >>> same_as_previous_fillna = SameAsPrevious(fill_method="bfill")
        >>> same_as_previous_fillna([1, None, 2, 4]).tolist()
        [False, False, True, False]

        The number of nan values that are filled can be limited

        >>> same_as_previous_limitfill = SameAsPrevious(limit=2)
        >>> same_as_previous_limitfill([1, None, None, None, 2, 3]).tolist()
        [False, True, True, False, False, False]
    """
    name = "same_as_previous"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "determines if a value is equal to the previous value in {}"

    def __init__(self, fill_method='pad', limit=None):
        self.fill_method = fill_method
        self.limit = limit

    def get_function(self):
        def same_as_pre(values):
            fill_values = values.fillna(method=self.fill_method, limit=self.limit)
            result = [False]
            if type(fill_values) is Series:
                fill_values = fill_values.tolist()
            for i in range(1, len(fill_values)):
                if fill_values[i-1] == fill_values[i]:
                    result.append(True)
                else:
                    result.append(False)
            return np.array(result)

        return same_as_pre


class SavgolFilter(TransformPrimitive):
    """Applies a Savitzky-Golay filter to a list of values.

    Description:
        Given a list of values, return a smoothed list which increases the signal to noise ratio without greatly distoring the signal.
        Uses the `Savitzky–Golay filter` method.
        If the input list has less than 20 values, it will be returned as is.

    Args:
        window_length (int) : The length of the filter window (i.e. the numberof coefficients).
            `window_length` must be a positive odd integer.
        polyorder (int) : The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.
        deriv (int) : Optional. The order of the derivative to compute.
            This must be a nonnegative integer.
            The default is 0, which means to filter the data without differentiating.
        delta (float) : Optional. The spacing of the samples to which the filter will be applied.
            This is only used if deriv > 0. Default is 1.0.
        mode (str) : Optional. Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
            This determines the type of extension to use for the padded signal to which the filter is applied.
            When `mode` is 'constant', the padding value is given by `cval`.
            See the Notes for more details on 'mirror', 'constant', 'wrap', and 'nearest'.
            When the 'interp' mode is selected (the default), no extensionis used.
            Instead, a degree `polyorder` polynomial is fit to the last `window_length` values of the edges, and this polynomial is used to evaluate the last `window_length // 2` output values.
        cval (scalar) : Optional. Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.

    Examples:
        >>> savgol_filter = SavgolFilter()
        >>> data = [0, 1, 1, 2, 3, 4, 5, 7, 8, 7, 9, 9, 12, 11, 12, 14, 15, 17, 17, 17, 20]
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [0.0429, 0.8286, 1.2571]

        We can control `window_length` and `polyorder` of the filter.

        >>> savgol_filter = SavgolFilter(window_length=13, polyorder=3)
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [-0.0962, 0.6484, 1.4451]

        We can control the `deriv` and `delta` parameters.

        >>> savgol_filter = SavgolFilter(deriv=1, delta=1.5)
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [0.754, 0.3492, 0.2778]

        We can use `mode` to control how edge values are handled.

        >>> savgol_filter = SavgolFilter(mode='constant', cval=5)
        >>> [round(x, 4) for x in savgol_filter(data).tolist()[:3]]
        [1.5429, 0.2286, 1.2571]
    """
    name = "savgol_filter"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "Applying Savitzky-Golay filter to {}"

    def __init__(self, window_length=5, polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.axis = axis
        self.mode = mode
        self.cval = cval

    def get_function(self):
        def sav_filter(values):
            if self.mode == "interp" and self.window_length > len(values):
                self.window_length = len(values)
                if self.window_length % 2 == 0:
                    self.window_length -= 1
                self.polyorder = self.window_length // 2
            return savgol_filter(values, self.window_length, self.polyorder, self.deriv, self.delta, self.axis, self.mode, self.cval)

        return sav_filter


class ScorePercentile(TransformPrimitive):
    """Determines the percentile of each value against an array of scores.

    Description:
        Given a list of numbers, return the approximate percentile of each number compared to a given array of scores.

    Args:
        scores (array) : Array of values to which our input values are compared.

    Examples:
        >>> percentile = ScorePercentile(scores=list(range(1, 11)))
        >>> percentile([1, 5, 10, 11, 0]).tolist()
        [10.0, 50.0, 100.0, 100.0, 0.0]
    """
    name = "score_percentile"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the percentile of {} against scores"

    def __init__(self, scores=[]):
        self.scores = scores

    def get_function(self):
        def score_percent(values):
            if len(self.scores) == 0:
                self.scores = values
            return np.array([stats.percentileofscore(self.scores, value) for value in values])

        return score_percent


class Season(TransformPrimitive):
    """Determines the season of a given datetime.

    Description:
        Given a list of datetimes, return the season of each one (`winter`, `spring`, `summer`, or `fall`).
        Uses the month of the datetime to determine the season.

    Args:
        hemisphere (str) : Specify northern or southern hemisphere.
            Could be 'northern' or 'north' or 'southern' or 'south'.
            Default is 'northern'.

    Examples:
        >>> from datetime import datetime
        >>> times = [datetime(2019, 1, 1),
        ...          datetime(2019, 3, 15),
        ...          datetime(2019, 7, 20),
        ...          datetime(2019, 12, 30)]
        >>> season = Season()
        >>> season(times).tolist()
        ['winter', 'spring', 'summer', 'winter']

        We can specify the hemisphere as well.

        >>> from datetime import datetime
        >>> season_southern = Season(hemisphere='southern')
        >>> season_southern(times).tolist()
        ['summer', 'fall', 'winter', 'summer']
    """
    name = "season"
    input_types = [Datetime]
    return_type = Categorical
    description_template = "the season of {}"

    def __init__(self, hemisphere="northern"):
        self.hemisphere = hemisphere.lower()

    def get_function(self):
        def season(values):
            result = []
            if self.hemisphere == "northern" or self.hemisphere == "north":
                for value in values:
                    month = value.month
                    if 3 <= month <= 5:
                        result.append("spring")
                    elif 6 <= month <= 8:
                        result.append("summer")
                    elif 9 <= month <= 11:
                        result.append("fall")
                    else:
                        result.append("winter")
            elif self.hemisphere == "southern" or self.hemisphere == "south":
                for value in values:
                    month = value.month
                    if 3 <= month <= 5:
                        result.append("fall")
                    elif 6 <= month <= 8:
                        result.append("winter")
                    elif 9 <= month <= 11:
                        result.append("spring")
                    else:
                        result.append("summer")
            return np.array(result)

        return season


class Sign(TransformPrimitive):
    """Determines the sign of numeric values.

    Description:
        Given a list of numbers, returns 0, 1, -1 if the number is zero, positive, or negative, respectively.
        If input value is NaN, returns NaN.

    Examples:
        >>> sign = Sign()
        >>> sign([1., -2., 3., -4., 0]).tolist()
        [1.0, -1.0, 1.0, -1.0, 0.0]
    """
    name = "sign"
    input_types = [Numeric]
    return_type = Numeric
    description_template = "the sign of {}"

    def get_function(self):
        def sign(values):
            return np.sign(values)

        return sign


class StopwordCount(TransformPrimitive):
    """Determines number of stopwords in a string.

    Description:
        Given list of strings, determine the number of stopwords characters in each string.
        Looks for any of the English stopwords defined in `nltk.corpus.stopwords`.
        Case insensitive.

    Examples:
        >>> x = ['This is a test string.', 'This is second string', 'third string']
        >>> stopword_count = StopwordCount()
        >>> stopword_count(x).tolist()
        [3.0, 2.0, 0.0]
    """
    name = "stopword_count"
    input_types = [NaturalLanguage]
    return_type = Numeric
    description_template = "the number of stopwords in {}"

    def get_function(self):
        def stop_cnt(values):
            result = []
            stop_words = stopwords.words('english')
            for words in values.str.split():
                cnt = 0
                for word in words:
                    if word.lower() in stop_words:
                        cnt += 1
                result.append(cnt)
            return np.array(result)

        return stop_cnt


class SubRegionCodeToRegion(TransformPrimitive):
    """Determines the region of a US sub-region.

    Description:
        Converts a ISO 3166-2 region code to a higher-level US region.
        Possible values include the following: `['West', 'South', 'Northeast', 'Midwest']`

    Examples:
        >>> sub_region_code_to_region = SubRegionCodeToRegion()
        >>> subregions = ["US-AL", "US-IA", "US-VT", "US-DC", "US-MI", "US-NY"]
        >>> sub_region_code_to_region(subregions).tolist()
        ['south', 'midwest', 'northeast', 'south', 'midwest', 'northeast']
    """
    name = "sub_region_code_to_region"
    input_types = [SubRegionCode]
    return_type = Categorical
    description_template = "the region of {}"

    def get_function(self):
        def sub_to_region(values):
            url = "https://raw.githubusercontent.com/cphalpert/census-regions/master/us%20census%20bureau%20regions%20and%20divisions.csv"
            data = pandas.read_csv(url)
            result = []
            for value in values:
                selected_data = data[data['State Code'] == value[-2:]]
                result.append(selected_data['Region'].to_list()[0].lower())
            return np.array(result)

        return sub_to_region


class TitleWordCount(TransformPrimitive):
    """Determines the number of title words in a string.

    Description:
        Given list of strings, determine the number of title words in each string.
        A title word is defined as any word starting with a capital letter.
        Words at the start of a sentence will be counted.

    Examples:
        >>> x = ['My favorite movie is Jaws.', 'this is a string', 'AAA']
        >>> title_word_count = TitleWordCount()
        >>> title_word_count(x).tolist()
        [2.0, 0.0, 1.0]
    """
    name = "title_word_count"
    input_types = [NaturalLanguage]
    return_type = Numeric
    description_template = "the number of title words in {}"

    def get_function(self):
        def title_word_cnt(values):
            result = []
            for words in values.str.split():
                cnt = 0
                for word in words:
                    if word[0].isupper():
                        cnt += 1
                result.append(cnt)
            return np.array(result)

        return title_word_cnt


class UpperCaseCount(TransformPrimitive):
    """Calculates the number of upper case letters in text.

    Description:
        Given a list of strings, determine the number of characters in each string that are capitalized.
        Counts every letter individually, not just every word that contains capitalized letters.

    Examples:
        >>> x = ['This IS a string.', 'This is a string', 'aaa']
        >>> upper_case_count = UpperCaseCount()
        >>> upper_case_count(x).tolist()
        [3.0, 1.0, 0.0]
    """
    name = "upper_case_count"
    input_types = [NaturalLanguage]
    return_types = Numeric
    description_template = "the number of upper case letters in {}"

    def get_function(self):
        def upper_cnt(values):
            return values.str.count(pat='[A-Z]')

        return upper_cnt


class UpperCaseWordCount(TransformPrimitive):
    """Determines the number of words in a string that are entirely capitalized.

    Description:
        Given list of strings, determine the number of words in each string that are entirely capitalized.

    Examples:
        >>> x = ['This IS a string.', 'This is a string', 'AAA']
        >>> upper_case_word_count = UpperCaseWordCount()
        >>> upper_case_word_count(x).tolist()
        [1.0, 0.0, 1.0]
    """
    name = "upper_case_word_count"
    input_types = [NaturalLanguage]
    return_type = Numeric
    description_template = "the number of words that are entirely capitalized in {}"

    def get_function(self):
        def upper_word_cnt(values):
            result = []
            for words in values.str.split():
                cnt = 0
                for word in words:
                    if word.isupper():
                        cnt += 1
                result.append(cnt)
            return np.array(result)

        return upper_word_cnt


class URLToProtocol(TransformPrimitive):
    """Determines the protocol(http or https) of a url.

    Description:
        Extract the protocol of a url using regex.
        It will be either https or http.
        Returns nan if the url doesn't contain a protocol.

    Examples:
        >>> url_to_protocol = URLToProtocol()
        >>> urls = ['https://www.google.com', 'http://www.google.co.in',
        ...         'www.facebook.com']
        >>> url_to_protocol(urls).to_list()
        ['https', 'http', nan]
    """
    name = "url_to_protocol"
    input_types = [URL]
    return_type = Categorical
    description_template = "the protocol of {}"

    def get_function(self):
        def url_to_protocol(values):
            result = []
            for value in values:
                pat = re.findall('https|http', value)
                if pat:
                    result.append(pat[0])
                else:
                    result.append(np.nan)
            return Series(result)

        return url_to_protocol


class ZIPCodeToState(TransformPrimitive):
    """Extracts the state from a ZIPCode.

    Description:
        Given a ZIPCode, return the state it's in.
        ZIPCodes can be 5-digit or 9-digit.
        In the case of 9-digit ZIPCodes, only the first 5 digits are used and any digits after the first five are discarded.
        Return nan if the ZIPCode is not found.

    Examples:
        >>> zipcode_to_state = ZIPCodeToState()
        >>> states = zipcode_to_state(['60622', '94120', '02111-1253'])
        >>> list(map(str, states))
        ['IL', 'CA', 'MA']
    """
    name = "zip_code_to_state"
    input_types = [ZIPCode]
    return_type = Categorical
    description_template = "the state from a ZIPCode {}"

    def get_function(self):
        def zip_to_state(values):
            result = []
            zipDb = ZipCodeDatabase()
            for value in values:
                result.append(zipDb[value[:5]].state)
            return np.array(result)

        return zip_to_state


class CountString(TransformPrimitive):
    """Determines how many times a given string shows up in a text field.

    Examples:
        >>> count_string = CountString(string="the")
        >>> count_string(["The problem was difficult.",
        ...               "He was there.",
        ...               "The girl went to the store."]).tolist()
        [1, 1, 2]
    """
    name = "count_string"
    input_types = [NaturalLanguage]
    return_type = Numeric

    def __init__(self, string = "", ignore_case = True, ignore_non_alphanumeric = False, is_regex = False, match_whole_words_only = False):
        self.string = string
        self.ignore_case = ignore_case
        self.ignore_non_alphanumeric = ignore_non_alphanumeric
        self.is_regex = is_regex
        self.match_whole_words_only = match_whole_words_only

    def get_function(self):
        def count_string(array):
            count = []
            for value in array:
                if self.ignore_case:
                    value = value.lower()
                    self.string = self.string.lower()
                if self.ignore_non_alphanumeric:
                    filtered = filter(str.isalnum, value)
                    value = "".join(filtered)
                if self.is_regex:
                    import re
                    temp = re.findall(self.string, value)
                    value = " ".join(temp)
                if self.match_whole_words_only:
                    count.append(sum(self.string in value))
                else: count.append(value.count(self.string))
            return pandas.Index(count)

        return count_string


class CumulativeTimeSinceLastFalse(TransformPrimitive):
    """Determines the time since last `False` value.

    Description:
        Given a list of booleans and a list of corresponding datetimes, determine the time at each point since the last `False` value.
        Returns time difference in seconds.
        `NaN` values are ignored.

    Examples:
        >>> cumulative_time_since_last_false = CumulativeTimeSinceLastFalse()
        >>> booleans = [False, True, False, True]
        >>> datetimes = [
        ...     datetime(2011, 4, 9, 10, 30, 0),
        ...     datetime(2011, 4, 9, 10, 30, 10),
        ...     datetime(2011, 4, 9, 10, 30, 15),
        ...     datetime(2011, 4, 9, 10, 30, 29)
        ... ]
        >>> cumulative_time_since_last_false(datetimes, booleans).tolist()
        [0.0, 10.0, 0.0, 14.0]
    """
    name = "cumulative_time_since_last_false"
    input_types = [DatetimeTimeIndex, Boolean]
    return_type = Numeric

    def get_function(self):
        def cumulative_time_since_last_false(datetimes, booleans):
            count = []
            last_false = 0
            for idx, val in enumerate(booleans):
                if val == False:
                    last_false = idx
                    count.append(0.0)
                else:
                    cum = datetimes[idx] - datetimes[last_false]
                    count.append(float(cum.total_seconds()))
            return pandas.Index(count)

        return cumulative_time_since_last_false


class CumulativeTimeSinceLastTrue(TransformPrimitive):
    """Determines the time (in seconds) since the last boolean was `True` given a datetime index column and boolean column

    Examples:
        >>> cumulative_time_since_last_true = CumulativeTimeSinceLastTrue()
        >>> booleans = [False, True, False, True]
        >>> datetimes = [
        ...     datetime(2011, 4, 9, 10, 30, 0),
        ...     datetime(2011, 4, 9, 10, 30, 10),
        ...     datetime(2011, 4, 9, 10, 30, 15),
        ...     datetime(2011, 4, 9, 10, 30, 30)
        ... ]
        >>> cumulative_time_since_last_true(datetimes, booleans).tolist()
        [nan, 0.0, 5.0, 0.0]
    """
    name = "cumulative_time_since_last_true"
    input_types = [DatetimeTimeIndex, Boolean]
    return_type = Numeric

    def get_function(self):
        def cumulative_time_since_last_true(datetimes, booleans):
            count = []
            last_true = 0
            for idx, val in enumerate(booleans):
                if val == True:
                    last_true = idx
                    count.append(0.0)
                else:
                    cum = datetimes[idx] - datetimes[last_true]
                    count.append(float(cum.total_seconds()))
            return pandas.Index(count)

        return cumulative_time_since_last_true


class DateToTimeZone(TransformPrimitive):
    """Determines the timezone of a datetime.

    Description:
        Given a list of datetimes, extract the timezone from each one.
        Looks for the `tzinfo` attribute on `datetime.datetime` objects.
        If the datetime has no timezone or the date is missing, return `NaN`.

    Examples:
        >>> date_to_time_zone = DateToTimeZone()
        >>> dates = [datetime(2010, 1, 1, tzinfo=timezone("America/Los_Angeles")),
        ...          datetime(2010, 1, 1, tzinfo=timezone("America/New_York")),
        ...          datetime(2010, 1, 1, tzinfo=timezone("America/Chicago")),
        ...          datetime(2010, 1, 1)]
        >>> date_to_time_zone(dates).tolist()
        ['America/Los_Angeles', 'America/New_York', 'America/Chicago', nan]
    """
    name = "date_to_time_zone"
    input_types = [Datetime]
    return_type = Categorical

    def get_function(self):
        def date_to_time_zone(dates):
            time_zone = []
            for value in dates:
                if value.tzinfo:
                    time_zone.append(str(value.tzinfo))
                else:
                    time_zone.append(None)
            return pandas.Index(time_zone)

        return date_to_time_zone


class DayName(TransformPrimitive):
    """Transforms a date into the weekday name for the date.

    Examples:
        >>> day_name = DayName()
        >>> dates = pd.Series([datetime(2016, 1, 1),
        ...          datetime(2016, 2, 27),
        ...          datetime(2017, 5, 29, 10, 30, 5),
        ...          datetime(2018, 7, 18)])
        >>> day_name(dates).tolist()
        ['Friday', 'Saturday', 'Monday', 'Wednesday']
    """
    name = "day_name"
    input_types = [Datetime]
    return_type = Categorical

    def get_function(self):
        def day_name(dates):
            days = []
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for value in dates:
                day = value.weekday()
                days.append(day_name[day])
            return pandas.Index(days)

        return day_name


class GreaterThanPrevious(TransformPrimitive):
    """Determines if a value is greater than the previous value in a list.

    Description:
        Compares a value in a list to the previous value and returns True if the value is greater than the previous value or False otherwise.
        The first item in the output will always be False, since there is no previous element for the first element comparison.
        Any nan values in the input will be filled using either a forward-fill or backward-fill method, specified by the fill_method argument.
        The number of consecutive nan values that get filled can be limited with the limit argument.
        Any nan values left after filling will result in False being returned for any comparison involving the nan value.

    Examples:
        >>> greater_than_previous = GreaterThanPrevious()
        >>> greater_than_previous([1, 2, 1, 4]).tolist()
        [False, True, False, True]
    """
    name = "greater_than_previous"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, fill_method = "pad", limit = None):
        self.fill_method = fill_method
        self.limit = limit

    def get_function(self):
        def greater_than_previous(numbers):
            df = pandas.DataFrame(numbers)
            df.fillna(method = self.fill_method, limit = self.limit)
            numbers = df[0].tolist()
            results = []
            prev = None
            for num in numbers:
                if prev is None:
                    results.append(False)
                else:
                    results.append(num > prev)
                prev = num
            return pandas.Index(results)

        return greater_than_previous


class IsFirstOccurrence(TransformPrimitive):
    """Determines whether a value is the first occurrence of the value in a list.

    Examples:
        >>> is_first_occurrence = IsFirstOccurrence()
        >>> is_first_occurrence([1, 2, 2, 3, 1]).tolist()
        [True, True, False, True, False]
    """
    name = "is_first_occurrence"
    input_types = [Discrete]
    return_type = Boolean

    def get_function(self):
        def is_first_occurrence(values):
            results = []
            for idx in range(len(values)):
                found = False
                for idx_before in range(idx):
                    if values.iloc[idx] == values.iloc[idx_before]:
                        results.append(False)
                        found = True
                        break
                if not found:
                    results.append(True)
            return pandas.Index(results)

        return is_first_occurrence


class IsLastOccurrence(TransformPrimitive):
    """Determines whether a value is the last occurrence of the value in a list.

    Examples:
        >>> is_last_occurrence = IsLastOccurrence()
        >>> is_last_occurrence([1, 2, 2, 3, 1]).tolist()
        [False, False, True, True, True]
    """
    name = "is_last_occurrence"
    input_types = [Discrete]
    return_type = Boolean

    def get_function(self):
        def is_last_occurrence(values):
            results = []
            for idx in range(len(values)):
                found = False
                for idx_after in range(idx + 1, len(values)):
                    if values.iloc[idx] == values.iloc[idx_after]:
                        results.append(False)
                        found = True
                        break
                if not found:
                    results.append(True)
            return pandas.Index(results)

        return is_last_occurrence


class IsMaxSoFar(TransformPrimitive):
    """Determines if a number in a list is larger than every value before it.

    Examples:
        >>> is_max_so_far = IsMaxSoFar()
        >>> is_max_so_far([2, 3, 5, 1, 3, 10]).tolist()
        [True, True, True, False, False, True]

    """
    name = "is_max_so_far"
    input_types = [Numeric]
    return_type = Boolean

    def get_function(self):
        def is_max_so_far(numbers):
            max_val = None
            results = []
            for val in numbers:
                if max_val is None or val >= max_val:
                    results.append(True)
                    max_val = val
                else:
                    results.append(False)
            return pandas.Index(results)

        return is_max_so_far


class IsMinSoFar(TransformPrimitive):
    """Determines if a number in a list is smaller than every value before it.

    Examples:
        >>> is_min_so_far = IsMinSoFar()
        >>> is_min_so_far([2, 3, 5, 1, 3, 10]).tolist()
        [True, False, False, True, False, False]

    """
    name = "is_min_so_far"
    input_types = [Numeric]
    return_type = Boolean

    def get_function(self):
        def is_min_so_far(numbers):
            min_val = None
            results = []
            for val in numbers:
                if min_val is None or val <= min_val:
                    results.append(True)
                    min_val = val
                else:
                    results.append(False)
            return pandas.Index(results)

        return is_min_so_far


class IsWholeNumber(TransformPrimitive):
    """Determines whether a float is a whole number.

    Description:
        Given a list of floats, determine whether each number is whole.
        If number has any non-zero decmial value, return `False`.
        If the number is missing, return `NaN`.

    Examples:
        >>> is_whole_number = IsWholeNumber()
        >>> x = [1.0, 1.1, 1.00000001, 100.0, None]
        >>> is_whole_number(x).tolist()
        [True, False, False, True, nan]
    """
    name = "is_whole_number"
    input_types = [Numeric]
    return_type = Boolean

    def get_function(self):
        def is_whole_number(numbers):
            results = []
            for val in numbers:
                if math.isnan(val):
                    results.append(None)
                elif val == int(val):
                    results.append(True)
                else:
                    results.append(False)
            return pandas.Index(results)

        return is_whole_number


class IsZero(TransformPrimitive):
    """Determines whether a number is equal to zero.

    Examples:
        >>> is_zero = IsZero()
        >>> is_zero([1, 0, 0.00, 4]).tolist()
        [False, True, True, False]
    """
    name = "is_zero"
    input_types = [Numeric]
    return_type = Boolean

    def get_function(self):
        def is_zero(numbers):
            results = []
            for val in numbers:
                if val == 0:
                    results.append(True)
                else:
                    results.append(False)
            return pandas.Index(results)

        return is_zero


class Lag(TransformPrimitive):
    """Shifts an array of values by a specified number of periods.

    Examples:
        >>> lag = Lag()
        >>> lag([1, 2, 3, 4, 5]).tolist()
        [nan, 1.0, 2.0, 3.0, 4.0]
    """
    name = "lag"
    input_types = [Variable]
    return_type = None

    def __init__(self, periods = 1, fill_value = None):
        self.periods = periods
        self.fill_value = fill_value

    def get_function(self):
        def lag(numbers):
            results = deque(numbers)
            results.rotate(self.periods)
            for i in range(self.periods):
                results[i] = None
            return pandas.Index(results)

        return lag


class LessThanPrevious(TransformPrimitive):
    """Determines if a value is less than the previous value in a list.

    Description:
    Compares a value in a list to the previous value and returns True if the value is less than the previous value or False otherwise.
    The first item in the output will always be False, since there is no previous element for the first element comparison.
    Any nan values in the input will be filled using either a forward-fill or backward-fill method, specified by the fill_method argument.
    The number of consecutive nan values that get filled can be limited with the limit argument.
    Any nan values left after filling will result in False being returned for any comparison involving the nan value.

    Examples:
        >>> less_than_previous = LessThanPrevious()
        >>> less_than_previous([1, 2, 1, 4]).tolist()
        [False, False, True, False]
    """
    name = "less_than_previous"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, fill_method = None, limit = None):
        self.fill_method = fill_method
        self.limit = limit

    def get_function(self):
        def less_than_previous(numbers):
            results = []
            prev = None
            for num in numbers:
                if prev is None:
                    results.append(False)
                else:
                    results.append(num < prev)
                prev = num
            return pandas.Index(results)

        return less_than_previous


class MeanCharactersPerWord(TransformPrimitive):
    """Determines the mean number of characters per word.

    Description:
    Given list of strings, determine the mean number of characters per word in each string.
    A word is defined as a series of any characters not separated by white space.
    Punctuation is removed before counting.
    If a string is empty or `NaN`, return `NaN`.

    Examples:
        >>> x = ['This is a test file', 'This is second line', 'third line $1,000']
        >>> mean_characters_per_word = MeanCharactersPerWord()
        >>> mean_characters_per_word(x).tolist()
        [3.0, 4.0, 5.0]
    """
    name = "mean_characters_per_word"
    input_types = [NaturalLanguage]
    return_type = Numeric

    def get_function(self):
        def mean_characters_per_word(sentences):
            count = []
            for sen in sentences:
                words = str(sen).split(" ")
                length = 0
                for word in words:
                    length += len(word)
                count.append(length/len(words))
            return pandas.Index(count)

        return mean_characters_per_word