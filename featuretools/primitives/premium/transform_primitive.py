import warnings

import numpy as np

import string
import pandas
import re

from pyzipcode import ZipCodeDatabase
from pandas import Series
from scipy.signal import savgol_filter
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
        [3.0, 5.0, 2.4]
    """
    name = "absolute_diff"
    input_types = [Numeric]
    return_type = Numeric
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the absolute diff of {}"

    def get_function(self):
        def func_absdiff(values):
            return np.insert(np.absolute(np.diff(values)), 0, float('nan'))
            # return convert_time_units(values.diff().apply(lambda x: x.total_seconds()), self.unit)
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
            return savgol_filter(values, self.window_length, self.polyorder, self.deriv, self.delta, self.axis, self.mode, self.cval)

        return sav_filter


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
