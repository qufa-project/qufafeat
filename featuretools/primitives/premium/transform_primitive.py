import warnings

import numpy as np

from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.utils import convert_time_units
from featuretools.utils.entity_utils import replace_latlong_nan
from featuretools.utils.gen_utils import Library
from featuretools.variable_types import (
    Boolean,
    DateOfBirth,
    Datetime,
    DatetimeTimeIndex,
    LatLong,
    NaturalLanguage,
    Numeric,
    Ordinal,
    Variable
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
