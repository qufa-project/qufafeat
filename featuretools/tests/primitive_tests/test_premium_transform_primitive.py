import featuretools as ft
import numpy as np
import pandas as pd

import pytest

from datetime import datetime
from pytz import timezone
from featuretools.primitives import (
    AbsoluteDiff,
    NaturalLanguageToYear,
    NthWeekOfMonth,
    PartOfDay,
    PercentChange,
    PhoneNumberToCountry,
    PolarityScore,
    PunctuationCount,
    Quarter,
    SameAsPrevious,
    SavgolFilter,
    ScorePercentile,
    Season,
    Sign,
    StopwordCount,
    SubRegionCodeToRegion,
    TitleWordCount,
    UpperCaseCount,
    UpperCaseWordCount,
    URLToProtocol,
    ZIPCodeToState,
    CountString,
    CumulativeTimeSinceLastFalse,
    CumulativeTimeSinceLastTrue,
    DateToTimeZone,
    DayName,
    GreaterThanPrevious,
    IsFirstOccurrence,
    IsLastOccurrence,
    IsMaxSoFar,
    IsMinSoFar,
    IsWholeNumber,
    IsZero,
    Lag
)


def test_AbsoluteDiff():
    absdiff = AbsoluteDiff()
    res = absdiff([3.0, -5.0, -2.4]).tolist()
    assert np.isnan(res[0])
    assert res[1:] == [ 8.0, 2.6]


def test_NaturalLanguageToYear():
    text_to_year = NaturalLanguageToYear()
    array = pd.Series(["The year was 1887.", "This string has no year", "Toy Story (1995)", "12451997abc"])
    res = text_to_year(array).tolist()
    assert res[0] == 1887
    assert np.isnan(res[1])
    assert res[2] == 1995
    assert np.isnan(res[3])


def test_NthWeekOfMonth():
    nth_week_of_month = NthWeekOfMonth()
    times = [datetime(2019, 3, 1), datetime(2019, 3, 3), datetime(2019, 3, 31), datetime(2019, 3, 30)]
    assert nth_week_of_month(times).tolist() == [1.0, 2.0, 6.0, 5.0]


def test_PartOfDay():
    part_of_day = PartOfDay()
    times = [datetime(2010, 1, 1, 1, 45, 0), datetime(2010, 1, 1, 8, 55, 15), datetime(2010, 1, 1, 16, 55, 15), datetime(2010, 1, 1, 23, 57, 30)]
    assert part_of_day(times).tolist() == ['Night', 'Morning', 'Afternoon', 'Night']

def test_PercentChange():
    percent_change = PercentChange()
    res = percent_change([2, 5, 15, 3, 3, 9, 4.5]).to_list()
    assert np.isnan(res[0])
    assert res[1:] == [1.5, 2.0, -0.8, 0.0, 2.0, -0.5]


def test_PhoneNumberToCountry():
    phone_to_country = PhoneNumberToCountry()
    assert phone_to_country(['+55 85 5555555', '+81 55-555-5555', '+1-541-754-3010',]).tolist() == ['BR', 'JP', 'US']


def test_PolarityScore():
    polarity_score = PolarityScore()
    x = ['He loves dogs', 'She hates cats', 'There is a dog', '']
    assert polarity_score(x).tolist() == [0.5719, -0.4404, 0.0, 0.0]


def test_PunctuationCount():
    punc_cnt = PunctuationCount()
    x = ['This is a test file.', 'This is second line', 'third line: $1,000']
    assert punc_cnt(x).tolist() == [1.0, 0.0, 3.0]


def test_Quarter():
    quarter = Quarter()
    times = [pd.to_datetime('2018-02-28'), pd.to_datetime('2018-08-15'), pd.to_datetime('2018-12-31'), pd.to_datetime('2018-05-01')]
    assert quarter(times).tolist() == [1, 3, 4, 2]


def test_SameAsPrevious():
    same_as_previous = SameAsPrevious()
    assert same_as_previous([1, 2, 2, 4]).tolist() == [False, False, True, False]


def test_SavgolFilter():
    sav_filter = SavgolFilter()
    data = [0, 1, 1, 2, 3, 4, 5, 7, 8, 7, 9, 9, 12, 11, 12, 14, 15, 17, 17, 17, 20]
    assert [round(x, 4) for x in sav_filter(data).tolist()[:3]] == [0.0429, 0.8286, 1.2571]


def test_ScorePercentile():
    percent = ScorePercentile(scores=list(range(1, 11)))
    assert percent([1, 5, 10, 11, 0]).tolist() == [10.0, 50.0, 100.0, 100.0, 0.0]


def test_Season():
    season = Season()
    times = [datetime(2019, 1, 1), datetime(2019, 3, 15), datetime(2019, 7, 20), datetime(2019, 12, 30)]
    assert season(times).tolist() == ['winter', 'spring', 'summer', 'winter']


def test_Sign():
    sign = Sign()
    assert sign([1., -2., 3., -4., 0]).tolist() == [1.0, -1.0, 1.0, -1.0, 0.0]


def test_StopwordCount():
    stop_cnt = StopwordCount()
    x = ['This is a test string.', 'This is second string', 'third string']
    assert stop_cnt(x).tolist() == [3.0, 2.0, 0.0]


def test_SubRegionCodeToRegion():
    sub_to_region = SubRegionCodeToRegion()
    subregions = ["US-AL", "US-IA", "US-VT", "US-DC", "US-MI", "US-NY"]
    assert sub_to_region(subregions).tolist() == ['south', 'midwest', 'northeast', 'south', 'midwest', 'northeast']


def test_TitleWordCount():
    title_word_cnt = TitleWordCount()
    x = ['My favorite movie is Jaws.', 'this is a string', 'AAA']
    assert title_word_cnt(x).tolist() == [2.0, 0.0, 1.0]


def test_UpperCaseCount():
    upper_cnt = UpperCaseCount()
    x = ['This IS a string.', 'This is a string', 'aaa']
    assert upper_cnt(x).tolist() == [3.0, 1.0, 0.0]


def test_UpperCaseWordCount():
    upper_word_cnt = UpperCaseWordCount()
    x = ['This IS a string.', 'This is a string', 'AAA']
    assert upper_word_cnt(x).tolist() == [1.0, 0.0, 1.0]


def test_URLToProtocol():
    url_to_protocol = URLToProtocol()
    urls = ['https://www.google.com', 'http://www.google.co.in', 'www.facebook.com']
    assert url_to_protocol(urls).to_list() == ['https', 'http', np.nan]


def test_ZIPCodeToState():
    zip_to_state = ZIPCodeToState()
    states = zip_to_state(['60622', '94120', '02111-1253'])
    assert list(map(str, states)) == ['IL', 'CA', 'MA']


def test_CountString():
    count_string = CountString(string="the")
    strings = ["The problem was difficult.", "He was there.", "The girl went to the store."]
    assert count_string(strings).tolist() == [1, 1, 2]


def test_CumulativeTimeSinceLastFalse():
    cumulative_time_since_last_false = CumulativeTimeSinceLastFalse()
    booleans = [False, True, False, True]
    datetimes = [
        datetime(2011, 4, 9, 10, 30, 0),
        datetime(2011, 4, 9, 10, 30, 10),
        datetime(2011, 4, 9, 10, 30, 15),
        datetime(2011, 4, 9, 10, 30, 29)
    ]
    assert cumulative_time_since_last_false(datetimes, booleans).tolist() == [0.0, 10.0, 0.0, 14.0]


def test_CumulativeTimeSinceLastTrue():
    cumulative_time_since_last_true = CumulativeTimeSinceLastTrue()
    booleans = [False, True, False, True]
    datetimes = [
        datetime(2011, 4, 9, 10, 30, 0),
        datetime(2011, 4, 9, 10, 30, 10),
        datetime(2011, 4, 9, 10, 30, 15),
        datetime(2011, 4, 9, 10, 30, 30)
    ]
    assert cumulative_time_since_last_true(datetimes, booleans).tolist() == [0.0, 0.0, 5.0, 0.0]


def test_DateToTimeZone():
    date_to_time_zone = DateToTimeZone()
    dates = [datetime(2010, 1, 1, tzinfo=timezone("America/Los_Angeles")),
             datetime(2010, 1, 1, tzinfo=timezone("America/New_York")),
             datetime(2010, 1, 1, tzinfo=timezone("America/Chicago")),
             datetime(2010, 1, 1)]
    assert date_to_time_zone(dates).tolist() == ['America/Los_Angeles', 'America/New_York', 'America/Chicago', None]


def test_DayName():
    day_name = DayName()
    dates = pd.Series([datetime(2016, 1, 1),
            datetime(2016, 2, 27),
            datetime(2017, 5, 29, 10, 30, 5),
            datetime(2018, 7, 18)])
    assert day_name(dates).tolist() == ['Friday', 'Saturday', 'Monday', 'Wednesday']


def test_GreaterThanPrevious():
    greater_than_previous = GreaterThanPrevious()
    assert greater_than_previous([1, 2, 1, 4]).tolist() == [False, True, False, True]


def test_IsFirstOccurrence():
    is_first_occurrence = IsFirstOccurrence()
    assert is_first_occurrence([1, 2, 2, 3, 1]).tolist() == [True, True, False, True, False]


def test_IsLastOccurrence():
    is_last_occurrence = IsLastOccurrence()
    assert is_last_occurrence([1, 2, 2, 3, 1]).tolist() == [False, False, True, True, True]


def test_IsMaxSoFar():
    is_max_so_far = IsMaxSoFar()
    assert is_max_so_far([2, 3, 5, 1, 3, 10]).tolist() == [True, True, True, False, False, True]


def test_IsMinSoFar():
    is_min_so_far = IsMinSoFar()
    assert is_min_so_far([2, 3, 5, 1, 3, 10]).tolist() == [True, False, False, True, False, False]


def test_IsWholeNumber():
    is_whole_number = IsWholeNumber()
    x = [1.0, 1.1, 1.00000001, 100.0, None]
    assert is_whole_number(x).tolist() == [True, False, False, True, None]


def test_IsZero():
    is_zero = IsZero()
    x = [1.0, 1.1, 1.00000001, 100.0, None]
    assert is_zero([1, 0, 0.00, 4]).tolist() == [False, True, True, False]


def test_Lag():
    lag = Lag()
    assert lag([1, 2, 3, 4, 5]).tolist() == [None, 1.0, 2.0, 3.0, 4.0]