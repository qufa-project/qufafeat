import featuretools as ft
import numpy as np
import pandas as pd

import pytest

from datetime import datetime
from featuretools.primitives import (
    AbsoluteDiff,
    PhoneNumberToCountry,
    PolarityScore,
    PunctuationCount,
    Quarter,
    SavgolFilter,
    Season,
    Sign,
    StopwordCount,
    SubRegionCodeToRegion,
    TitleWordCount,
    UpperCaseCount,
    UpperCaseWordCount,
    URLToProtocol,
    ZIPCodeToState
)


def test_AbsoluteDiff():
    absdiff = AbsoluteDiff()
    res = absdiff([3.0, -5.0, -2.4]).tolist()
    assert np.isnan(res[0])
    assert res[1:] == [ 8.0, 2.6]


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


def test_SavgolFilter():
    sav_filter = SavgolFilter()
    data = [0, 1, 1, 2, 3, 4, 5, 7, 8, 7, 9, 9, 12, 11, 12, 14, 15, 17, 17, 17, 20]
    assert [round(x, 4) for x in sav_filter(data).tolist()[:3]] == [0.0429, 0.8286, 1.2571]


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