import pandas as pd
import pytest

from dream.covariates import _clear_nodata, _parse_year_month


def test_year_and_month_are_read_from_file_name():
    assert _parse_year_month('data/ndvi/2009_Jan.xlsx') == (2009, 'jan')


def test_file_name_without_month_is_rejected():
    with pytest.raises(ValueError):
        _parse_year_month('2009.xlsx')


def test_unknown_month_is_rejected():
    with pytest.raises(ValueError, match='month'):
        _parse_year_month('2009_foo.xlsx')


def test_nodata_sentinel_becomes_missing():
    df = pd.DataFrame({'RASTERVALU': ['-9999', '12.5']})
    _clear_nodata(df, 'RASTERVALU', '2009_jan.xlsx')
    assert df['RASTERVALU'].isna().tolist() == [True, False]
