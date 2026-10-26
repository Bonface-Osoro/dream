import numpy as np
import pandas as pd
import pytest

from dream.survGeo import (
    RISK_CATEGORIES, _calculate_wilson_interval, _first_survey_year,
    _month_number, category_edges, derive_risk_category,
)


def test_first_survey_year_reads_year_before_dash():
    assert _first_survey_year('2014-15') == 2014


def test_first_survey_year_accepts_plain_year():
    assert _first_survey_year('2009') == 2009


def test_first_survey_year_rejects_bad_label():
    with pytest.raises(ValueError):
        _first_survey_year('14-15')


def test_month_number_accepts_any_case_and_sept():
    months = pd.Series(['Jan', 'SEPT', 'dec'])
    assert _month_number(months).tolist() == [1, 9, 12]


def test_month_number_names_unknown_month():
    with pytest.raises(ValueError, match='xyz'):
        _month_number(pd.Series(['jan', 'xyz']))


def _scored_table():
    scores = pd.Series([0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95])
    categories = pd.Series(np.repeat(RISK_CATEGORIES, 2))
    return scores, categories


def test_category_edges_sit_halfway_between_categories():
    scores, categories = _scored_table()
    assert category_edges(scores, categories) == pytest.approx([0.3, 0.6, 0.85])


def test_category_edges_reject_overlapping_categories():
    scores, categories = _scored_table()
    scores.iloc[2] = 0.15
    with pytest.raises(ValueError, match='overlap'):
        category_edges(scores, categories)


def test_category_edges_reject_missing_category():
    scores, categories = _scored_table()
    with pytest.raises(ValueError, match='lacks'):
        category_edges(scores[:6], categories[:6])


def test_derived_categories_reproduce_the_original_ones():
    scores, categories = _scored_table()
    edges = category_edges(scores, categories)
    derived = derive_risk_category(scores, edges)
    assert derived.astype(str).tolist() == categories.tolist()


def test_wilson_interval_with_no_observations_is_nan():
    lower, upper = _calculate_wilson_interval(0, 0)
    assert np.isnan(lower) and np.isnan(upper)


def test_wilson_interval_stays_within_zero_and_one():
    lower, upper = _calculate_wilson_interval(10, 10)
    assert 0.0 <= lower < upper <= 1.0


def test_wilson_interval_contains_the_observed_proportion():
    lower, upper = _calculate_wilson_interval(30, 100)
    assert lower < 0.3 < upper
