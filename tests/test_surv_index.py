import numpy as np
import pandas as pd

from dream.surv_index import (
    _numeric_suffix_range, crowding_ratio, material_risk_tier,
    net_coverage_score,
)


def test_numeric_suffix_range_is_inclusive():
    assert _numeric_suffix_range('h22', 1, 3) == ['h22_1', 'h22_2', 'h22_3']


def test_numeric_suffix_range_pads_to_width():
    assert _numeric_suffix_range('a', 8, 10, width=2) == ['a_08', 'a_09', 'a_10']


def test_material_tiers_follow_tens_digit_convention():
    tiers = material_risk_tier(pd.Series([11, 23, 34]))
    assert tiers.tolist() == [3, 2, 1]


def test_material_value_outside_convention_is_missing():
    tiers = material_risk_tier(pd.Series([96, np.nan]))
    assert tiers.isna().all()


def test_net_coverage_follows_hv228_coding():
    scores = net_coverage_score(pd.Series([3, 0, 2, 1]))
    assert scores.tolist() == [0, 1, 2, 3]


def test_net_coverage_outside_coding_is_missing():
    assert net_coverage_score(pd.Series([7])).isna().all()


def test_crowding_is_people_per_sleeping_room():
    df = pd.DataFrame({'household_size': [6, 4], 'sleeping_rooms_count': [2, 4]})
    assert crowding_ratio(df).tolist() == [3.0, 1.0]


def test_crowding_with_zero_rooms_is_missing():
    df = pd.DataFrame({'household_size': [5], 'sleeping_rooms_count': [0]})
    assert crowding_ratio(df).isna().all()
