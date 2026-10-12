import json
from pathlib import Path

import pytest

from seg.ground import ground_holdings, holding_rows

TEXT = """Your investments
Fund Units Price Value
Meridian Japan Equity Acc GBP 1,234.567 4.4000 £5,432.10
Meridian Japan Equity C Acc 2,000.000 1.0000 £2,000.00
Orwell UK Gilt Tracker Inc 50.125 £100.25
Saltmarsh Japan Equity Inc §, 5,512.379 £9,000.00
12 January 2026 Monthly contribution £100.00
"""


def holding(name, units=None):
    return {"holdings": [{"fund_name": name, "units": units, "value": 1.0}]}


def grounded(name, units=None):
    prediction, _ = ground_holdings(holding(name, units), TEXT)
    return prediction["holdings"][0]


def test_rows_are_read_with_their_unit_counts():
    assert holding_rows(TEXT) == [
        ("Meridian Japan Equity Acc GBP", 1234.567),
        ("Meridian Japan Equity C Acc", 2000.0),
        ("Orwell UK Gilt Tracker Inc", 50.125),
        ("Saltmarsh Japan Equity Inc", 5512.379),  # OCR debris before the number is dropped
    ]


def test_dropped_ending_is_restored():
    assert grounded("Meridian Japan Equity Acc")["fund_name"] == "Meridian Japan Equity Acc GBP"


def test_added_ending_is_removed():
    assert grounded("Orwell UK Gilt Tracker Inc GBP")["fund_name"] == "Orwell UK Gilt Tracker Inc"


def test_exact_name_is_kept_even_when_another_row_extends_it():
    text = "Wrekin US Equity Tracker Acc 1.500 £1.00\nWrekin US Equity Tracker Acc GBP 2.500 £2.00"
    prediction, changes = ground_holdings(holding("Wrekin US Equity Tracker Acc"), text)
    assert prediction["holdings"][0]["fund_name"] == "Wrekin US Equity Tracker Acc" and changes == []


def test_ambiguous_name_is_left_alone():
    assert grounded("Meridian Japan Equity")["fund_name"] == "Meridian Japan Equity"


def test_part_of_a_word_is_not_a_match():
    assert grounded("Orwell UK Gilt Track")["fund_name"] == "Orwell UK Gilt Track"


def test_unmatched_name_is_left_alone():
    assert grounded("Calder Global Property Acc")["fund_name"] == "Calder Global Property Acc"


@pytest.mark.parametrize("model_units", [1234.57, 1234.6, 1235.0])
def test_rounded_units_get_their_decimals_back(model_units):
    assert grounded("Meridian Japan Equity Acc GBP", model_units)["units"] == 1234.567


@pytest.mark.parametrize("model_units", [1243.567, 1.234567, None])
def test_units_that_are_not_a_rounding_are_left_alone(model_units):
    assert grounded("Meridian Japan Equity Acc GBP", model_units)["units"] == model_units


def test_changes_are_reported():
    _, changes = ground_holdings(holding("Meridian Japan Equity Acc", 1234.57), TEXT)
    assert changes == [
        "fund_name: 'Meridian Japan Equity Acc' -> 'Meridian Japan Equity Acc GBP'",
        "units of 'Meridian Japan Equity Acc GBP': 1234.57 -> 1234.567",
    ]


def test_an_empty_or_invalid_prediction_passes_through():
    assert ground_holdings({}, TEXT) == ({}, [])


def test_the_right_answers_are_never_changed():
    """Grounding the ground truth must be a no-op on every committed statement, scans included."""
    data = Path(__file__).parent.parent / "data"
    for truth_file in sorted((data / "truth").glob("*.json")):
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        text = (data / "text" / f"{truth_file.stem}.txt").read_text(encoding="utf-8")
        assert ground_holdings(truth, text) == (truth, []), truth_file.stem
