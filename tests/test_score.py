from seg.score import doc_counts, f1, holding_counts, scalar_counts


def test_correct_value_is_a_true_positive():
    assert scalar_counts("closing_value", 1234.5, "1,234.50") == [1, 0, 0]


def test_wrong_value_counts_as_both_false_positive_and_false_negative():
    assert scalar_counts("closing_value", 1234.5, 1234.55) == [0, 1, 1]


def test_invented_value_where_truth_is_null_is_a_false_positive():
    assert scalar_counts("fees", None, 0.0) == [0, 1, 0]


def test_missing_value_is_a_false_negative():
    assert scalar_counts("fees", 12.0, None) == [0, 0, 1]


def test_null_for_null_scores_nothing():
    assert scalar_counts("fees", None, None) == [0, 0, 0]


def test_wrongly_formatted_date_is_wrong_not_missing():
    assert scalar_counts("statement_end", "2026-06-30", "30/06/2026") == [0, 1, 1]


def test_duplicated_holding_costs_precision_only():
    truth = [{"fund_name": "A Fund", "units": 1.0, "value": 2.0}, {"fund_name": "B Fund", "units": 3.0, "value": 4.0}]
    predicted = truth + [dict(truth[0])]
    assert holding_counts(truth, predicted) == [2, 1, 0]


def test_holding_with_wrong_value_is_a_miss_and_an_extra():
    truth = [{"fund_name": "A Fund", "units": 1.0, "value": 2.0}]
    predicted = [{"fund_name": "a  fund", "units": 1.0, "value": 2.01}]
    assert holding_counts(truth, predicted) == [0, 1, 1]


def test_empty_prediction_misses_every_field():
    truth = {"provider": "X", "fees": None, "holdings": [{"fund_name": "A", "units": 1, "value": 1}]}
    counts = doc_counts(truth, {})
    assert counts["provider"] == [0, 0, 1]
    assert counts["fees"] == [0, 0, 0]
    assert counts["holdings"] == [0, 0, 1]


def test_f1_with_nothing_to_find_and_nothing_found_is_perfect():
    assert f1(0, 0, 0) == 1.0
    assert f1(1, 1, 1) == 0.5
