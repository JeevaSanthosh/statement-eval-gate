import pytest

from seg.compare import compare
from seg.schema import ALL_FIELDS


def scores(per_doc_counts: dict[str, dict[str, list[int]]]) -> dict:
    return {"documents": per_doc_counts}


def uniform(n: int, field_counts: dict[str, list[int]]) -> dict:
    default = {f: [1, 0, 0] for f in ALL_FIELDS}
    return scores({f"d{i}": {**default, **field_counts} for i in range(n)})


def verdicts(results):
    return {r.field: r.verdict for r in results}


def test_identical_runs_pass_with_zero_width_interval():
    run = uniform(30, {})
    results = compare(run, run, resamples=500)
    assert set(verdicts(results).values()) == {"pass"}
    assert all(r.ci_low == r.ci_high == 0 for r in results)


def test_consistent_drop_fails():
    baseline = uniform(40, {})
    candidate = uniform(40, {"fees": [0, 1, 1]})
    assert verdicts(compare(baseline, candidate, resamples=500))["fees"] == "fail"


def test_one_bad_document_in_a_small_sample_only_warns():
    baseline = uniform(10, {})
    candidate = scores({**baseline["documents"], "d0": {**baseline["documents"]["d0"], "fees": [0, 1, 1]}})
    result = verdicts(compare(baseline, candidate, resamples=2000))
    assert result["fees"] == "warn"


def test_improvement_is_reported():
    baseline = uniform(40, {"holdings": [3, 1, 0]})
    candidate = uniform(40, {"holdings": [3, 0, 0]})
    assert verdicts(compare(baseline, candidate, resamples=500))["holdings"] == "better"


def test_runs_on_different_statements_are_refused():
    with pytest.raises(ValueError, match="different statements"):
        compare(uniform(5, {}), uniform(6, {}), resamples=10)
