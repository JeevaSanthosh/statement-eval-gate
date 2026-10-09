"""Generate a small dataset, read it (including OCR), extract with the stand-in
models, and check the gate blocks the damaged candidate for the right fields."""

import json

import pytest

from seg.compare import compare
from seg.extract import run_extraction
from seg.generate import generate
from seg.prepare import prepare
from seg.score import score_run


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    truths = generate(data, n=12, seed=3, scanned_share=0.25)
    prepare(data)
    return data, truths


def test_truth_is_internally_consistent(dataset):
    _, truths = dataset
    for t in truths:
        assert round(sum(h["value"] for h in t.holdings), 2) == t.closing_value
        assert (t.fees is None) == (t.layout == "D")


def test_scans_go_through_ocr_and_native_pdfs_do_not(dataset):
    data, truths = dataset
    assert any(t.scanned for t in truths) and not all(t.scanned for t in truths)
    routes = json.loads((data / "text" / "routes.json").read_text(encoding="utf-8"))
    for t in truths:
        assert set(routes[t.id]) == ({"ocr"} if t.scanned else {"text_layer"})


def test_split_table_really_repeats_a_row(dataset):
    data, truths = dataset
    t = next(t for t in truths if t.layout == "B" and not t.scanned)
    text = (data / "text" / f"{t.id}.txt").read_text(encoding="utf-8")
    assert any(text.count(h["fund_name"]) == 2 for h in t.holdings)


def test_gate_blocks_the_damaged_candidate(dataset, tmp_path):
    data, _ = dataset
    for name in ("oracle", "sloppy"):
        run_extraction(data, _prompt(tmp_path), "fake", name, tmp_path / name)
    oracle = score_run(tmp_path / "oracle", data)
    sloppy = score_run(tmp_path / "sloppy", data)

    assert oracle["macro_f1"] == 1.0
    verdicts = {r.field: r.verdict for r in compare(oracle, sloppy, resamples=2000)}
    assert verdicts["provider"] == verdicts["account_number"] == "pass"
    assert verdicts["fees"] in {"fail", "warn"}
    assert sloppy["slices"]["Kestrel Lane Wealth"]["f1"]["fees"] == 0.0


def test_command_line_compare_writes_report_and_enforces(dataset, tmp_path):
    from seg.cli import main

    data, _ = dataset
    prompt = _prompt(tmp_path)
    for name in ("oracle", "sloppy"):
        main(["--data", str(data), "extract", "--provider", "fake", "--model", name,
              "--prompt", str(prompt), "--out", str(tmp_path / name)])
    report = tmp_path / "report.md"
    with pytest.raises(SystemExit) as exit_info:
        main(["--data", str(data), "compare", str(tmp_path / "oracle"), str(tmp_path / "sloppy"),
              "--report", str(report), "--enforce"])
    assert exit_info.value.code in (0, 1)
    assert "| fees |" in report.read_text(encoding="utf-8")


def _prompt(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text("Extract the fields.", encoding="utf-8")
    return p
