"""Command line: seg generate | prepare | extract | score | compare | gate | promote"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

from .compare import compare, report
from .extract import run_extraction
from .llm import CassetteMiss
from .score import log_to_mlflow, score_run

BASELINE_FILES = ["predictions.jsonl", "run.json", "scores.json"]


def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {"candidate": {}, "gate": {}}


def cmd_generate(a):
    from .generate import generate

    truths = generate(a.data, n=a.n, seed=a.seed)
    print(f"Generated {len(truths)} statements in {a.data}, {sum(t.scanned for t in truths)} of them scanned.")


def cmd_prepare(a):
    from .prepare import prepare

    routes = prepare(a.data)
    ocr = sum("ocr" in r for r in routes.values())
    print(f"Wrote text for {len(routes)} statements to {a.data / 'text'}; {ocr} needed OCR.")


def cmd_extract(a):
    cand = load_config(a.config)["candidate"]
    prompt = a.prompt or Path(cand["prompt"])
    provider = a.provider or cand.get("provider", "ollama")
    model = a.model or cand["model"]
    out = a.out or Path("runs") / f"{prompt.stem}__{provider}-{model.replace(':', '-').replace('/', '-')}"
    try:
        run_extraction(a.data, prompt, provider, model, out, mode=a.mode, cassette_dir=a.cassettes, limit=a.limit,
                       think=cand.get("think"), verbose=True)
    except CassetteMiss as exc:
        sys.exit(f"error: {exc}")
    print(f"Predictions written to {out}")


def cmd_score(a):
    scores = score_run(a.run, a.data)
    print(f"{a.run}: macro F1 {scores['macro_f1']:.3f}")
    for field, v in scores["fields"].items():
        print(f"  {field:<16} F1 {v['f1']:.3f}   (tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
    if a.mlflow:
        log_to_mlflow(a.run, scores)
        print("Logged to MLflow experiment 'statement-eval-gate'.")


def _compare_and_report(baseline: dict, candidate: dict, gate: dict, out: Path | None) -> int:
    max_drop = gate.get("max_drop", 0.02)
    confidence = gate.get("confidence", 0.95)
    results = compare(baseline, candidate, max_drop, confidence, gate.get("resamples", 10_000))
    text = report(baseline, candidate, results, max_drop, confidence)
    print(text)
    if out:
        out.write_text(text, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)
    return 1 if any(r.verdict == "fail" for r in results) else 0


def cmd_compare(a):
    baseline = score_run(a.baseline, a.data)
    candidate = score_run(a.candidate, a.data)
    code = _compare_and_report(baseline, candidate, load_config(a.config)["gate"], a.report)
    sys.exit(code if a.enforce else 0)


def cmd_gate(a):
    """Replay the configured candidate, score it and the baseline with the same code, and compare."""
    config = load_config(a.config)
    if not (a.baseline / "predictions.jsonl").exists():
        print(f"No baseline at {a.baseline}; nothing to compare against yet. Gate skipped.")
        return
    cand = config["candidate"]
    out = Path("runs") / "candidate"
    try:
        run_extraction(a.data, Path(cand["prompt"]), cand.get("provider", "ollama"), cand["model"], out,
                       mode="replay", cassette_dir=a.cassettes, think=cand.get("think"))
    except CassetteMiss as exc:
        sys.exit(f"error: {exc}")
    baseline = score_run(a.baseline, a.data)
    candidate = score_run(out, a.data)
    sys.exit(_compare_and_report(baseline, candidate, config["gate"], out / "report.md"))


def cmd_promote(a):
    score_run(a.run, a.data)
    a.to.mkdir(parents=True, exist_ok=True)
    for name in BASELINE_FILES:
        # copyfile, not copy2: keeping the source's old timestamp can make git miss the change
        # when the new file is the same size as the one it replaces.
        shutil.copyfile(a.run / name, a.to / name)
    run = json.loads((a.run / "run.json").read_text(encoding="utf-8"))
    print(f"Baseline is now {run['model']} with {run['prompt']} ({a.to}). Commit it with your change.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="seg", description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--config", type=Path, default=Path("eval.toml"))
    p.add_argument("--cassettes", type=Path, default=Path("cassettes"))
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create synthetic statements and their answers")
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)

    sub.add_parser("prepare", help="extract text from each PDF, using OCR for scans").set_defaults(func=cmd_prepare)

    e = sub.add_parser("extract", help="run the model over every statement")
    e.add_argument("--prompt", type=Path)
    e.add_argument("--provider", choices=["ollama", "fake"])
    e.add_argument("--model", help="for ollama: e.g. qwen3.5:4b; for fake: oracle or sloppy")
    e.add_argument("--mode", choices=["record", "replay"], default="record")
    e.add_argument("--out", type=Path)
    e.add_argument("--limit", type=int)
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("score", help="per-field F1 for a run")
    s.add_argument("run", type=Path)
    s.add_argument("--mlflow", action="store_true", help="also log the scores to MLflow")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="paired bootstrap comparison of two runs")
    c.add_argument("baseline", type=Path)
    c.add_argument("candidate", type=Path)
    c.add_argument("--report", type=Path)
    c.add_argument("--enforce", action="store_true", help="exit 1 if any field fails")
    c.set_defaults(func=cmd_compare)

    gt = sub.add_parser("gate", help="replay the candidate from eval.toml and compare it with the baseline")
    gt.add_argument("--baseline", type=Path, default=Path("results/baseline"))
    gt.set_defaults(func=cmd_gate)

    pr = sub.add_parser("promote", help="make a run the new baseline")
    pr.add_argument("run", type=Path)
    pr.add_argument("--to", type=Path, default=Path("results/baseline"))
    pr.set_defaults(func=cmd_promote)

    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the report has ✅/❌; a Windows console or pipe may default to cp1252
    a.func(a)


if __name__ == "__main__":
    main()
