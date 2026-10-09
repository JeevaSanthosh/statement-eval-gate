"""Per-field scoring.

Each field gets its own true positive, false positive and false negative counts
on every document, so a failing field cannot hide inside one overall accuracy
number.

Scalar fields, per document:
  correct value                      -> TP
  value given but wrong              -> FP and FN (a wrong answer is both)
  value given where the truth is null -> FP (an invented figure)
  null where the truth has a value   -> FN

Holdings are scored as a multiset of (fund, units, value). A fund listed twice
earns one TP and one FP, so duplicated rows cost precision.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .schema import ALL_FIELDS, SCALAR_FIELDS, norm_holding, norm_scalar


def f1(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def scalar_counts(field: str, truth_value, predicted_value) -> list[int]:
    t = norm_scalar(field, truth_value)
    p = norm_scalar(field, predicted_value)
    if p is None and predicted_value not in (None, ""):
        p = ("unparseable", str(predicted_value))  # an answer in the wrong format is a wrong answer
    if t is None and p is None:
        return [0, 0, 0]
    if t is None:
        return [0, 1, 0]
    if p is None:
        return [0, 0, 1]
    return [1, 0, 0] if t == p else [0, 1, 1]


def holding_counts(truth: list[dict], predicted: list[dict]) -> list[int]:
    t = Counter(norm_holding(h) for h in truth or [])
    p = Counter(norm_holding(h) for h in predicted or [])
    tp = sum((t & p).values())
    return [tp, sum(p.values()) - tp, sum(t.values()) - tp]


def doc_counts(truth: dict, predicted: dict) -> dict[str, list[int]]:
    counts = {f: scalar_counts(f, truth.get(f), predicted.get(f)) for f in SCALAR_FIELDS}
    counts["holdings"] = holding_counts(truth.get("holdings"), predicted.get("holdings"))
    return counts


def field_summary(per_doc: dict[str, dict[str, list[int]]], ids: list[str]) -> dict[str, dict]:
    out = {}
    for field in ALL_FIELDS:
        tp, fp, fn = (int(x) for x in np.sum([per_doc[i][field] for i in ids], axis=0))
        out[field] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "f1": round(f1(tp, fp, fn), 4),
        }
    return out


def macro_f1(fields: dict[str, dict]) -> float:
    return round(float(np.mean([v["f1"] for v in fields.values()])), 4)


def score_run(run_dir: Path, data_dir: Path) -> dict:
    manifest = {e["id"]: e for e in json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    per_doc, failures = {}, {f: [] for f in ALL_FIELDS}
    for row in rows:
        truth = json.loads((data_dir / "truth" / f"{row['id']}.json").read_text(encoding="utf-8"))
        counts = doc_counts(truth, row["prediction"])
        per_doc[row["id"]] = counts
        for field, (tp, fp, fn) in counts.items():
            if fp or fn:
                if field == "holdings":
                    detail = {"missed": fn, "extra": fp}
                else:
                    detail = {"truth": truth.get(field), "predicted": row["prediction"].get(field)}
                failures[field].append({"id": row["id"], **detail})

    ids = list(per_doc)
    fields = field_summary(per_doc, ids)

    slices = {}
    for name, members in _slices(ids, manifest).items():
        s = field_summary(per_doc, members)
        slices[name] = {"documents": len(members), "macro_f1": macro_f1(s), "f1": {k: v["f1"] for k, v in s.items()}}

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    scores = {
        "run": json.loads((run_dir / "run.json").read_text(encoding="utf-8")),
        "macro_f1": macro_f1(fields),
        "fields": fields,
        "slices": slices,
        "ops": {
            "documents": len(rows),
            "invalid_responses": sum(1 for r in rows if r["error"]),
            "latency_ms_p50": int(np.percentile(latencies, 50)) if latencies else None,
            "latency_ms_p95": int(np.percentile(latencies, 95)) if latencies else None,
            "mean_completion_tokens": _mean([r.get("completion_tokens") for r in rows]),
        },
        "documents": per_doc,
        "failures": failures,
    }
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=1) + "\n", encoding="utf-8")
    return scores


def _slices(ids: list[str], manifest: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(manifest[i]["provider"], []).append(i)
        groups.setdefault("scanned" if manifest[i]["scanned"] else "native PDF", []).append(i)
    return dict(sorted(groups.items()))


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 1) if values else None


def log_to_mlflow(run_dir: Path, scores: dict) -> None:
    import mlflow  # tracks to MLflow's default local store, or to MLFLOW_TRACKING_URI if set

    run = scores["run"]
    mlflow.set_experiment("statement-eval-gate")
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params(
            {
                "provider": run["provider"],
                "model": run["model"],
                "prompt": run["prompt"],
                "prompt_sha256": run["prompt_sha256"][:12],
                "documents": run["documents"],
            }
        )
        metrics = {f"f1_{k}": v["f1"] for k, v in scores["fields"].items()}
        metrics["macro_f1"] = scores["macro_f1"]
        metrics["invalid_responses"] = scores["ops"]["invalid_responses"]
        if scores["ops"]["latency_ms_p50"] is not None:
            metrics["latency_ms_p50"] = scores["ops"]["latency_ms_p50"]
            metrics["latency_ms_p95"] = scores["ops"]["latency_ms_p95"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(run_dir / "scores.json"))
