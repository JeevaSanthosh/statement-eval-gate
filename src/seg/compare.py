"""Compare two runs field by field and decide whether the candidate may ship.

The comparison is paired: both runs saw the same statements, so each bootstrap
resample draws the same documents for both and the per-field F1 difference is
measured on identical inputs. That removes the noise of which statements
happened to be in the sample, and gives a confidence interval for the change.

Verdict per field:
  fail    F1 fell by more than max_drop and the interval sits wholly below zero
  warn    F1 fell by more than max_drop but the interval still includes zero,
          so this sample cannot tell a real drop from noise
  better  F1 rose by more than max_drop and the interval sits wholly above zero
  pass    anything else
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import ALL_FIELDS


@dataclass
class FieldResult:
    field: str
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str


def _f1(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = 2 * tp + fp + fn
    return np.where(denom == 0, 1.0, 2 * tp / np.maximum(denom, 1))


def compare(
    baseline: dict,
    candidate: dict,
    max_drop: float = 0.02,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> list[FieldResult]:
    ids = sorted(baseline["documents"])
    if ids != sorted(candidate["documents"]):
        raise ValueError(
            "The two runs scored different statements, so they cannot be compared pairwise. "
            "Re-run the baseline on the current dataset."
        )

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(resamples, len(ids)))
    tail = (1 - confidence) / 2 * 100

    results = []
    for field in ALL_FIELDS:
        b = np.array([baseline["documents"][i][field] for i in ids], dtype=float)
        c = np.array([candidate["documents"][i][field] for i in ids], dtype=float)
        deltas = _f1(c[idx].sum(axis=1)) - _f1(b[idx].sum(axis=1))
        base_f1, cand_f1 = float(_f1(b.sum(axis=0))), float(_f1(c.sum(axis=0)))
        delta = cand_f1 - base_f1
        low, high = (float(x) for x in np.percentile(deltas, [tail, 100 - tail]))

        if delta < -max_drop and high < 0:
            verdict = "fail"
        elif delta < -max_drop:
            verdict = "warn"
        elif delta > max_drop and low > 0:
            verdict = "better"
        else:
            verdict = "pass"
        results.append(FieldResult(field, base_f1, cand_f1, delta, low, high, verdict))
    return results


ICON = {"fail": "❌ fail", "warn": "⚠️ warn", "better": "✅ better", "pass": "pass"}


def report(baseline: dict, candidate: dict, results: list[FieldResult], max_drop: float, confidence: float) -> str:
    b_run, c_run = baseline["run"], candidate["run"]
    failed = [r.field for r in results if r.verdict == "fail"]
    lines = [
        "## Statement extraction: evaluation gate",
        "",
        f"**{'BLOCKED' if failed else 'PASSED'}**"
        + (f" — significant drop in: {', '.join(failed)}" if failed else ""),
        "",
        f"Baseline: `{b_run['model']}` with `{b_run['prompt']}` · "
        f"Candidate: `{c_run['model']}` with `{c_run['prompt']}` · "
        f"{candidate['ops']['documents']} statements",
        "",
        f"| Field | Baseline F1 | Candidate F1 | Change | {int(confidence * 100)}% interval | Verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.field} | {r.baseline:.3f} | {r.candidate:.3f} | {r.delta:+.3f} | "
            f"{r.ci_low:+.3f} to {r.ci_high:+.3f} | {ICON[r.verdict]} |"
        )
    lines += [
        f"| **macro F1** | {baseline['macro_f1']:.3f} | {candidate['macro_f1']:.3f} | "
        f"{candidate['macro_f1'] - baseline['macro_f1']:+.3f} | | |",
        "",
        f"A field fails when its F1 drops by more than {max_drop:.2f} and the paired bootstrap interval "
        "for the change lies wholly below zero. A warning means the drop is real in this sample but the "
        "sample is too small to rule out noise.",
        "",
        "### Candidate by slice",
        "",
        "| Slice | Docs | Macro F1 | " + " | ".join(ALL_FIELDS) + " |",
        "|---|---:|---:|" + "---:|" * len(ALL_FIELDS),
    ]
    for name, s in candidate["slices"].items():
        lines.append(
            f"| {name} | {s['documents']} | {s['macro_f1']:.3f} | "
            + " | ".join(f"{s['f1'][f]:.2f}" for f in ALL_FIELDS)
            + " |"
        )
    ops_b, ops_c = baseline["ops"], candidate["ops"]
    lines += [
        "",
        "### Operations",
        "",
        "| | Baseline | Candidate |",
        "|---|---:|---:|",
        f"| Invalid responses | {ops_b['invalid_responses']} | {ops_c['invalid_responses']} |",
        f"| Latency p50 (ms) | {ops_b['latency_ms_p50'] or '–'} | {ops_c['latency_ms_p50'] or '–'} |",
        f"| Latency p95 (ms) | {ops_b['latency_ms_p95'] or '–'} | {ops_c['latency_ms_p95'] or '–'} |",
        f"| Mean output tokens | {ops_b['mean_completion_tokens'] or '–'} | {ops_c['mean_completion_tokens'] or '–'} |",
    ]
    return "\n".join(lines) + "\n"
