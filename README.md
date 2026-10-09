# statement-eval-gate

An evaluation harness and CI release gate for LLM extraction from financial
statements. Every target field is scored separately, and a pull request that
makes any field significantly worse cannot merge.

Everything runs on local models through [Ollama](https://ollama.com). CI never
calls a model: it replays recorded responses, so the gate is free and
deterministic.

## Why per-field

A single "accuracy" number for document extraction hides where the failures
are. If the provider name is right on every statement and the fees are wrong on
a third of them, a document-level score still looks healthy. Scoring each field
on its own shows which one is failing, on which layout, and whether a change
helped or hurt it.

## How it works

```
generate ──► prepare ──► extract ──────────► score ──► compare / gate
synthetic    text layer   Ollama, recorded    per-field   paired bootstrap,
statements   or OCR       to cassettes/       F1          blocks the PR
+ answers                 (replayed in CI)
```

| Step | What it does |
|---|---|
| `seg generate` | Builds 60 statements from four fictional providers, with the right answer for each. About a fifth are rendered as noisy scans with no text layer. |
| `seg prepare` | Reads the text layer, or runs Tesseract OCR on pages that have none, and saves the text. The text is committed so every model request is reproducible. |
| `seg extract` | Sends each statement to the model with a JSON schema for structured output. In `record` mode the response is saved under a hash of the full request; in `replay` mode it is read back. |
| `seg score` | Per-field precision, recall and F1, broken down by provider and by scanned vs native. Optional MLflow logging. |
| `seg gate` | Replays the candidate in `eval.toml`, scores it and the baseline with the same code, and fails if any field drops significantly. |

### The statements

All providers, funds and account numbers are invented. Each provider's layout
carries traps that real statements set for extraction pipelines:

| Provider | Trap |
|---|---|
| Brackenford Investments | An extra price column next to units and value |
| Tidewell Platform | The holdings table breaks across pages and repeats its last row after the break; payments, withdrawals and charges appear only in a summary at the end |
| Ashcombe Pensions | Opening and closing values are in a sentence; individual transactions are listed and the totals sit in a separate summary |
| Kestrel Lane Wealth | No fees figure at all, so the right answer is null; withdrawals printed in brackets; ordinal dates |
| Scanned (any provider) | Low resolution, rotation, noise, blur and JPEG artefacts. Tesseract misreads about one amount in ten, e.g. `£3,280.84` read as `£3,280.64`, or `£` read as `€` |

### Scoring

| Prediction vs truth | Counts as |
|---|---|
| Correct value | true positive |
| Wrong value | false positive **and** false negative |
| A value where the truth is null (an invented figure) | false positive |
| Null where the truth has a value | false negative |
| Holding listed twice | one true positive, one false positive |

Values are normalised before comparison (case, spacing, `£` and commas, two
decimal places), so formatting never costs a point. Dates must be ISO
`YYYY-MM-DD`, as the schema asks.

### The gate

Both runs saw the same statements, so the comparison is paired: each of 10,000
bootstrap resamples draws the same statements for baseline and candidate, and
records the change in each field's F1.

| Verdict | Rule |
|---|---|
| fail | F1 dropped by more than `max_drop` and the 95% interval for the change is wholly below zero |
| warn | F1 dropped by more than `max_drop`, but the interval still includes zero, so this sample cannot separate the drop from noise |
| better | F1 rose by more than `max_drop` and the interval is wholly above zero |
| pass | anything else |

The baseline always comes from the target branch, never from the pull request,
so a PR cannot pass by editing its own baseline.

## Results

_To be filled in from the first recorded runs: models compared, prompt v1 vs v2,
per-field F1 with intervals, the slice that fails worst, latency per statement._

## Running it

You need Python 3.11+ and [Ollama](https://ollama.com). Tesseract and Poppler
are only needed to regenerate the statements or redo the OCR, because the
extracted text is committed. Works the same on Windows, macOS and Linux.

```bash
pip install -e ".[dev,mlflow]"
ollama pull qwen3.5:4b

seg extract --limit 3                  # try three statements first: check they say "ok", and how long each takes
seg extract                            # record the rest; prints one line per statement
seg promote runs/extract_v1__ollama-qwen3.5-4b
seg score results/baseline --mlflow    # per-field F1, logged to MLflow
```

Recording is incremental: stop it half way and run it again, and it picks up
where it left off.

If the first statements report `invalid response`, the model is ignoring the
JSON schema. Some Ollama versions do this for Qwen3.5 with thinking switched off
([ollama#14645](https://github.com/ollama/ollama/issues/14645)); update Ollama,
or remove `think = false` from `eval.toml`.

### Changing the prompt or the model

1. Edit the prompt in `prompts/`, or the `[candidate]` model in `eval.toml`.
2. `seg extract --mode record` to record the new responses.
3. `seg gate` to see the per-field report locally.
4. `seg promote runs/<run>` and commit the prompt, `eval.toml`, `cassettes/` and `results/baseline/`.
5. Open a pull request. CI replays the recordings and posts the report to the job summary.

If you change the prompt or model and forget step 2, CI stops with an error
naming the statement that has no recording.

### Without a model

`--provider fake` swaps in a stand-in that returns the ground truth (`oracle`)
or damages it in known ways (`sloppy`). It exists to test the harness. Its
results are never reported as model results.

```bash
seg extract --provider fake --model oracle --out runs/oracle
seg extract --provider fake --model sloppy --out runs/sloppy
seg compare runs/oracle runs/sloppy
```

### Regenerating the data

```bash
# macOS: brew install tesseract poppler    Ubuntu: apt install tesseract-ocr poppler-utils
seg generate --n 60 --seed 7
seg prepare
```

## Limitations

- The statements are synthetic. The traps are modelled on real failure modes,
  but real statements are messier.
- 60 statements give wide intervals. A one-statement regression shows as a
  warning, not a failure. That is deliberate: the gate only blocks changes it
  can show are real.
- Matching is exact after normalisation. A fund name with one OCR error counts
  as wrong.

## Next

- An adversarial set: instructions hidden in statement text, totals that do not
  reconcile, pages out of order.
- Guardrails: a reconciliation check (holdings sum to the closing value) that
  rejects an extraction before it is used.
- RAGAS and DeepEval run alongside per-field F1, with a write-up of where they
  disagree.
- Change-based selection: re-run only the evaluation slices a change can affect.

## Licence

MIT
