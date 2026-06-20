# LLM Evaluation Harness

A small, dependency-light harness for **evaluating and comparing classifiers**
on a labelled dataset — built for the common task of measuring how well an LLM
prompt actually performs before shipping it.

It ships with two interchangeable predictors:

- a **rule-based baseline** that runs completely offline, and
- a **Claude classifier** that you enable by setting an API key.

Both are scored with the same metrics, so you can quantify exactly how much the
LLM improves over a simple baseline.

## What it does

```
labelled data ─▶ predictor.predict(text) ─▶ metrics ─▶ comparison report
```

1. Load a JSONL dataset of support messages labelled with their intent.
2. Run each predictor over every example.
3. Compute accuracy, macro-F1, per-class precision/recall/F1 and a confusion
   matrix — all implemented from scratch in `metrics.py`.
4. Print a comparison table and save a detailed JSON report (including every
   misclassified example).

## Dataset

`data/support_intents.jsonl` — 50 customer-support messages labelled across five
intents: `billing`, `technical`, `account`, `cancellation`, `other`.

## Project structure

```
src/llmeval/
  config.py      paths, labels, model name
  dataset.py     JSONL loader
  metrics.py     accuracy, precision/recall/F1, macro-F1, confusion matrix
  predictors.py  KeywordBaseline and ClaudeClassifier
  harness.py     run a predictor and compute its metrics
  report.py      comparison table and JSON output
  run.py         CLI entry point
tests/           metrics, dataset and predictor tests
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m llmeval.run     # evaluates the baseline (and Claude if a key is set)
pytest
```

## Results

The offline baseline on the 50-example set:

| Predictor          | Accuracy | Macro-F1 | Errors |
| ------------------ | -------- | -------- | ------ |
| keyword-baseline   | 0.840    | 0.853    | 8 / 50 |

The detailed report shows where it breaks down: because unmatched messages fall
back to `other`, that class collects most of the errors — precisely the kind of
weakness an LLM classifier is expected to fix. With `ANTHROPIC_API_KEY` set, the
Claude classifier is evaluated alongside the baseline in the same table.

## Possible improvements

- An LLM-as-judge evaluator for free-text answers (not just classification).
- Multiple prompt variants compared side by side.
- Cross-validation and confidence intervals on the metrics.
