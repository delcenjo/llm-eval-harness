# LLM Evaluation Harness

![CI](https://github.com/delcenjo/llm-eval-harness/actions/workflows/ci.yml/badge.svg)

Before you reach for an LLM to classify text, it is worth asking whether it
actually earns its place. A handful of keyword rules is free, instant, and easy
to reason about. This project gives you a way to answer the question with
numbers instead of a hunch: on a labelled dataset, does an LLM classifier
beat a plain keyword baseline, and by how much?

The setup is deliberately small. There is one task (routing customer-support
messages to an intent), two predictors that expose the same `predict(text)`
method, and one set of metrics applied to both. The baseline runs offline. The
LLM classifier runs only if you provide an API key. Whatever predictors are
available get scored side by side in the same table.

## Reading the result

Running the harness prints a row per predictor:

```
predictor                     accuracy  macro_f1   errors
---------------------------------------------------------
keyword-baseline                 0.840     0.853    8 / 50
```

That is the baseline alone, on the 50-example dataset. Accuracy is the share of
messages it got right; macro-F1 averages the per-class F1 scores so a rare
intent counts as much as a common one; `errors` is the raw miss count.

If you dig into the saved JSON report, the failure pattern is the interesting
part. Anything the rules do not match falls through to `other`, so that bucket
quietly absorbs most of the mistakes. That is exactly the soft spot you would
hope an LLM classifier closes, since it can read intent from messages that
share no obvious keyword. Set an API key and its row appears next to the
baseline, scored on the same examples, so the comparison is direct.

## How it fits together

The flow is a straight line:

```
labelled data -> predictor.predict(text) -> metrics -> comparison report
```

A predictor is just any object with a `name` and a `predict(text)` that returns
one of the labels. Two ship in `predictors.py`:

- `KeywordBaseline` matches each message against per-intent keyword lists and
  defaults to `other` when nothing hits. No network, no dependencies.
- `LLMClassifier` sends the message to an LLM with a short prompt asking for a
  single category word, then maps the reply back onto the known labels. It needs
  an API key to run.

The labels are the five support intents: `billing`, `technical`, `account`,
`cancellation`, and `other`. The dataset lives in
`data/support_intents.jsonl`, 50 messages each tagged with one of them.

Everything in `metrics.py` is written from scratch rather than pulled from a
library: accuracy, per-class precision/recall/F1, macro-F1, and a confusion
matrix. There is no scikit-learn dependency to install, and the math is short
enough to read in one sitting.

The rest of the modules are small connectors:

| File            | What it holds                                            |
| --------------- | -------------------------------------------------------- |
| `config.py`     | paths, the label list, the model name                    |
| `dataset.py`    | the JSONL loader                                         |
| `harness.py`    | runs a predictor over the data and computes its metrics  |
| `report.py`     | prints the comparison table and writes the JSON report   |
| `run.py`        | the command-line entry point                             |

Tests under `tests/` cover the metric functions, the dataset loader, and the
keyword baseline's behaviour on a few clear cases.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m llmeval.run
pytest
```

With no key set, `python -m llmeval.run` evaluates the baseline and says so. To
add the LLM classifier, copy `.env.example` to `.env` and fill in your API key
(or export it in the shell). The run then scores both predictors and saves a
detailed report to `reports/results.json`, including every misclassified
message so you can see precisely where each predictor went wrong.

## Where this could go

The same shape extends naturally to a few things that are not built yet: an
LLM-as-judge step for grading free-text answers rather than fixed categories,
comparing several prompt variants in one run, or putting confidence intervals
around the numbers so small differences between predictors are not read as real.
