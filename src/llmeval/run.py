import os

from dotenv import load_dotenv

from .dataset import load_examples
from .harness import evaluate
from .predictors import KeywordBaseline
from .report import print_summary, save


def build_predictors():
    predictors = [KeywordBaseline()]
    if os.getenv("ANTHROPIC_API_KEY"):
        from .predictors import ClaudeClassifier

        predictors.append(ClaudeClassifier())
    else:
        print("(ANTHROPIC_API_KEY not set — evaluating the offline baseline only)\n")
    return predictors


def main():
    load_dotenv()
    examples = load_examples()
    results = [evaluate(predictor, examples) for predictor in build_predictors()]
    print_summary(results)
    print(f"\nSaved detailed report -> {save(results)}")


if __name__ == "__main__":
    main()
