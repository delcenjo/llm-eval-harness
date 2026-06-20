import json

from .config import REPORTS_DIR


def print_summary(results):
    print(f"{'predictor':<28}{'accuracy':>10}{'macro_f1':>10}{'errors':>9}")
    print("-" * 57)
    for result in results:
        print(
            f"{result['predictor']:<28}"
            f"{result['accuracy']:>10.3f}"
            f"{result['macro_f1']:>10.3f}"
            f"{len(result['errors']):>9}"
        )


def save(results, filename="results.json"):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    path.write_text(json.dumps(results, indent=2))
    return path
