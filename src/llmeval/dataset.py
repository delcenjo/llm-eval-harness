import json

from .config import DATA_PATH


def load_examples(path=DATA_PATH):
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
