from llmeval.config import LABELS
from llmeval.dataset import load_examples


def test_dataset_is_loaded_and_labelled():
    examples = load_examples()
    assert len(examples) >= 40
    assert all(example["label"] in LABELS for example in examples)
    assert all(example["text"].strip() for example in examples)
