from .config import LABELS
from .metrics import accuracy, confusion_matrix, macro_f1, per_class_metrics


def evaluate(predictor, examples, labels=LABELS):
    y_true = [example["label"] for example in examples]
    y_pred = [predictor.predict(example["text"]) for example in examples]
    return {
        "predictor": predictor.name,
        "n": len(examples),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class": per_class_metrics(y_true, y_pred, labels),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels)},
        "errors": [
            {"text": example["text"], "true": true, "pred": pred}
            for example, true, pred in zip(examples, y_true, y_pred)
            if true != pred
        ],
    }
