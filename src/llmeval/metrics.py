from collections import Counter


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def precision_recall_f1(y_true, y_pred, label):
    true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def per_class_metrics(y_true, y_pred, labels):
    support = Counter(y_true)
    metrics = {}
    for label in labels:
        precision, recall, f1 = precision_recall_f1(y_true, y_pred, label)
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.get(label, 0),
        }
    return metrics


def macro_f1(y_true, y_pred, labels):
    f1_scores = [precision_recall_f1(y_true, y_pred, label)[2] for label in labels]
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def confusion_matrix(y_true, y_pred, labels):
    position = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for true, pred in zip(y_true, y_pred):
        if true in position and pred in position:
            matrix[position[true]][position[pred]] += 1
    return matrix
