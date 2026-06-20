from llmeval.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1


def test_accuracy():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3


def test_precision_recall_f1_perfect():
    assert precision_recall_f1(["a", "a"], ["a", "a"], "a") == (1.0, 1.0, 1.0)


def test_precision_recall_f1_partial():
    precision, recall, f1 = precision_recall_f1(["a", "a", "b"], ["a", "b", "b"], "a")
    assert precision == 1.0
    assert recall == 0.5
    assert round(f1, 3) == 0.667


def test_macro_f1_averages_labels():
    score = macro_f1(["a", "b"], ["a", "b"], ["a", "b"])
    assert score == 1.0


def test_confusion_matrix():
    matrix = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert matrix == [[1, 1], [0, 1]]
