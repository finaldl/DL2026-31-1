"""Metrics used by the frozen-feature evaluation pipeline."""

from __future__ import annotations

import numpy as np


def _validated_labels(y_true, y_pred):
    """Return flat, non-empty label arrays with matching lengths."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if y_true.size != y_pred.size:
        raise ValueError(
            f"Label lengths do not match: y_true={y_true.size}, "
            f"y_pred={y_pred.size}"
        )
    return y_true, y_pred


def top1_accuracy(y_true, y_pred):
    """Return top-1 accuracy in the inclusive range [0, 1]."""
    y_true, y_pred = _validated_labels(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred):
    """Return unweighted mean F1 across labels present in either input."""
    y_true, y_pred = _validated_labels(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for label in classes:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def per_class_accuracy(y_true, y_pred, classes=None):
    """Return ``{class_label: accuracy}`` for classes present in y_true."""
    y_true, y_pred = _validated_labels(y_true, y_pred)
    if classes is None:
        classes = np.unique(y_true)
    result = {}
    for label in classes:
        mask = y_true == label
        if mask.any():
            result[label] = float(np.mean(y_pred[mask] == label))
    return result


def confusion_matrix(y_true, y_pred, classes=None):
    """Return a confusion matrix and its corresponding class order."""
    y_true, y_pred = _validated_labels(y_true, y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    classes = list(classes)
    class_to_index = {label: index for index, label in enumerate(classes)}
    unknown = set(np.unique(np.concatenate([y_true, y_pred]))) - set(classes)
    if unknown:
        raise ValueError(f"classes does not include labels: {sorted(unknown)!r}")
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[class_to_index[true_label], class_to_index[predicted_label]] += 1
    return matrix, classes


def random_baseline(n_classes):
    """Return the expected accuracy of uniform random guessing."""
    if not isinstance(n_classes, (int, np.integer)) or n_classes <= 0:
        raise ValueError("n_classes must be a positive integer")
    return 1.0 / int(n_classes)
