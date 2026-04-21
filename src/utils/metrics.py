"""Evaluation metrics for Playground Series S6E4."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy(y_true, y_pred):
    """Compute balanced accuracy score.

    Args:
        y_true: True labels (string or int)
        y_pred: Predicted labels (string or int)

    Returns:
        float: Balanced accuracy score
    """
    return balanced_accuracy_score(y_true, y_pred)
