"""Evaluation metrics for place recognition (TODO)."""


def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, and F1-score."""
    raise NotImplementedError


def average_precision(y_true, y_scores):
    """Compute average precision (AP)."""
    raise NotImplementedError


def pr_curve(y_true, y_scores):
    """Compute points for a PR curve."""
    raise NotImplementedError


def roc_curve(y_true, y_scores):
    """Compute points for an ROC curve."""
    raise NotImplementedError
