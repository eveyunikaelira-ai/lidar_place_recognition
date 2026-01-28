"""Threshold sweeps and condition comparisons (TODO)."""
from __future__ import annotations

import numpy as np

from .metrics import precision_recall_f1

def run_threshold_sweep():
    """Evaluate metrics across thresholds."""
    raise NotImplementedError("Use run_threshold_sweep(y_true, y_scores) instead.")

def run_threshold_sweep_from_scores(y_true, y_scores, thresholds=None):
    """Evaluate metrics across thresholds."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    metrics = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precision, recall, f1 = precision_recall_f1(y_true, y_pred)
        metrics.append(
            "threshhold": float(threshold)
            "precision" precision,
            "recall": recall,
            "f1": f1,
        )
            
        
    return metrics
