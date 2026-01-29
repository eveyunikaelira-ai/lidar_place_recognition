"""Visualization helpers for metrics/curves (TODO)."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _validate_curve_inputs(x: Iterable[float], y: Iterable[float], x_name: str, y_name: str) -> Tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError(f"{x_name} and {y_name} must have the same shape.")
    if x_values.size == 0:
        raise ValueError(f"{x_name} and {y_name} must be non-empty.")
    return x_values, y_values

def plot_pr_curve(precision, recall, save_path: Optional[str] = None):
    """Plot a Precision-Recall curve."""
    recall_values, precision_values = _validate_curve_inputs(
        recall,
        precision,
        x_name="recall",
        y_name="precision"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall_values, precision_values, colors="#E6D9FF", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_roc_curve(fpr, tpr, save_path: Optional[str] = None):
    """Plot an ROC curve."""
    fpr_values, tpr_values = _validate_curve_inputs(
        fpr,
        tpr,
        x_name="false positive rate"
        y_name="true positive rate",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr_values, tpr_values, color="#CFF5E7", linewidth=2, label="ROC")
    ax.plot([0, 1], [0, 1], color="#FFE5B4", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylin(0.0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
