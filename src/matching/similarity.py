"""Similarity measures for descriptors (TODO)."""
from __future__ import annotations

import numpy as np

def cosine_similarity(a, b):
    """Compute cosine similarity between two descriptors."""
    a = np.asarray(a, dtype= float)
    b = np.asarray(b, dtype= float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_distance(a, b):
    """Compute L2 distance between two descriptors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))
