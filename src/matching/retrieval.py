"""Retrieval utilities for place recognition (TODO)."""
from __future__ import annotations

import numpy as np

from .similarity import cosine_similarity

def top_k(query_descriptor, database_descriptors, k=5):
    """Return top-k most similar descriptors in the database."""
    if k <= 0:
        return []
    scores = [
        cosine_similarity(query_descriptor, descriptor)
        for descriptor in database_descriptors
    ]
    ranked = np.argsort(scores)[::-1]
    return [(int(idx), float(scores[idx])) for idx in ranked[:k]]


def build_index(database_descriptors):
    """Build an index for fast retrieval (e.g., KD-tree, FAISS)."""
    return np.asarray(database_descriptors, dtype = float)
