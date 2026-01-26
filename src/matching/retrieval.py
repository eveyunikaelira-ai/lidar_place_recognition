"""Retrieval utilities for place recognition (TODO)."""


def top_k(query_descriptor, database_descriptors, k=5):
    """Return top-k most similar descriptors in the database."""
    raise NotImplementedError


def build_index(database_descriptors):
    """Build an index for fast retrieval (e.g., KD-tree, FAISS)."""
    raise NotImplementedError
