"""I/O helper functions for saving/loading data (TODO)."""

import json
from pathlib import Path

import numpy as np


def save_json(path, data):
    """Save dict-like data to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, sort_keys=True)


def load_json(path):
    """Load JSON data from disk."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_npz(path, **arrays):
    """Save arrays to NPZ."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(target), **arrays)
