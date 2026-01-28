"""Dataset loading utilities (TODO)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from src.config import PROCESSED_DATA, SPLITS_DIR


def load_scan(file_path: str):
    """Load a single LiDAR scan from disk.

    TODO: implement dataset-specific loading.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {file_path}")
    
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    raise ValueError(f"Unsupported scan format: {path.suffix}")


def iter_scans(split: str) -> Iterable[Tuple[str, str]]:
    """Iterate over scan file paths and metadata for a split.

    TODO: return (scan_path, pose_path) or similar for the given split.
    """
    split_path = SPLITS_DIR / f"{split}.txt"
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith('#'):
                    continue
                parts = [part for part in cleaned.replace(",", " ").split() if part]
                if len(parts) == 1:
                    yield(parts[0], "")
                else:
                    yield (parts[0], " ".join(parts[1:]))
        return
    processed_scans = sorted(PROCESSED_DATA.glob("**/*.npy"))
    for scan_path in processed_scans:
        yield (str(scan_path), "")
