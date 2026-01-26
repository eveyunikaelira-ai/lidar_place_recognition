"""Dataset loading utilities (TODO)."""
from typing import Iterable, Tuple


def load_scan(file_path: str):
    """Load a single LiDAR scan from disk.

    TODO: implement dataset-specific loading.
    """
    raise NotImplementedError


def iter_scans(split: str) -> Iterable[Tuple[str, str]]:
    """Iterate over scan file paths and metadata for a split.

    TODO: return (scan_path, pose_path) or similar for the given split.
    """
    raise NotImplementedError
