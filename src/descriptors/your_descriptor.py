"""Custom descriptor implementation (TODO)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from .base import Descriptor

class YourDescriptor(Descriptor):
    """Example descriptor placeholder."""

    def __init__(
            self,
            num_bins: int = 16,
            max_distance: float | None = None,
            include_stats: bool = True,
    ):
        self.num_bins = num_bins
        self.max_distance = max_distance
        self.include_stats = include_stats

    def compute(self, points):
        """Compute descriptor for given points."""
        if points is None or len(points) == 0:
            raise ValueError("Points array is empty.")
        
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shapes[1] < 3:
            raise ValueError("Points array must be of shape (N, 3+).*")
        
        xyz = points[:, :3]
        distances = np.linalg.norm(xyz, axis=1)
        max_distance = (
            float(np.max(distances)) if self.max_distance is None else self.max_distance
        )
        max_distance = max(max_distance, np.finfo(float).eps)
        hist, _ = np.histogram(distances, bins=self.num_bins, range=(0.0, max_distance))
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        features: Sequence[np.ndarray] = [hist]
        if self.include_stats:
            means = xyz.mean(axis=0)
            stds = xyz.std(axis=0)
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
            stats = np.concatenate([means, stds, mins, maxs])
            features = [hist, stats]

        return np.concatenate(features)
