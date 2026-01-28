"""Descriptor interface definitions (TODO)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

class Descriptor(ABC):
    """Abstract descriptor interface."""

    @abstractmethod
    def compute(self, points: ArrayLike) -> np.ndarray:
        """Compute a descriptor from a point cloud."""
        raise NotImplementedError
    
    def __call__(self, points: ArrayLike) -> np.ndarray:
        return self.compute(points)
    
    @staticmethod
    def ensure_numpy(points: ArrayLike) -> np.ndarray:
        return np.asarray(points, dtype=float)
    
    @staticmethod
    def validate_points(points: np.ndarray) -> None:
        if points is None or len(points) == 0:
            raise ValueError("Points array is empty.")
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("Points array must be of shape (N, 3+).")
        
    def compute_checked(self, points: ArrayLike) -> np.ndarray:
        points_array = self.ensure_numpy(points)
        self.validate_points(points_array)
        return self.compute(points_array)
    
    def metadata(self) -> dict[str, Any]:
        return{}
