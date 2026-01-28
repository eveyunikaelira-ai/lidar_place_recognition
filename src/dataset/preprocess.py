"""Preprocessing utilities for LiDAR scans (TODO)."""
from __future__ import annotations

import numpy as np

def filter_points(points):
    """Filter points by range, height, or other criteria."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Points array must be of shape (N, 3+).")
    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = distances > np.finfo(float).eps
    return points[mask]


def downsample(points):
    """Downsample a point cloud (e.g., voxel grid)."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    step = max(1, len(points) // 2048)
    return points[::step]


def normalize(points):
    """Normalize points (e.g., centering/scaling)."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    xyz = points[:, :3]
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    std[std == 0] = 1.0
    normalized_xyz = (xyz - mean) / std
    if points.shape[1] > 3:
        return np.concatenate([normalized_xyz, points[:, 3:]], axis=1)
    return normalized_xyz

