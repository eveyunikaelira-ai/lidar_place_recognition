"""Descriptor interface definitions (TODO)."""
from abc import ABC, abstractmethod


class Descriptor(ABC):
    """Abstract descriptor interface."""

    @abstractmethod
    def compute(self, points):
        """Compute a descriptor from a point cloud."""
        raise NotImplementedError
