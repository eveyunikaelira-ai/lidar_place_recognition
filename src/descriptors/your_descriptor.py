"""Custom descriptor implementation (TODO)."""
from .base import Descriptor


class YourDescriptor(Descriptor):
    """Example descriptor placeholder."""

    def __init__(self, **kwargs):
        self.params = kwargs

    def compute(self, points):
        """Compute descriptor for given points."""
        raise NotImplementedError
