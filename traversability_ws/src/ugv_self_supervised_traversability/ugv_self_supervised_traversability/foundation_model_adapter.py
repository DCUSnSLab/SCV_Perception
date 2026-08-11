"""Future interface for expanding positive seeds with SAM-like models."""

from abc import ABC, abstractmethod
import numpy as np


class FoundationModelAdapter(ABC):
    """Keep model-specific dependencies outside the positive label generator."""

    @abstractmethod
    def expand(self, image: np.ndarray, positive_seed: np.ndarray) -> np.ndarray:
        """Return an expanded mask without changing UNKNOWN into negative evidence."""
        raise NotImplementedError
