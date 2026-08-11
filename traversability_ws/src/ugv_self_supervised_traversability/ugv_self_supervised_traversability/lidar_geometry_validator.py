"""Future LiDAR geometry evidence interface."""

from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np


class LidarGeometryValidator(ABC):
    """Produce independent positive/negative evidence from registered point clouds."""

    @abstractmethod
    def validate(self, point_cloud: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return positive and negative evidence; unclassified space stays unknown."""
        raise NotImplementedError
