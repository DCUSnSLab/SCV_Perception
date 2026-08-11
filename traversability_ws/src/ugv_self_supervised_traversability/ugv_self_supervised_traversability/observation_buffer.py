"""Bounded timestamped sensor-observation buffer for delayed supervision."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

import numpy as np


@dataclass
class Observation:
    """A camera observation frozen with the poses valid at capture time."""

    timestamp_ns: int
    frame_id: str
    rgb: np.ndarray
    depth_m: Optional[np.ndarray]
    intrinsic: np.ndarray
    world_t_camera: np.ndarray
    robot_pose: Dict[str, float]
    label: np.ndarray
    source_trajectory_timestamps_ns: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservationBuffer:
    """Time- and frame-bounded FIFO buffer."""

    def __init__(self, duration_seconds: float, max_frames: int) -> None:
        if duration_seconds <= 0.0 or max_frames <= 0:
            raise ValueError('buffer limits must be positive')
        self.duration_ns = int(duration_seconds * 1e9)
        self.max_frames = max_frames
        self._items: deque[Observation] = deque()

    def append(self, observation: Observation) -> List[Observation]:
        """Append an observation and return entries evicted by frame count."""
        evicted: List[Observation] = []
        self._items.append(observation)
        while len(self._items) > self.max_frames:
            evicted.append(self._items.popleft())
        return evicted

    def pop_expired(self, now_ns: int) -> List[Observation]:
        """Remove entries older than the configured delayed-labeling horizon."""
        expired: List[Observation] = []
        threshold = now_ns - self.duration_ns
        while self._items and self._items[0].timestamp_ns < threshold:
            expired.append(self._items.popleft())
        return expired

    def observations_before(self, timestamp_ns: int) -> List[Observation]:
        """Return snapshots that may receive evidence from a future traversal."""
        return [item for item in self._items if item.timestamp_ns <= timestamp_ns]

    def flush(self) -> List[Observation]:
        """Remove and return all buffered observations."""
        result = list(self._items)
        self._items.clear()
        return result

    def __len__(self) -> int:
        return len(self._items)
