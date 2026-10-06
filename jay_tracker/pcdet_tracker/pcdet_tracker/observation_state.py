"""Data structures for object-wise camera/LiDAR observability."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Tuple


class ObservationClass(str, Enum):
    MULTIMODAL_STRONG = 'MULTIMODAL_STRONG'
    LIDAR_DOMINANT = 'LIDAR_DOMINANT'
    CAMERA_DOMINANT = 'CAMERA_DOMINANT'
    WEAK_OBSERVATION = 'WEAK_OBSERVATION'


@dataclass
class ObservationState:
    """Per-object evidence without treating unavailable sensors as negatives."""

    camera_available: bool = False
    camera_visible: bool = False
    camera_supported: bool = False
    lidar_available: bool = False
    lidar_visible: bool = False
    lidar_supported: bool = False
    depth_available: bool = False

    lidar_point_count: int = 0
    lidar_point_density: float = 0.0
    object_range: float = 0.0

    camera_iou: float = 0.0
    camera_detection_score: float = 0.0
    camera_detection_index: int = -1
    depth_consistency: Optional[float] = None
    depth_error: Optional[float] = None

    camera_score: float = 0.0
    lidar_score: float = 0.0
    observation_score: float = 0.0
    state: str = ObservationClass.WEAK_OBSERVATION.value
    projected_roi: Optional[Tuple[float, float, float, float]] = None

    @property
    def camera_strong(self):
        return self.camera_supported

    @property
    def lidar_strong(self):
        return self.lidar_supported

    def to_dict(self):
        return asdict(self)


def classify_observation(lidar_strong, camera_supported):
    if lidar_strong and camera_supported:
        return ObservationClass.MULTIMODAL_STRONG.value
    if lidar_strong:
        return ObservationClass.LIDAR_DOMINANT.value
    if camera_supported:
        return ObservationClass.CAMERA_DOMINANT.value
    return ObservationClass.WEAK_OBSERVATION.value
