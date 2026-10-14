"""Sensor and lifecycle states shared by the observation tracker."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class LidarState(str, Enum):
    NO_EVIDENCE = "NO_EVIDENCE"
    WEAK = "WEAK"
    STRONG = "STRONG"


class CameraState(str, Enum):
    AVAILABLE_SUPPORTED = "AVAILABLE_SUPPORTED"
    AVAILABLE_UNSUPPORTED = "AVAILABLE_UNSUPPORTED"
    UNAVAILABLE = "UNAVAILABLE"


class DepthState(str, Enum):
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    OCCLUDED = "OCCLUDED"
    UNAVAILABLE = "UNAVAILABLE"


class ObservationClass(str, Enum):
    MULTIMODAL_STRONG = "MULTIMODAL_STRONG"
    LIDAR_DOMINANT = "LIDAR_DOMINANT"
    CAMERA_DOMINANT = "CAMERA_DOMINANT"
    WEAK_OBSERVATION = "WEAK_OBSERVATION"


class TrackStatus(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    RECOVERING = "RECOVERING"
    OCCLUDED = "OCCLUDED"
    SUPPRESSED = "SUPPRESSED"
    LOST = "LOST"
    TERMINATED = "TERMINATED"


@dataclass
class ObservationResult:
    """Object-wise evidence kept separate from detector confidence."""

    lidar_state: LidarState = LidarState.NO_EVIDENCE
    lidar_point_count: int = 0
    lidar_point_density: float = 0.0
    lidar_centroid_error: Optional[float] = None
    lidar_score: float = 0.0
    lidar_available: bool = False
    lidar_visible: bool = False
    lidar_supported: bool = False

    camera_state: CameraState = CameraState.UNAVAILABLE
    camera_available: bool = False
    camera_visible: bool = False
    camera_supported: bool = False
    camera_iou: float = 0.0
    camera_center_distance: Optional[float] = None
    camera_class_match: bool = False
    camera_detection_score: float = 0.0
    camera_detection_index: int = -1
    camera_detection_class: str = ""
    camera_score: float = 0.0

    depth_state: DepthState = DepthState.UNAVAILABLE
    depth_available: bool = False
    depth_error: Optional[float] = None
    depth_consistency: Optional[float] = None
    depth_valid_pixel_count: int = 0
    depth_support_pixel_count: int = 0
    depth_support_ratio: float = 0.0
    camera_depth_center: Optional[Tuple[float, float, float]] = None
    camera_depth_overlap: float = 0.0

    object_range: float = 0.0
    state: str = ObservationClass.WEAK_OBSERVATION.value
    raw_state: str = ObservationClass.WEAK_OBSERVATION.value
    observation_score: float = 0.0
    projected_roi: Optional[Tuple[float, float, float, float]] = None

    @property
    def observation_state(self):
        return self.state

    @property
    def lidar_strong(self):
        return self.lidar_state == LidarState.STRONG

    @property
    def camera_strong(self):
        return self.camera_state == CameraState.AVAILABLE_SUPPORTED
