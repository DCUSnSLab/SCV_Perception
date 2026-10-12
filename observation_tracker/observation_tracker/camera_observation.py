"""Project 3-D regions and match them to existing 2-D detections."""

from dataclasses import dataclass, field
import math

from pcdet_tracker.observation_analyzer import bbox_iou, project_box

from .observation_state import CameraState


@dataclass(frozen=True)
class CameraDetection:
    bbox: tuple
    class_name: str
    score: float


@dataclass
class CameraObservationConfig:
    iou_threshold: float = 0.20
    center_distance_threshold: float = 100.0
    detector_threshold: float = 0.25
    min_roi_pixels: float = 4.0
    min_projected_depth: float = 0.10
    class_mapping: dict = field(default_factory=lambda: {
        "vehicle": ["car", "truck", "bus", "vehicle"],
        "pedestrian": ["person", "pedestrian"],
        "cyclist": ["bicycle", "motorcycle", "cyclist"],
    })


@dataclass
class CameraEvidence:
    state: CameraState
    available: bool
    visible: bool
    supported: bool
    iou: float
    center_distance: float | None
    class_match: bool
    detection_score: float
    detection_index: int
    detection_class: str
    score: float
    roi: tuple | None


class CameraObservationAnalyzer:
    def __init__(self, config=None):
        self.config = config or CameraObservationConfig()
        self.mapping = {
            str(key).lower(): {str(value).lower() for value in values}
            for key, values in self.config.class_mapping.items()}

    def class_compatible(self, lidar_class, camera_class):
        left, right = str(lidar_class).lower(), str(camera_class).lower()
        if left == right:
            return True
        if left in {"car", "truck", "bus"}:
            left = "vehicle"
        elif left == "person":
            left = "pedestrian"
        elif left in {"bicycle", "motorcycle"}:
            left = "cyclist"
        return right in self.mapping.get(left, set())

    def analyze(self, box, class_name, detections, camera_from_source,
                intrinsics, image_size, data_available):
        if camera_from_source is None or intrinsics is None or image_size is None:
            return CameraEvidence(
                CameraState.UNAVAILABLE, False, False, False, 0.0, None,
                False, 0.0, -1, "", 0.0, None)
        roi = project_box(
            box, camera_from_source, intrinsics, image_size,
            min_depth=self.config.min_projected_depth,
            min_roi_pixels=self.config.min_roi_pixels)
        if roi is None or not data_available:
            return CameraEvidence(
                CameraState.UNAVAILABLE, False, roi is not None, False, 0.0,
                None, False, 0.0, -1, "", 0.0, roi)

        roi_center = ((roi[0] + roi[2]) * 0.5, (roi[1] + roi[3]) * 0.5)
        best = None
        for index, detection in enumerate(detections or ()):
            if float(detection.score) < self.config.detector_threshold:
                continue
            compatible = self.class_compatible(class_name, detection.class_name)
            if not compatible:
                continue
            overlap = bbox_iou(roi, detection.bbox)
            det_center = (
                (detection.bbox[0] + detection.bbox[2]) * 0.5,
                (detection.bbox[1] + detection.bbox[3]) * 0.5)
            center_distance = math.hypot(
                roi_center[0] - det_center[0], roi_center[1] - det_center[1])
            center_score = max(
                0.0, 1.0 - center_distance /
                max(self.config.center_distance_threshold, 1e-3))
            score = 0.65 * overlap + 0.20 * center_score + 0.15 * float(detection.score)
            if best is None or score > best[0]:
                best = (score, overlap, center_distance, index, detection)
        if best is None:
            return CameraEvidence(
                CameraState.AVAILABLE_UNSUPPORTED, True, True, False, 0.0,
                None, False, 0.0, -1, "", 0.0, roi)
        score, overlap, center_distance, index, detection = best
        supported = (
            overlap >= self.config.iou_threshold and
            center_distance <= self.config.center_distance_threshold)
        state = (
            CameraState.AVAILABLE_SUPPORTED if supported
            else CameraState.AVAILABLE_UNSUPPORTED)
        return CameraEvidence(
            state, True, True, supported, float(overlap),
            float(center_distance), True, float(detection.score), index,
            str(detection.class_name), float(score), roi)
