"""Lightweight point-count and centroid evidence inside oriented 3-D boxes."""

from dataclasses import dataclass, field
import math

import numpy as np

from .observation_state import LidarState


DEFAULT_THRESHOLDS = {
    "vehicle": {"near": 12, "middle": 8, "far": 4},
    "pedestrian": {"near": 6, "middle": 4, "far": 2},
    "cyclist": {"near": 8, "middle": 5, "far": 3},
    "unknown": {"near": 10, "middle": 6, "far": 3},
}


@dataclass
class LidarObservationConfig:
    near_range: float = 15.0
    middle_range: float = 30.0
    far_range: float = 50.0
    weak_min_points: int = 2
    centroid_distance_threshold: float = 1.5
    margin_xy: float = 0.25
    margin_z: float = 0.15
    thresholds: dict = field(default_factory=lambda: {
        name: dict(values) for name, values in DEFAULT_THRESHOLDS.items()})


@dataclass
class LidarEvidence:
    state: LidarState
    count: int
    density: float
    centroid_error: float | None
    object_range: float
    score: float
    available: bool
    visible: bool


def _class_key(class_name):
    value = str(class_name).strip().lower()
    if value in {"car", "truck", "bus", "vehicle"}:
        return "vehicle"
    if value in {"person", "pedestrian"}:
        return "pedestrian"
    if value in {"bicycle", "motorcycle", "cyclist"}:
        return "cyclist"
    return "unknown"


class LidarObservationAnalyzer:
    def __init__(self, config=None):
        self.config = config or LidarObservationConfig()

    def _range_band(self, distance):
        if distance <= self.config.near_range:
            return "near"
        if distance <= self.config.middle_range:
            return "middle"
        return "far"

    def min_points(self, class_name, distance):
        values = self.config.thresholds.get(
            _class_key(class_name), self.config.thresholds["unknown"])
        return max(1, int(values[self._range_band(distance)]))

    def analyze(self, box, class_name, points):
        box = np.asarray(box, dtype=np.float32)
        distance = float(np.linalg.norm(box[:2]))
        visible = distance <= self.config.far_range
        if points is None:
            return LidarEvidence(
                LidarState.NO_EVIDENCE, 0, 0.0, None, distance, 0.0,
                False, visible)
        points = np.asarray(points, dtype=np.float32)
        if not visible or points.size == 0:
            return LidarEvidence(
                LidarState.NO_EVIDENCE, 0, 0.0, None, distance, 0.0,
                True, visible)

        center = box[:3]
        length = max(1e-3, float(box[3]) + 2.0 * self.config.margin_xy)
        width = max(1e-3, float(box[4]) + 2.0 * self.config.margin_xy)
        height = max(1e-3, float(box[5]) + 2.0 * self.config.margin_z)
        # Reject obviously unrelated points before rotating them.  The XY
        # radius circumscribes the oriented box, so this preserves the exact
        # set of points accepted by the test below.
        radius = math.hypot(length, width) * 0.5
        candidate = (
            (np.abs(points[:, 0] - center[0]) <= radius) &
            (np.abs(points[:, 1] - center[1]) <= radius) &
            (np.abs(points[:, 2] - center[2]) <= height * 0.5))
        candidates = points[candidate, :3]
        delta = candidates - center
        cosine, sine = math.cos(float(box[6])), math.sin(float(box[6]))
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        inside = (
            (np.abs(local_x) <= length * 0.5) &
            (np.abs(local_y) <= width * 0.5))
        selected = candidates[inside]
        count = int(len(selected))
        density = float(count / max(length * width * height, 1e-3))
        centroid_error = (
            float(np.linalg.norm(selected.mean(axis=0) - center))
            if count else None)
        strong_points = self.min_points(class_name, distance)
        centroid_ok = (
            centroid_error is not None and
            centroid_error <= self.config.centroid_distance_threshold)
        if count >= strong_points and centroid_ok:
            state = LidarState.STRONG
        elif count >= self.config.weak_min_points:
            state = LidarState.WEAK
        else:
            state = LidarState.NO_EVIDENCE
        count_score = min(1.0, count / max(strong_points, 1))
        centroid_score = (
            max(0.0, 1.0 - centroid_error /
                max(self.config.centroid_distance_threshold, 1e-3))
            if centroid_error is not None else 0.0)
        score = 0.75 * count_score + 0.25 * centroid_score
        return LidarEvidence(
            state, count, density, centroid_error, distance, float(score),
            True, visible)
