"""Deterministic object-wise sensor observability estimation.

The module intentionally has no ROS dependency so geometry and policy can be
tested without a running graph.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .observation_state import ObservationState, classify_observation


@dataclass(frozen=True)
class CameraDetection:
    bbox: Sequence[float]
    class_name: str
    score: float


@dataclass
class ObservationConfig:
    lidar_min_points: int = 3
    lidar_strong_points: int = 10
    lidar_strong_density: float = 1.0
    lidar_max_range: float = 40.0
    lidar_full_score_range: float = 15.0
    lidar_roi_scale: float = 1.10
    lidar_strong_score: float = 0.55
    lidar_count_weight: float = 0.55
    lidar_density_weight: float = 0.25
    lidar_range_weight: float = 0.20

    camera_iou_threshold: float = 0.20
    camera_strong_score: float = 0.40
    camera_min_roi_pixels: float = 4.0
    camera_min_projected_depth: float = 0.10
    class_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        'vehicle': ['car', 'truck', 'bus', 'vehicle'],
        'pedestrian': ['person', 'pedestrian'],
        'cyclist': ['bicycle', 'motorcycle', 'cyclist'],
    })

    depth_min_valid_pixels: int = 20
    depth_min_m: float = 0.20
    depth_max_m: float = 20.0
    depth_sigma_m: float = 1.0
    depth_scale: float = 0.001


def _normalized(value, low, high):
    if high <= low:
        return float(value >= high)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def box_corners_3d(box):
    """Return eight corners for [x,y,z,length,width,height,yaw]."""
    x, y, z, length, width, height, yaw = map(float, box[:7])
    local = np.array([
        [sx * length * 0.5, sy * width * 0.5, sz * height * 0.5]
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    ], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    local[:, :2] = local[:, :2] @ rotation.T
    local += np.array([x, y, z], dtype=np.float32)
    return local


def transform_points(points, target_from_source):
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(target_from_source, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_box(box, camera_from_source, intrinsics, image_size,
                min_depth=0.10, min_roi_pixels=4.0):
    """Project a 3-D box and return its clipped image rectangle or None."""
    corners_camera = transform_points(box_corners_3d(box), camera_from_source)
    valid = corners_camera[:, 2] > float(min_depth)
    if not np.any(valid):
        return None
    points = corners_camera[valid]
    fx, fy, cx, cy = map(float, intrinsics)
    u = fx * points[:, 0] / points[:, 2] + cx
    v = fy * points[:, 1] / points[:, 2] + cy
    raw = np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float64)
    width, height = map(int, image_size)
    clipped = np.array([
        np.clip(raw[0], 0.0, width - 1.0),
        np.clip(raw[1], 0.0, height - 1.0),
        np.clip(raw[2], 0.0, width - 1.0),
        np.clip(raw[3], 0.0, height - 1.0),
    ])
    if (clipped[2] - clipped[0] < min_roi_pixels or
            clipped[3] - clipped[1] < min_roi_pixels):
        return None
    return tuple(float(value) for value in clipped)


class ObservationAnalyzer:
    def __init__(self, config=None):
        self.config = config or ObservationConfig()
        self._class_mapping = {
            str(key).strip().lower(): {
                str(value).strip().lower() for value in values
            }
            for key, values in self.config.class_mapping.items()
        }

    def _class_compatible(self, lidar_class, camera_class):
        lidar_name = str(lidar_class).strip().lower()
        camera_name = str(camera_class).strip().lower()
        if lidar_name == camera_name:
            return True
        compatible = self._class_mapping.get(lidar_name)
        return compatible is not None and camera_name in compatible

    def _lidar_evidence(self, box, points):
        cfg = self.config
        object_range = float(np.linalg.norm(np.asarray(box[:2], dtype=np.float32)))
        if points is None:
            return False, False, 0, 0.0, object_range, 0.0, False
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return (True, object_range <= cfg.lidar_max_range, 0, 0.0,
                    object_range, 0.0, False)

        center = np.asarray(box[:3], dtype=np.float32)
        length, width, height = np.maximum(
            np.asarray(box[3:6], dtype=np.float32) * cfg.lidar_roi_scale,
            1e-3)
        yaw = float(box[6])
        delta = points[:, :3] - center
        c, s = math.cos(yaw), math.sin(yaw)
        local_x = c * delta[:, 0] + s * delta[:, 1]
        local_y = -s * delta[:, 0] + c * delta[:, 1]
        inside = (
            (np.abs(local_x) <= length * 0.5) &
            (np.abs(local_y) <= width * 0.5) &
            (np.abs(delta[:, 2]) <= height * 0.5))
        point_count = int(np.count_nonzero(inside))
        density = float(point_count / max(length * width * height, 1e-3))

        count_score = _normalized(
            point_count, cfg.lidar_min_points, cfg.lidar_strong_points)
        density_score = _normalized(
            density, 0.0, cfg.lidar_strong_density)
        if object_range <= cfg.lidar_full_score_range:
            range_score = 1.0
        else:
            range_score = 1.0 - _normalized(
                object_range, cfg.lidar_full_score_range,
                cfg.lidar_max_range)
        if object_range > cfg.lidar_max_range:
            range_score = 0.0
        score = (
            cfg.lidar_count_weight * count_score +
            cfg.lidar_density_weight * density_score +
            cfg.lidar_range_weight * range_score)
        visible = object_range <= cfg.lidar_max_range
        supported = (
            visible and point_count >= cfg.lidar_min_points and
            score >= cfg.lidar_strong_score)
        return True, visible, point_count, density, object_range, float(score), supported

    def _camera_evidence(self, box, lidar_class, camera_detections,
                         camera_from_source, intrinsics, image_size,
                         camera_available):
        cfg = self.config
        if camera_from_source is None or intrinsics is None or image_size is None:
            return False, False, False, 0.0, 0.0, -1, None
        roi = project_box(
            box, camera_from_source, intrinsics, image_size,
            min_depth=cfg.camera_min_projected_depth,
            min_roi_pixels=cfg.camera_min_roi_pixels)
        visible = roi is not None
        if not visible or not camera_available:
            return camera_available, visible, False, 0.0, 0.0, -1, roi

        best_iou, best_score, best_index = 0.0, 0.0, -1
        for index, detection in enumerate(camera_detections or ()):
            if not self._class_compatible(lidar_class, detection.class_name):
                continue
            overlap = bbox_iou(roi, detection.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_score = float(detection.score)
                best_index = index
        iou_score = _normalized(
            best_iou, cfg.camera_iou_threshold, 1.0)
        camera_score = float(math.sqrt(max(0.0, iou_score * best_score)))
        supported = (
            best_iou >= cfg.camera_iou_threshold and
            camera_score >= cfg.camera_strong_score)
        return (camera_available, visible, supported, best_iou,
                camera_score, best_index, roi)

    def _depth_evidence(self, box, roi, depth_image, camera_from_source):
        cfg = self.config
        if depth_image is None or roi is None or camera_from_source is None:
            return False, None, None
        image = np.asarray(depth_image)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = roi
        x1 = max(0, min(width - 1, int(math.floor(x1))))
        x2 = max(0, min(width, int(math.ceil(x2))))
        y1 = max(0, min(height - 1, int(math.floor(y1))))
        y2 = max(0, min(height, int(math.ceil(y2))))
        if x2 <= x1 or y2 <= y1:
            return True, None, None
        values = image[y1:y2, x1:x2].reshape(-1).astype(np.float32)
        if image.dtype == np.uint16:
            values *= cfg.depth_scale
        values = values[
            np.isfinite(values) & (values >= cfg.depth_min_m) &
            (values <= cfg.depth_max_m)]
        if len(values) < cfg.depth_min_valid_pixels:
            return True, None, None
        camera_depth = float(np.median(values))
        center_camera = transform_points(
            np.asarray(box[:3], dtype=np.float32).reshape(1, 3),
            camera_from_source)[0]
        error = abs(camera_depth - float(center_camera[2]))
        consistency = float(math.exp(-error / max(cfg.depth_sigma_m, 1e-3)))
        return True, consistency, error

    def analyze(self, boxes, class_names, lidar_points=None,
                camera_detections: Optional[Iterable[CameraDetection]] = None,
                camera_from_source=None, intrinsics=None, image_size=None,
                camera_available=False, depth_image=None):
        results = []
        camera_detections = list(camera_detections or ())
        boxes = np.asarray(boxes, dtype=np.float32)
        for box, class_name in zip(boxes, class_names):
            lidar = self._lidar_evidence(box, lidar_points)
            (lidar_available, lidar_visible, count, density, obj_range,
             lidar_score, lidar_supported) = lidar
            camera = self._camera_evidence(
                box, class_name, camera_detections, camera_from_source,
                intrinsics, image_size, camera_available)
            (camera_data_available, camera_visible, camera_supported,
             camera_iou, camera_score, camera_index, roi) = camera
            depth_available, depth_consistency, depth_error = self._depth_evidence(
                box, roi, depth_image, camera_from_source)

            weights = []
            values = []
            if lidar_available and lidar_visible:
                weights.append(1.0)
                values.append(lidar_score)
            if camera_data_available and camera_visible:
                weights.append(1.0)
                values.append(camera_score)
            if depth_consistency is not None:
                weights.append(0.5)
                values.append(depth_consistency)
            observation_score = (
                float(np.average(values, weights=weights)) if values else 0.0)
            results.append(ObservationState(
                camera_available=camera_data_available,
                camera_visible=camera_visible,
                camera_supported=camera_supported,
                lidar_available=lidar_available,
                lidar_visible=lidar_visible,
                lidar_supported=lidar_supported,
                depth_available=depth_available,
                lidar_point_count=count,
                lidar_point_density=density,
                object_range=obj_range,
                camera_iou=camera_iou,
                camera_detection_score=(
                    float(camera_detections[camera_index].score)
                    if camera_index >= 0 else 0.0),
                camera_detection_index=camera_index,
                depth_consistency=depth_consistency,
                depth_error=depth_error,
                camera_score=camera_score,
                lidar_score=lidar_score,
                observation_score=observation_score,
                state=classify_observation(
                    lidar_supported, camera_supported),
                projected_roi=roi,
            ))
        return results
