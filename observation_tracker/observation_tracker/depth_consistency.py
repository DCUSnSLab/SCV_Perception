"""Median aligned-depth consistency for a projected object ROI."""

from dataclasses import dataclass
import math

import numpy as np

from pcdet_tracker.observation_analyzer import box_corners_3d, transform_points

from .observation_state import DepthState


@dataclass
class DepthConsistencyConfig:
    consistency_threshold: float = 1.5
    occlusion_depth_threshold: float = 1.0
    min_valid_pixels: int = 20
    roi_inner_scale: float = 0.60
    min_support_ratio: float = 0.15
    min_depth_m: float = 0.20
    max_depth_m: float = 30.0
    depth_scale: float = 0.001


@dataclass
class DepthEvidence:
    state: DepthState
    available: bool
    error: float | None
    consistency: float | None
    median_depth: float | None
    valid_pixel_count: int = 0
    support_pixel_count: int = 0
    support_ratio: float = 0.0


class DepthConsistencyAnalyzer:
    def __init__(self, config=None):
        self.config = config or DepthConsistencyConfig()

    def analyze(self, box, roi, depth_image, camera_from_source):
        if depth_image is None or roi is None or camera_from_source is None:
            return DepthEvidence(DepthState.UNAVAILABLE, False, None, None, None)
        image = np.asarray(depth_image)
        height, width = image.shape[:2]
        scale = float(np.clip(self.config.roi_inner_scale, 0.05, 1.0))
        center_x = (float(roi[0]) + float(roi[2])) * 0.5
        center_y = (float(roi[1]) + float(roi[3])) * 0.5
        half_width = (float(roi[2]) - float(roi[0])) * scale * 0.5
        half_height = (float(roi[3]) - float(roi[1])) * scale * 0.5
        x1 = max(0, min(width - 1, int(math.floor(center_x - half_width))))
        y1 = max(0, min(height - 1, int(math.floor(center_y - half_height))))
        x2 = max(0, min(width, int(math.ceil(center_x + half_width))))
        y2 = max(0, min(height, int(math.ceil(center_y + half_height))))
        if x2 <= x1 or y2 <= y1:
            return DepthEvidence(DepthState.UNAVAILABLE, True, None, None, None)
        values = image[y1:y2, x1:x2].reshape(-1).astype(np.float32)
        if image.dtype == np.uint16:
            values *= self.config.depth_scale
        valid = values[
            np.isfinite(values) &
            (values >= self.config.min_depth_m) &
            (values <= self.config.max_depth_m)]
        if len(valid) < self.config.min_valid_pixels:
            return DepthEvidence(DepthState.UNAVAILABLE, True, None, None, None)
        corners_camera = transform_points(box_corners_3d(box), camera_from_source)
        positive_depths = corners_camera[
            corners_camera[:, 2] >= self.config.min_depth_m, 2]
        if len(positive_depths) == 0:
            return DepthEvidence(DepthState.UNAVAILABLE, True, None, None, None)
        expected = float(np.min(positive_depths))
        if expected < self.config.min_depth_m:
            return DepthEvidence(DepthState.UNAVAILABLE, True, None, None, None)

        support = valid[
            np.abs(valid - expected) <= self.config.consistency_threshold]
        support_count = int(len(support))
        support_ratio = float(support_count / len(valid))
        raw_median = float(np.median(valid))
        median = float(np.median(support)) if support_count else raw_median
        signed_error = raw_median - expected
        error = abs(signed_error)
        if (support_count >= self.config.min_valid_pixels and
                support_ratio >= self.config.min_support_ratio):
            state = DepthState.CONSISTENT
            error = abs(median - expected)
        elif signed_error <= -self.config.occlusion_depth_threshold:
            state = DepthState.OCCLUDED
        else:
            state = DepthState.INCONSISTENT
        consistency = support_ratio * math.exp(
            -error / max(self.config.consistency_threshold, 1e-3))
        return DepthEvidence(
            state, True, error, consistency, median,
            int(len(valid)), support_count, support_ratio)
