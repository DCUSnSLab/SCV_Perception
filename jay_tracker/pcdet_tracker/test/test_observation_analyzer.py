import math

import numpy as np

from pcdet_tracker.observation_analyzer import (
    CameraDetection, ObservationAnalyzer, ObservationConfig, bbox_iou,
    project_box)
from pcdet_tracker.observation_state import ObservationClass


def camera_from_lidar_identity_axes():
    # LiDAR x-forward/y-left/z-up -> optical z-forward/x-right/y-down.
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    return matrix


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_projection_distinguishes_fov_from_camera_support():
    transform = camera_from_lidar_identity_axes()
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0])
    roi = project_box(box, transform, (500, 500, 320, 240), (640, 480))
    assert roi is not None
    assert roi[0] < 320 < roi[2]

    analyzer = ObservationAnalyzer()
    state = analyzer.analyze(
        [box], ['Vehicle'], lidar_points=None, camera_detections=[],
        camera_from_source=transform, intrinsics=(500, 500, 320, 240),
        image_size=(640, 480), camera_available=False)[0]
    assert state.camera_visible
    assert not state.camera_available
    assert not state.camera_supported


def test_multimodal_strong_state_and_point_count():
    config = ObservationConfig(
        lidar_min_points=3, lidar_strong_points=6,
        lidar_strong_density=0.2, lidar_strong_score=0.4,
        camera_iou_threshold=0.1, camera_strong_score=0.2)
    analyzer = ObservationAnalyzer(config)
    transform = camera_from_lidar_identity_axes()
    box = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0])
    points = np.array([
        [10.0 + dx, dy, dz]
        for dx in (-1.0, 0.0, 1.0)
        for dy in (-0.5, 0.5)
        for dz in (-0.5, 0.5)
    ], dtype=np.float32)
    roi = project_box(box, transform, (500, 500, 320, 240), (640, 480))
    camera = [CameraDetection(roi, 'car', 0.9)]
    state = analyzer.analyze(
        [box], ['Vehicle'], lidar_points=points,
        camera_detections=camera, camera_from_source=transform,
        intrinsics=(500, 500, 320, 240), image_size=(640, 480),
        camera_available=True)[0]
    assert state.lidar_point_count == len(points)
    assert state.lidar_supported
    assert state.camera_supported
    assert state.state == ObservationClass.MULTIMODAL_STRONG.value


def test_depth_unavailable_is_not_zero_evidence():
    analyzer = ObservationAnalyzer(ObservationConfig(depth_min_valid_pixels=4))
    transform = camera_from_lidar_identity_axes()
    box = np.array([5.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0])
    state = analyzer.analyze(
        [box], ['Vehicle'], camera_from_source=transform,
        intrinsics=(100, 100, 50, 50), image_size=(100, 100),
        depth_image=np.zeros((100, 100), dtype=np.uint16))[0]
    assert state.depth_available
    assert state.depth_consistency is None


def test_depth_consistency_uses_camera_forward_depth():
    config = ObservationConfig(depth_min_valid_pixels=4, depth_sigma_m=1.0)
    analyzer = ObservationAnalyzer(config)
    transform = camera_from_lidar_identity_axes()
    box = np.array([5.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0])
    depth = np.full((100, 100), 5000, dtype=np.uint16)
    state = analyzer.analyze(
        [box], ['Vehicle'], camera_from_source=transform,
        intrinsics=(100, 100, 50, 50), image_size=(100, 100),
        depth_image=depth)[0]
    assert math.isclose(state.depth_consistency, 1.0, rel_tol=1e-5)
