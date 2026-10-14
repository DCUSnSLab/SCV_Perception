import numpy as np
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from observation_tracker.camera_observation import CameraDetection
from observation_tracker.depth_consistency import (
    DepthConsistencyAnalyzer, DepthConsistencyConfig)
from observation_tracker.lidar_observation import (
    LidarObservationAnalyzer, LidarObservationConfig)
from observation_tracker.observation_analyzer import ObservationAnalyzer
from observation_tracker.observation_state import (
    CameraState, DepthState, LidarState, ObservationClass, ObservationResult)
from observation_tracker.visualization import ObservationVisualizer


def camera_from_lidar():
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    return matrix


def test_class_distance_lidar_threshold_and_centroid():
    config = LidarObservationConfig(
        near_range=10.0, middle_range=20.0, far_range=40.0,
        weak_min_points=2, centroid_distance_threshold=1.0,
        thresholds={
            "vehicle": {"near": 5, "middle": 3, "far": 2},
            "pedestrian": {"near": 3, "middle": 2, "far": 2},
            "cyclist": {"near": 4, "middle": 3, "far": 2},
            "unknown": {"near": 5, "middle": 3, "far": 2},
        })
    analyzer = LidarObservationAnalyzer(config)
    box = np.array([5.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0])
    points = np.array([
        [5.0, 0.0, 0.0], [5.2, 0.1, 0.1], [4.8, -0.1, -0.1],
        [5.1, -0.2, 0.2], [4.9, 0.2, -0.2]], dtype=np.float32)
    evidence = analyzer.analyze(box, "Vehicle", points)
    assert evidence.state == LidarState.STRONG
    assert evidence.count == 5
    assert evidence.centroid_error < 0.1


def test_depth_distinguishes_consistent_occluded_and_inconsistent():
    analyzer = DepthConsistencyAnalyzer(DepthConsistencyConfig(
        consistency_threshold=0.5, occlusion_depth_threshold=1.0,
        min_valid_pixels=4))
    box = np.array([5.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0])
    roi = (10, 10, 20, 20)
    transform = camera_from_lidar()
    consistent = analyzer.analyze(
        box, roi, np.full((30, 30), 4000, np.uint16), transform)
    occluded = analyzer.analyze(
        box, roi, np.full((30, 30), 2000, np.uint16), transform)
    inconsistent = analyzer.analyze(
        box, roi, np.full((30, 30), 6000, np.uint16), transform)
    assert consistent.state == DepthState.CONSISTENT
    assert consistent.support_ratio == 1.0
    assert occluded.state == DepthState.OCCLUDED
    assert inconsistent.state == DepthState.INCONSISTENT


def test_depth_finds_expected_object_among_roi_background():
    analyzer = DepthConsistencyAnalyzer(DepthConsistencyConfig(
        consistency_threshold=0.5, occlusion_depth_threshold=1.0,
        min_valid_pixels=20, roi_inner_scale=0.6, min_support_ratio=0.15))
    box = np.array([5.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0])
    depth = np.full((30, 30), 15000, np.uint16)
    depth[13:18, 13:18] = 4000
    evidence = analyzer.analyze(
        box, (5, 5, 25, 25), depth, camera_from_lidar())
    assert evidence.state == DepthState.CONSISTENT
    assert evidence.support_pixel_count == 25
    assert evidence.support_ratio >= 0.15


def test_multimodal_state_uses_raw_sensor_evidence():
    lidar = LidarObservationAnalyzer(LidarObservationConfig(
        weak_min_points=2, centroid_distance_threshold=1.0,
        thresholds={
            name: {"near": 3, "middle": 3, "far": 2}
            for name in ("vehicle", "pedestrian", "cyclist", "unknown")}))
    analyzer = ObservationAnalyzer(lidar=lidar)
    box = np.array([5.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0])
    points = np.array([
        [5.0, 0.0, 0.0], [5.1, 0.1, 0.0], [4.9, -0.1, 0.0]],
        dtype=np.float32)
    transform = camera_from_lidar()
    # First obtain the projected ROI, then provide a matching camera detection.
    initial = analyzer.analyze(
        [box], ["Vehicle"], lidar_points=points,
        camera_from_source=transform, intrinsics=(100, 100, 50, 50),
        image_size=(100, 100), camera_available=True,
        depth_image=np.full((100, 100), 5000, np.uint16))[0]
    result = analyzer.analyze(
        [box], ["Vehicle"], lidar_points=points,
        camera_detections=[CameraDetection(initial.projected_roi, "car", 0.9)],
        camera_from_source=transform, intrinsics=(100, 100, 50, 50),
        image_size=(100, 100), camera_available=True,
        depth_image=np.full((100, 100), 5000, np.uint16))[0]
    assert result.state == ObservationClass.MULTIMODAL_STRONG.value
    assert result.camera_supported
    assert result.camera_detection_class == "car"
    assert result.depth_state == DepthState.CONSISTENT


def test_person_depth_projection_selects_overlapping_pedestrian():
    analyzer = ObservationAnalyzer()
    box = np.array([5.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    transform = camera_from_lidar()
    initial = analyzer.analyze(
        [box], ["Pedestrian"], camera_from_source=transform,
        intrinsics=(100, 100, 50, 50), image_size=(100, 100),
        camera_available=True)[0]
    detection = CameraDetection(initial.projected_roi, "person", 0.9)
    depth = np.full((100, 100), 5000, np.uint16)
    result = analyzer.analyze(
        [box], ["Pedestrian"], camera_detections=[detection],
        camera_from_source=transform, intrinsics=(100, 100, 50, 50),
        image_size=(100, 100), camera_available=True,
        depth_image=depth)[0]
    assert result.camera_supported
    assert result.camera_detection_class == "person"
    assert result.camera_depth_overlap > 0.0


def test_person_depth_projection_rejects_non_overlapping_pedestrian():
    analyzer = ObservationAnalyzer()
    box = np.array([5.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    transform = camera_from_lidar()
    initial = analyzer.analyze(
        [box], ["Pedestrian"], camera_from_source=transform,
        intrinsics=(100, 100, 50, 50), image_size=(100, 100),
        camera_available=True)[0]
    detection = CameraDetection(initial.projected_roi, "person", 0.9)
    result = analyzer.analyze(
        [box], ["Pedestrian"], camera_detections=[detection],
        camera_from_source=transform, intrinsics=(100, 100, 50, 50),
        image_size=(100, 100), camera_available=True,
        depth_image=np.full((100, 100), 9000, np.uint16))[0]
    assert not result.camera_supported
    assert result.camera_depth_overlap == 0.0


def test_one_person_detection_supports_only_one_3d_pedestrian():
    analyzer = ObservationAnalyzer()
    boxes = np.array([
        [5.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0],
        [5.0, 0.2, 0.0, 1.0, 1.0, 2.0, 0.0],
    ])
    transform = camera_from_lidar()
    initial = analyzer.analyze(
        boxes[:1], ["Pedestrian"], camera_from_source=transform,
        intrinsics=(100, 100, 50, 50), image_size=(100, 100),
        camera_available=True)[0]
    results = analyzer.analyze(
        boxes, ["Pedestrian", "Pedestrian"],
        camera_detections=[CameraDetection(initial.projected_roi, "person", 0.9)],
        camera_from_source=transform, intrinsics=(100, 100, 50, 50),
        image_size=(100, 100), camera_available=True,
        depth_image=np.full((100, 100), 5000, np.uint16))
    assert sum(result.camera_supported for result in results) == 1


def test_missing_person_depth_falls_back_to_2d_camera_support():
    analyzer = ObservationAnalyzer()
    box = np.array([5.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    transform = camera_from_lidar()
    initial = analyzer.analyze(
        [box], ["Pedestrian"], camera_from_source=transform,
        intrinsics=(100, 100, 50, 50), image_size=(100, 100),
        camera_available=True)[0]
    result = analyzer.analyze(
        [box], ["Pedestrian"],
        camera_detections=[CameraDetection(initial.projected_roi, "person", 0.9)],
        camera_from_source=transform, intrinsics=(100, 100, 50, 50),
        image_size=(100, 100), camera_available=True,
        depth_image=np.zeros((100, 100), np.uint16))[0]
    assert result.camera_supported
    assert result.depth_state == DepthState.UNAVAILABLE


def test_fused_marker_contains_3d_and_matched_2d_result():
    visualizer = ObservationVisualizer()
    detection = np.array(
        [[5.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0, 1.0, 0.9]],
        dtype=np.float32)
    observation = ObservationResult(
        camera_state=CameraState.AVAILABLE_SUPPORTED,
        camera_available=True,
        camera_supported=True,
        camera_iou=0.42,
        camera_detection_score=0.87,
        camera_detection_class="car")
    markers = visualizer.build_fused_detections(
        detection, [observation], ["Vehicle", "Pedestrian"], Header(),
        tracks=[{
            "id": 42,
            "matched_detection_id": 0,
            "observation_state": ObservationClass.MULTIMODAL_STRONG.value,
            "box": detection[0, :7],
        }])
    assert len(markers.markers) == 2
    assert markers.markers[0].ns == "fused_3d_detections"
    assert "ID 42 3D Vehicle 0.90" in markers.markers[1].text
    assert "2D car 0.87 IoU=0.42" in markers.markers[1].text


def test_fused_marker_hides_track_without_detection_match():
    visualizer = ObservationVisualizer()
    markers = visualizer.build_fused_detections(
        [], [], ["Vehicle", "Pedestrian"], Header(),
        tracks=[{
            "id": 7,
            "matched_detection_id": -1,
            "observation_state": ObservationClass.CAMERA_DOMINANT.value,
            "track_status": "RECOVERING",
            "box": np.array([5.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0]),
        }])
    assert markers.markers == []


def test_fused_marker_deletes_box_immediately_after_detection_miss():
    visualizer = ObservationVisualizer()
    detection = np.array(
        [[5.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0, 1.0, 0.9]],
        dtype=np.float32)
    visualizer.build_fused_detections(
        detection, [], ["Vehicle", "Pedestrian"], Header())
    markers = visualizer.build_fused_detections(
        [], [], ["Vehicle", "Pedestrian"], Header(),
        tracks=[{
            "id": 7,
            "matched_detection_id": -1,
            "observation_state": ObservationClass.WEAK_OBSERVATION.value,
            "box": detection[0, :7],
        }])
    assert len(markers.markers) == 2
    assert all(marker.action == Marker.DELETE for marker in markers.markers)


def test_fused_marker_contains_camera_depth_candidate():
    visualizer = ObservationVisualizer()
    candidate = {
        "id": 3,
        "position": np.array([5.0, 1.0, 0.5]),
        "size": np.array([0.6, 0.8, 1.7]),
        "class_name": "car",
        "score": 0.8,
        "depth": 5.2,
        "track_id": -1,
        "state": "CAMERA_DEPTH_CANDIDATE",
    }
    markers = visualizer.build_fused_detections(
        [], [], ["Vehicle", "Pedestrian"], Header(), [candidate])
    assert len(markers.markers) == 2
    assert markers.markers[0].ns == "camera_depth_candidates"
    assert markers.markers[0].type == Marker.CUBE
    assert markers.markers[0].scale.z == 1.7
    assert "CAMERA_DEPTH_CANDIDATE" in markers.markers[1].text


def test_untracked_fused_detection_does_not_show_detection_id():
    visualizer = ObservationVisualizer()
    detection = np.array(
        [[5.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0, 1.0, 0.9]],
        dtype=np.float32)
    markers = visualizer.build_fused_detections(
        detection, [], ["Vehicle", "Pedestrian"], Header())
    assert markers.markers[1].text.startswith("3D Vehicle 0.90")
    assert "DET" not in markers.markers[1].text
