import numpy as np

from pcdet_tracker.jay_tracker import TimeAwareTrack

from observation_tracker.cascaded_association import CascadedTracker
from observation_tracker.observation_state import (
    CameraState, DepthState, LidarState, ObservationClass, ObservationResult)
from observation_tracker.track_lifecycle import LifecycleConfig


def detection(x=5.0):
    return np.asarray([
        [x, 0.0, 0.0, 4.0, 2.0, 1.7, 0.0, 1.0, 0.9]
    ], dtype=np.float32)


def observation(state):
    common = dict(
        state=state, raw_state=state, lidar_available=True,
        camera_available=True, observation_score=0.9)
    if state == ObservationClass.MULTIMODAL_STRONG.value:
        return ObservationResult(
            **common, lidar_state=LidarState.STRONG,
            lidar_supported=True,
            camera_state=CameraState.AVAILABLE_SUPPORTED,
            camera_supported=True, camera_iou=0.8,
            camera_detection_index=0,
            depth_state=DepthState.CONSISTENT, depth_consistency=1.0)
    if state == ObservationClass.CAMERA_DOMINANT.value:
        return ObservationResult(
            **common, lidar_state=LidarState.WEAK,
            camera_state=CameraState.AVAILABLE_SUPPORTED,
            camera_supported=True, camera_iou=0.8,
            camera_detection_index=0,
            depth_state=DepthState.CONSISTENT, depth_consistency=1.0)
    return ObservationResult(**common)


def test_stage1_confirmation_then_stage3_camera_recovery_keeps_id():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1, max_age_frames=2,
        lifecycle_config=LifecycleConfig(strong_confirmation_hits=2))
    multimodal = observation(ObservationClass.MULTIMODAL_STRONG.value)

    tracker.update(
        detection(), timestamp=0.0,
        detection_observations=[multimodal])

    def multimodal_provider(boxes, labels):
        del labels
        return [observation(ObservationClass.MULTIMODAL_STRONG.value)
                for _ in boxes]

    confirmed = tracker.update(
        detection(5.1), timestamp=0.1,
        observation_provider=multimodal_provider,
        detection_observations=[multimodal])
    assert len(confirmed) == 1
    assert confirmed[0]["id"] == 0
    assert confirmed[0]["association_stage"] == "STAGE_1"

    def camera_provider(boxes, labels):
        del labels
        results = [observation(ObservationClass.CAMERA_DOMINANT.value)
                   for _ in boxes]
        for result in results:
            result.camera_depth_center = (6.0, 0.0, 0.0)
        return results

    recovered = tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.2,
        observation_provider=camera_provider)
    assert len(recovered) == 1
    assert recovered[0]["id"] == 0
    assert recovered[0]["association_stage"] == "STAGE_3"
    assert recovered[0]["miss_count"] == 0
    assert 5.1 < recovered[0]["box"][0] <= 5.35


def test_camera_dominant_detection_does_not_create_3d_track():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker()
    camera = observation(ObservationClass.CAMERA_DOMINANT.value)
    tracker.update(
        detection(), timestamp=0.0, detection_observations=[camera])
    assert tracker.tracks == []


def test_camera_only_does_not_spawn_when_lidar_stream_is_unavailable():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker()
    camera = observation(ObservationClass.CAMERA_DOMINANT.value)
    camera.lidar_available = False
    tracker.update(
        detection(), timestamp=0.0, detection_observations=[camera])
    assert tracker.tracks == []


def test_weak_detection_keeps_confirmed_track_id_without_counting_hit():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        lifecycle_config=LifecycleConfig(strong_confirmation_hits=2))
    strong = observation(ObservationClass.MULTIMODAL_STRONG.value)
    tracker.update(detection(), timestamp=0.0, detection_observations=[strong])

    def strong_provider(boxes, labels):
        del labels
        return [observation(ObservationClass.MULTIMODAL_STRONG.value)
                for _ in boxes]

    tracker.update(
        detection(5.1), timestamp=0.1,
        observation_provider=strong_provider,
        detection_observations=[strong])
    hits_before = tracker.tracks[0].hits
    weak = observation(ObservationClass.WEAK_OBSERVATION.value)
    weak.lidar_visible = True
    weak.camera_visible = True

    def weak_provider(boxes, labels):
        del labels
        return [observation(ObservationClass.WEAK_OBSERVATION.value)
                for _ in boxes]

    tracker.update(
        detection(5.2), timestamp=0.2,
        observation_provider=weak_provider,
        detection_observations=[weak])
    result = tracker.last_debug_tracks[0]
    assert result["id"] == 0
    assert result["matched_detection_id"] == 0
    assert result["association_stage"] == "STAGE_4_WEAK_CONTINUITY"
    assert tracker.tracks[0].hits == hits_before
