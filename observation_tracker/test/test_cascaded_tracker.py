import numpy as np

from observation_tracker.tracking_core import TimeAwareTrack

from observation_tracker.cascaded_association import (
    AssociationConfig, CascadedTracker, StaticCvTrack,
    mahalanobis_squared_2d)
from observation_tracker.observation_state import (
    CameraState, DepthState, LidarState, ObservationClass, ObservationResult,
    TrackStatus)
from observation_tracker.track_lifecycle import LifecycleConfig


def detection(x=5.0):
    return np.asarray([
        [x, 0.0, 0.0, 4.0, 2.0, 1.7, 0.0, 1.0, 0.9]
    ], dtype=np.float32)


def test_fast_mahalanobis_matches_pseudoinverse():
    covariance = np.asarray([[2.0, 0.3], [0.3, 1.5]], dtype=np.float32)
    delta = np.asarray([1.2, -0.7], dtype=np.float32)
    expected = float(delta.T @ np.linalg.pinv(covariance) @ delta)
    assert np.isclose(
        mahalanobis_squared_2d(delta, covariance), expected, rtol=1e-6)


def pedestrian_detection(x=5.0):
    item = detection(x)
    item[0, 3:6] = (0.8, 0.7, 1.7)
    item[0, 7] = 2
    return item


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


def test_confirmation_then_camera_supported_prediction_keeps_id():
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
    assert confirmed[0]["association_stage"] == "UNIFIED_ASSOCIATION"

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
    assert recovered[0]["association_stage"] == "SUPPORTED_PREDICTION"
    assert recovered[0]["miss_count"] == 0
    assert 5.1 < recovered[0]["box"][0] <= 5.35


def test_sensor_supported_track_survives_long_3d_detector_gap():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        lifecycle_config=LifecycleConfig(
            strong_confirmation_hits=2, max_miss_count=6))
    strong = observation(ObservationClass.MULTIMODAL_STRONG.value)
    tracker.update(
        pedestrian_detection(5.0), timestamp=0.0,
        detection_observations=[strong])

    def strong_provider(boxes, labels):
        del labels
        return [observation(ObservationClass.MULTIMODAL_STRONG.value)
                for _ in boxes]

    tracker.update(
        pedestrian_detection(5.0), timestamp=0.1,
        observation_provider=strong_provider,
        detection_observations=[strong])
    original_id = tracker.tracks[0].id

    for frame in range(2, 12):
        predicted = tracker.update(
            np.empty((0, 9), dtype=np.float32), timestamp=0.1 * frame,
            observation_provider=strong_provider)
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].id == original_id
        assert tracker.lifecycle.memories[original_id].miss_count == 0
        assert predicted[0]["association_stage"] == "SUPPORTED_PREDICTION"

    reappeared = tracker.update(
        pedestrian_detection(5.0), timestamp=1.2,
        observation_provider=strong_provider,
        detection_observations=[strong])

    assert len(tracker.tracks) == 1
    assert reappeared[0]["id"] == original_id
    assert reappeared[0]["association_stage"] == "UNIFIED_ASSOCIATION"


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
    assert result["association_stage"] == "UNIFIED_ASSOCIATION"
    assert tracker.tracks[0].hits == hits_before
    assert result["miss_count"] == 0
    assert tracker.tracks[0].box[0] > 5.1


def test_repeated_weak_detection_reuses_tentative_track_id():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(expected_dt=0.1)
    weak = observation(ObservationClass.WEAK_OBSERVATION.value)
    tracker.update(detection(), timestamp=0.0, detection_observations=[weak])

    def weak_provider(boxes, labels):
        del labels
        return [observation(ObservationClass.WEAK_OBSERVATION.value)
                for _ in boxes]

    for frame in range(1, 8):
        tracker.update(
            detection(5.0 + 0.02 * frame), timestamp=0.1 * frame,
            observation_provider=weak_provider,
            detection_observations=[observation(
                ObservationClass.WEAK_OBSERVATION.value)])
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].id == 0
        assert tracker.last_debug_tracks[0]["miss_count"] == 0
        assert tracker.last_debug_tracks[0]["association_stage"] == (
            "UNIFIED_ASSOCIATION")


def test_static_ego_fast_path_skips_redundant_observation_analysis():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        association_config=AssociationConfig(
            static_observation_interval=10),
        lifecycle_config=LifecycleConfig(strong_confirmation_hits=2))
    strong = observation(ObservationClass.MULTIMODAL_STRONG.value)
    tracker.update(detection(), timestamp=0.0, detection_observations=[strong])
    tracker.update(detection(), timestamp=0.1, detection_observations=[strong])
    track = tracker.tracks[0]
    track.static_time_sec = 1.0
    calls = []

    def provider(boxes, labels):
        calls.append(len(boxes))
        return [observation(ObservationClass.MULTIMODAL_STRONG.value)
                for _ in labels]

    result = tracker.update(
        detection(), timestamp=0.2,
        ego_motion={"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
        observation_provider=provider)

    assert calls == []
    assert result[0]["association_stage"] == "STATIC_EGO_ASSOCIATION"


def test_static_fast_path_wakes_on_ego_rotation():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        association_config=AssociationConfig(
            static_observation_interval=10,
            static_fast_max_yaw_rate=0.2),
        lifecycle_config=LifecycleConfig(strong_confirmation_hits=2))
    strong = observation(ObservationClass.MULTIMODAL_STRONG.value)
    tracker.update(detection(), timestamp=0.0, detection_observations=[strong])
    tracker.update(detection(), timestamp=0.1, detection_observations=[strong])
    tracker.tracks[0].static_time_sec = 1.0
    calls = []

    def provider(boxes, labels):
        calls.append(len(boxes))
        return [observation(ObservationClass.MULTIMODAL_STRONG.value)
                for _ in labels]

    result = tracker.update(
        detection(), timestamp=0.2,
        ego_motion={"vx": 0.0, "vy": 0.0, "yaw_rate": 0.3},
        observation_provider=provider)

    assert calls == [1]
    assert result[0]["association_stage"] == "UNIFIED_ASSOCIATION"


def test_final_output_hides_unsupported_prediction_but_keeps_internal_id():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        association_config=AssociationConfig(output_prediction_max_misses=2),
        lifecycle_config=LifecycleConfig(
            strong_confirmation_hits=2, recovery_max_age=2,
            max_miss_count=8, weak_observation_max_frames=8,
            min_track_confidence=-1.0))
    strong = observation(ObservationClass.MULTIMODAL_STRONG.value)
    tracker.update(detection(), timestamp=0.0, detection_observations=[strong])

    def strong_provider(boxes, labels):
        del labels
        return [observation(ObservationClass.MULTIMODAL_STRONG.value)
                for _ in boxes]

    assert tracker.update(
        detection(5.1), timestamp=0.1,
        observation_provider=strong_provider,
        detection_observations=[strong])

    def unavailable_provider(boxes, labels):
        del labels
        return [ObservationResult(
            lidar_available=False, camera_available=False)
            for _ in boxes]

    assert tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.2,
        observation_provider=unavailable_provider)
    assert tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.3,
        observation_provider=unavailable_provider)
    assert tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.4,
        observation_provider=unavailable_provider) == []
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].id == 0


def test_pedestrian_and_vehicle_use_same_base_motion_model():
    pedestrian_tracker = CascadedTracker()
    pedestrian_tracker.update(pedestrian_detection(0.0), timestamp=0.0)
    vehicle_tracker = CascadedTracker()
    vehicle_tracker.update(detection(0.0), timestamp=0.0)
    assert type(pedestrian_tracker.tracks[0]) is StaticCvTrack
    assert type(vehicle_tracker.tracks[0]) is StaticCvTrack


def test_association_gate_expands_with_prediction_uncertainty():
    TimeAwareTrack._next_id = 0
    confident = CascadedTracker(expected_dt=0.1)
    confident.update(detection(0.0), timestamp=0.0)
    confident.tracks[0].P *= 0.01
    confident.update(detection(2.0), timestamp=0.1)
    assert len(confident.tracks) == 2

    TimeAwareTrack._next_id = 0
    uncertain = CascadedTracker(expected_dt=0.1)
    uncertain.update(detection(0.0), timestamp=0.0)
    uncertain.tracks[0].P *= 25.0
    uncertain.update(detection(2.0), timestamp=0.1)
    assert len(uncertain.tracks) == 1
    assert uncertain.last_debug_tracks[0]["association_stage"] == (
        "UNIFIED_ASSOCIATION")


def test_stationary_measurement_switches_motion_probability_from_cv():
    item = StaticCvTrack(
        pedestrian_detection(0.0)[0], 0.0, 0.1, 0.2, 0.6)
    item.x[2, 0] = 5.0
    item.object_velocity[0] = 5.0
    item.motion_mode_probabilities[:] = (0.1, 0.9)

    item.predict(0.1)
    item.update(
        pedestrian_detection(0.0)[0], 0.1, 0.2, 0.04, 0.20)

    assert item.motion_mode_probabilities[0] > 0.1
    assert item.speed < 5.0


def test_stopped_pedestrian_recovers_id_after_prediction_overshoots_box():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        lifecycle_config=LifecycleConfig(strong_confirmation_hits=2))
    strong = observation(ObservationClass.MULTIMODAL_STRONG.value)
    tracker.update(
        pedestrian_detection(0.0), timestamp=0.0,
        detection_observations=[strong])
    tracker.update(
        pedestrian_detection(0.1), timestamp=0.1,
        detection_observations=[strong])
    track = tracker.tracks[0]
    original_id = track.id
    track.box[0] = track.x[0, 0] = 0.0
    track.x[2, 0] = 5.0
    track.object_velocity[0] = 5.0
    track.dynamic_votes = 2

    tracker.update(np.empty((0, 9), dtype=np.float32), timestamp=0.2)
    assert tracker.lifecycle.memories[original_id].miss_count == 1
    recovered = tracker.update(
        pedestrian_detection(0.0), timestamp=0.3,
        detection_observations=[strong])

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].id == original_id
    assert recovered[0]["association_stage"] == "UNIFIED_ASSOCIATION"


def test_overlapping_pedestrian_is_kept_occluded_and_recovers_same_id():
    TimeAwareTrack._next_id = 0
    tracker = CascadedTracker(
        expected_dt=0.1,
        association_config=AssociationConfig(
            occlusion_overlap_threshold=0.45,
            occlusion_confirmation_frames=2,
            occlusion_max_age=8),
        lifecycle_config=LifecycleConfig(strong_confirmation_hits=2))
    pair = np.vstack((pedestrian_detection(3.0), pedestrian_detection(5.0)))
    tracker.update(
        pair, timestamp=0.0,
        detection_observations=[
            observation(ObservationClass.MULTIMODAL_STRONG.value),
            observation(ObservationClass.MULTIMODAL_STRONG.value)])
    tracker.update(
        pair, timestamp=0.1,
        detection_observations=[
            observation(ObservationClass.MULTIMODAL_STRONG.value),
            observation(ObservationClass.MULTIMODAL_STRONG.value)])
    front_id, back_id = [track.id for track in tracker.tracks]

    def occlusion_provider(boxes, labels):
        del labels
        results = []
        for box in boxes:
            if box[0] < 4.0:
                item = observation(ObservationClass.MULTIMODAL_STRONG.value)
                item.projected_roi = (20.0, 20.0, 80.0, 90.0)
                item.object_range = 3.0
            else:
                item = ObservationResult(
                    camera_available=True, camera_visible=True,
                    depth_available=True, depth_state=DepthState.OCCLUDED,
                    state=ObservationClass.WEAK_OBSERVATION.value,
                    raw_state=ObservationClass.WEAK_OBSERVATION.value,
                    projected_roi=(30.0, 30.0, 70.0, 85.0),
                    object_range=5.0)
            results.append(item)
        return results

    front_detection = pedestrian_detection(3.0)
    for timestamp in (0.2, 0.3):
        tracker.update(
            front_detection, timestamp=timestamp,
            observation_provider=occlusion_provider,
            detection_observations=[observation(
                ObservationClass.MULTIMODAL_STRONG.value)])

    hidden = tracker.lifecycle.memories[back_id]
    assert hidden.current_track_status == TrackStatus.OCCLUDED
    assert hidden.occluded_by_track_id == front_id
    assert hidden.occlusion_ratio > 0.45

    reappeared = np.vstack((
        pedestrian_detection(3.0), pedestrian_detection(6.2)))
    tracker.update(
        reappeared, timestamp=0.4,
        observation_provider=occlusion_provider,
        detection_observations=[
            observation(ObservationClass.MULTIMODAL_STRONG.value),
            observation(ObservationClass.MULTIMODAL_STRONG.value)])
    recovered = next(
        track for track in tracker.last_debug_tracks if track["id"] == back_id)
    assert len(tracker.tracks) == 2
    assert recovered["association_stage"] == "UNIFIED_ASSOCIATION"
    assert recovered["track_status"] == TrackStatus.CONFIRMED.value
