import numpy as np

from pcdet_tracker.adaptive_track_policy import (
    AdaptivePolicyConfig, AdaptiveTrackPolicy)
from pcdet_tracker.jay_tracker import KittiHzTracker, TimeAwareTrack
from pcdet_tracker.observation_state import ObservationClass, ObservationState


def detection(x=5.0, score=0.9):
    return np.asarray([
        [x, 0.0, 0.0, 4.0, 2.0, 1.7, 0.0, 1.0, score]
    ], dtype=np.float32)


def test_baseline_confirmation_path_is_unchanged():
    TimeAwareTrack._next_id = 0
    tracker = KittiHzTracker(expected_dt=0.1, max_age_frames=2)
    assert tracker.update(detection(), timestamp=0.0) == []
    result = tracker.update(detection(5.1), timestamp=0.1)
    assert len(result) == 1
    assert result[0]['id'] == 0
    assert result[0]['was_detected']
    assert result[0]['observation'] is None


def test_weak_observation_suppresses_new_track_only_when_enabled():
    policy = AdaptiveTrackPolicy(AdaptivePolicyConfig(
        enable_adaptive_association=False,
        enable_adaptive_lifecycle=True))
    tracker = KittiHzTracker(adaptive_policy=policy)
    weak = ObservationState(
        lidar_available=True, lidar_visible=True,
        camera_available=True, camera_visible=True,
        state=ObservationClass.WEAK_OBSERVATION.value)
    tracker.update(
        detection(), timestamp=0.0,
        detection_observations=[weak])
    assert tracker.tracks == []


def test_camera_support_extends_unmatched_track_lifetime():
    TimeAwareTrack._next_id = 0
    policy = AdaptiveTrackPolicy(AdaptivePolicyConfig(
        enable_adaptive_association=False,
        enable_adaptive_lifecycle=True,
        camera_grace_multiplier=2.0))
    tracker = KittiHzTracker(
        expected_dt=0.1, max_age_frames=1,
        adaptive_policy=policy)
    camera = ObservationState(
        camera_available=True, camera_visible=True, camera_supported=True,
        state=ObservationClass.CAMERA_DOMINANT.value,
        observation_score=0.8)
    tracker.update(detection(), timestamp=0.0,
                   detection_observations=[camera])
    tracker.update(detection(5.1), timestamp=0.1,
                   detection_observations=[camera])

    def provider(boxes, labels):
        return [camera for _ in boxes]

    assert tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.2,
        observation_provider=provider)
    assert tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.3,
        observation_provider=provider)
    assert tracker.update(
        np.empty((0, 9), dtype=np.float32), timestamp=0.4,
        observation_provider=provider) == []


def test_adaptive_kf_trusts_high_confidence_measurement_more():
    high = ObservationState(observation_score=1.0)
    low = ObservationState(observation_score=0.0)
    det0 = detection(0.0)[0]
    det1 = detection(1.0)[0]
    track_high = TimeAwareTrack(det0, 0.0, 0.1, 0.2, 0.6)
    track_low = TimeAwareTrack(det0, 0.0, 0.1, 0.2, 0.6)
    track_high.update(
        det1, 0.1, 0.2, 0.04, 0.2, observation=high,
        measurement_noise_scale=0.6)
    track_low.update(
        det1, 0.1, 0.2, 0.04, 0.2, observation=low,
        measurement_noise_scale=2.0)
    assert track_high.box[0] > track_low.box[0]


def test_prediction_moves_once_and_fades_when_detection_is_missing():
    TimeAwareTrack._next_id = 0
    tracker = KittiHzTracker(
        expected_dt=0.1, max_age_frames=9,
        missed_velocity_decay=0.5, publish_prediction_frames=2)
    tracker.update(detection(0.0), timestamp=0.0)
    tracker.update(detection(1.0), timestamp=0.1)
    track = tracker.tracks[0]
    track.x[2, 0] = 1.0
    track.object_velocity[0] = 1.0
    start_x = float(track.box[0])

    first = tracker.update(np.empty((0, 9)), timestamp=0.2)
    assert np.isclose(first[0]['box'][0] - start_x, 0.1, atol=1e-5)
    assert np.isclose(track.x[2, 0], 0.5)
    assert tracker.update(np.empty((0, 9)), timestamp=0.3)
    assert tracker.update(np.empty((0, 9)), timestamp=0.4) == []
    assert tracker.tracks
