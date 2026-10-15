import numpy as np

from observation_tracker.tracking_core import TimeAwareTrack

from observation_tracker.observation_state import (
    ObservationClass, ObservationResult, TrackStatus)
from observation_tracker.track_lifecycle import (
    LifecycleConfig, TrackLifecycleManager)


def track():
    detection = np.array(
        [5.0, 0.0, 0.0, 4.0, 2.0, 1.7, 0.0, 1.0, 0.9],
        dtype=np.float32)
    return TimeAwareTrack(detection, 0.0, 0.1, 0.2, 0.6)


def weak():
    return ObservationResult(
        state=ObservationClass.WEAK_OBSERVATION.value,
        raw_state=ObservationClass.WEAK_OBSERVATION.value,
        lidar_available=True, camera_available=True,
        observation_score=0.0)


def test_persistent_weak_observation_terminates_but_one_frame_does_not():
    manager = TrackLifecycleManager(LifecycleConfig(
        recovery_max_age=1, max_miss_count=5,
        weak_observation_max_frames=3, confidence_decay=0.2))
    item = track()
    memory = manager.create(item, weak())
    memory = manager.missed(item, weak())
    assert memory.current_track_status != TrackStatus.TERMINATED
    memory = manager.missed(item, weak())
    assert memory.current_track_status == TrackStatus.TERMINATED


def test_unavailable_sensors_are_not_negative_evidence():
    manager = TrackLifecycleManager(LifecycleConfig(tentative_min_hits=3))
    item = track()
    unavailable = ObservationResult(
        state=ObservationClass.WEAK_OBSERVATION.value,
        raw_state=ObservationClass.WEAK_OBSERVATION.value,
        lidar_available=False, camera_available=False)
    manager.create(item, unavailable)
    item.hits = 3
    memory = manager.matched(item, unavailable)
    assert memory.current_track_status == TrackStatus.CONFIRMED


def test_unobservable_track_decays_slower_than_visible_empty_track():
    config = LifecycleConfig(confidence_decay=0.20, min_track_confidence=-1.0)
    hidden_manager = TrackLifecycleManager(config)
    visible_manager = TrackLifecycleManager(config)
    hidden_track, visible_track = track(), track()
    initial = weak()
    initial.observation_score = 0.9
    hidden_manager.create(hidden_track, initial)
    visible_manager.create(visible_track, initial)
    hidden = ObservationResult(lidar_available=False, camera_available=False)
    visible = ObservationResult(
        lidar_available=True, lidar_visible=True,
        camera_available=True, camera_visible=True)

    hidden_memory = hidden_manager.missed(hidden_track, hidden)
    visible_memory = visible_manager.missed(visible_track, visible)

    assert hidden_memory.track_confidence > visible_memory.track_confidence


def test_supported_prediction_resets_detector_miss_streak():
    manager = TrackLifecycleManager(LifecycleConfig(
        recovery_max_age=1, max_miss_count=3,
        weak_observation_max_frames=10, min_track_confidence=0.0))
    item = track()
    supported = ObservationResult(
        state=ObservationClass.LIDAR_DOMINANT.value,
        raw_state=ObservationClass.LIDAR_DOMINANT.value,
        lidar_available=True, lidar_supported=True,
        observation_score=0.8)
    manager.create(item, supported)
    memory = manager.missed(item, weak())
    assert memory.miss_count == 1

    memory = manager.recovered(item, supported)

    assert memory.miss_count == 0
    assert memory.current_track_status == TrackStatus.RECOVERING
