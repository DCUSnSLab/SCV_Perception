import numpy as np

from pcdet_tracker.jay_tracker import TimeAwareTrack

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
