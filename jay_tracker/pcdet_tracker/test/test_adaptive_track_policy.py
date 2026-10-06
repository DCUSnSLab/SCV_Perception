from pcdet_tracker.adaptive_track_policy import (
    AdaptivePolicyConfig, AdaptiveTrackPolicy)
from pcdet_tracker.observation_state import ObservationClass, ObservationState


def state(name, **kwargs):
    return ObservationState(state=name, **kwargs)


def test_outside_camera_fov_does_not_suppress_lidar_track():
    policy = AdaptiveTrackPolicy()
    observation = state(
        ObservationClass.LIDAR_DOMINANT.value,
        lidar_available=True, lidar_visible=True, lidar_supported=True,
        camera_available=True, camera_visible=False, camera_supported=False)
    assert policy.allow_spawn(observation)
    assert policy.max_age(1.0, observation) == 1.5


def test_weak_new_detection_suppressed_only_when_both_observable():
    policy = AdaptiveTrackPolicy()
    observable = state(
        ObservationClass.WEAK_OBSERVATION.value,
        lidar_available=True, lidar_visible=True,
        camera_available=True, camera_visible=True)
    unavailable_camera = state(
        ObservationClass.WEAK_OBSERVATION.value,
        lidar_available=True, lidar_visible=True,
        camera_available=False, camera_visible=True)
    assert not policy.allow_spawn(observable)
    assert policy.allow_spawn(unavailable_camera)


def test_shared_camera_detection_reduces_cost():
    policy = AdaptiveTrackPolicy()
    track = state(
        ObservationClass.CAMERA_DOMINANT.value,
        camera_supported=True, camera_detection_index=2)
    detection = state(
        ObservationClass.MULTIMODAL_STRONG.value,
        camera_supported=True, camera_detection_index=2)
    assert policy.association_adjustment(track, detection) < 0.0


def test_disabled_policy_is_baseline_noop():
    policy = AdaptiveTrackPolicy(AdaptivePolicyConfig(
        enable_adaptive_association=False,
        enable_adaptive_lifecycle=False))
    observation = state(
        ObservationClass.MULTIMODAL_STRONG.value,
        camera_supported=True, lidar_supported=True,
        camera_detection_index=1)
    assert policy.gate_scale(observation, observation) == 1.0
    assert policy.association_adjustment(observation, observation) == 0.0
    assert policy.max_age(1.0, observation) == 1.0
    assert policy.allow_spawn(observation)
