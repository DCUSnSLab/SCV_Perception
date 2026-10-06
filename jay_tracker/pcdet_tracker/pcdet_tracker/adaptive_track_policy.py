"""Small, ablatable policy layer on top of Jay_Tracker's baseline logic."""

from dataclasses import dataclass

from .observation_state import ObservationClass


@dataclass
class AdaptivePolicyConfig:
    enable_adaptive_association: bool = True
    enable_adaptive_lifecycle: bool = True
    multimodal_cost_bonus: float = 0.35
    camera_match_cost_bonus: float = 0.30
    camera_conflict_penalty: float = 0.60
    weak_cost_penalty: float = 0.35
    camera_dominant_gate_scale: float = 1.20
    weak_gate_scale: float = 0.75
    camera_grace_multiplier: float = 2.0
    lidar_grace_multiplier: float = 1.5
    suppress_weak_new_tracks: bool = True


class AdaptiveTrackPolicy:
    def __init__(self, config=None):
        self.config = config or AdaptivePolicyConfig()

    def gate_scale(self, track_observation, detection_observation):
        if not self.config.enable_adaptive_association:
            return 1.0
        states = {
            item.state for item in (track_observation, detection_observation)
            if item is not None
        }
        if ObservationClass.CAMERA_DOMINANT.value in states:
            return self.config.camera_dominant_gate_scale
        if states and states == {ObservationClass.WEAK_OBSERVATION.value}:
            return self.config.weak_gate_scale
        return 1.0

    def association_adjustment(self, track_observation,
                               detection_observation):
        if (not self.config.enable_adaptive_association or
                track_observation is None or detection_observation is None):
            return 0.0
        adjustment = 0.0
        if detection_observation.state == ObservationClass.MULTIMODAL_STRONG.value:
            adjustment -= self.config.multimodal_cost_bonus
        track_camera = track_observation.camera_detection_index
        detection_camera = detection_observation.camera_detection_index
        if track_camera >= 0 and detection_camera >= 0:
            if track_camera == detection_camera:
                adjustment -= self.config.camera_match_cost_bonus
            else:
                adjustment += self.config.camera_conflict_penalty
        if (track_observation.state == ObservationClass.WEAK_OBSERVATION.value and
                detection_observation.state == ObservationClass.WEAK_OBSERVATION.value):
            adjustment += self.config.weak_cost_penalty
        return adjustment

    def max_age(self, base_max_age, observation):
        if not self.config.enable_adaptive_lifecycle or observation is None:
            return base_max_age
        if observation.camera_supported:
            return base_max_age * self.config.camera_grace_multiplier
        if observation.lidar_supported:
            return base_max_age * self.config.lidar_grace_multiplier
        return base_max_age

    def allow_spawn(self, observation):
        if (not self.config.enable_adaptive_lifecycle or
                not self.config.suppress_weak_new_tracks or
                observation is None):
            return True
        # Only suppress when both sensors had a real chance to observe it.
        both_observable = (
            observation.lidar_available and observation.lidar_visible and
            observation.camera_available and observation.camera_visible)
        return not (
            both_observable and
            observation.state == ObservationClass.WEAK_OBSERVATION.value)
