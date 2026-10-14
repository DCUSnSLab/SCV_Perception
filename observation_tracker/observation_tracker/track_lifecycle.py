"""Observation-aware track initialization, recovery and termination policy."""

from dataclasses import dataclass

from .observation_state import ObservationClass, TrackStatus
from .track_memory import TrackMemory


@dataclass
class LifecycleConfig:
    tentative_min_hits: int = 3
    strong_confirmation_hits: int = 2
    max_miss_count: int = 8
    recovery_max_age: int = 4
    weak_observation_max_frames: int = 5
    history_size: int = 7
    confidence_decay: float = 0.12
    min_track_confidence: float = 0.10
    enable_adaptive_lifecycle: bool = True
    enable_temporal_smoothing: bool = True
    smoothing_switch_count: int = 2


class TrackLifecycleManager:
    def __init__(self, config=None):
        self.config = config or LifecycleConfig()
        self.memories = {}

    def create(self, track, observation):
        state = (
            observation.state if observation is not None
            else ObservationClass.WEAK_OBSERVATION.value)
        memory = TrackMemory(
            track_id=track.id,
            class_id=track.cls_id,
            history_size=self.config.history_size,
            current_observation_state=state,
            track_confidence=(
                max(float(track.score), float(observation.observation_score))
                if observation is not None else float(track.score)))
        memory.observe(
            observation, self.config.enable_temporal_smoothing,
            self.config.smoothing_switch_count)
        if (self.config.enable_adaptive_lifecycle and observation is not None and
                observation.state == ObservationClass.WEAK_OBSERVATION.value and
                observation.lidar_available and observation.lidar_visible and
                observation.camera_available and observation.camera_visible):
            memory.current_track_status = TrackStatus.SUPPRESSED
        self.memories[track.id] = memory
        return memory

    def _memory(self, track, observation=None):
        return self.memories.get(track.id) or self.create(track, observation)

    def matched(self, track, observation):
        memory = self._memory(track, observation)
        memory.age += 1
        memory.hit_count = int(track.hits)
        memory.miss_count = 0
        memory.occlusion_count = 0
        memory.occluded_by_track_id = -1
        memory.occlusion_ratio = 0.0
        state = memory.observe(
            observation, self.config.enable_temporal_smoothing,
            self.config.smoothing_switch_count)
        if not self.config.enable_adaptive_lifecycle:
            memory.current_track_status = (
                TrackStatus.CONFIRMED
                if memory.hit_count >= self.config.tentative_min_hits
                else TrackStatus.TENTATIVE)
            return memory
        strong = state in {
            ObservationClass.MULTIMODAL_STRONG.value,
            ObservationClass.LIDAR_DOMINANT.value,
        }
        both_observable = (
            observation is not None and
            observation.lidar_available and observation.lidar_visible and
            observation.camera_available and observation.camera_visible)
        required = (
            self.config.strong_confirmation_hits if strong
            else self.config.tentative_min_hits)
        if memory.hit_count >= required and not (
                state == ObservationClass.WEAK_OBSERVATION.value and
                both_observable):
            memory.current_track_status = TrackStatus.CONFIRMED
        elif (state == ObservationClass.WEAK_OBSERVATION.value and
              both_observable):
            memory.current_track_status = TrackStatus.SUPPRESSED
        else:
            memory.current_track_status = TrackStatus.TENTATIVE
        memory.track_confidence = min(
            1.0, max(memory.track_confidence, float(track.score)))
        return memory

    def recovered(self, track, observation):
        memory = self._memory(track, observation)
        memory.age += 1
        memory.miss_count = 0
        memory.recovery_count += 1
        memory.occlusion_count = 0
        memory.occluded_by_track_id = -1
        memory.occlusion_ratio = 0.0
        memory.observe(
            observation, self.config.enable_temporal_smoothing,
            self.config.smoothing_switch_count)
        memory.current_track_status = TrackStatus.RECOVERING
        memory.track_confidence = min(
            1.0, memory.track_confidence + 0.03)
        return memory

    def weak_matched(self, track, observation):
        """Keep identity and position continuity without confirming a weak hit."""
        memory = self._memory(track, observation)
        previous_status = memory.current_track_status
        memory.age += 1
        memory.miss_count = 0
        memory.occlusion_count = 0
        memory.occluded_by_track_id = -1
        memory.occlusion_ratio = 0.0
        memory.observe(
            observation, self.config.enable_temporal_smoothing,
            self.config.smoothing_switch_count)
        if previous_status == TrackStatus.CONFIRMED:
            memory.current_track_status = TrackStatus.CONFIRMED
        elif previous_status in {
                TrackStatus.RECOVERING, TrackStatus.OCCLUDED,
                TrackStatus.LOST}:
            memory.current_track_status = TrackStatus.RECOVERING
        else:
            memory.current_track_status = previous_status
        return memory

    def occluded(self, track, observation, occluder_id, overlap,
                 confirmation_frames, max_age):
        """Retain a missed track while a nearer tracked object blocks its ROI."""
        memory = self._memory(track, observation)
        memory.age += 1
        memory.miss_count += 1
        memory.observe(
            observation, self.config.enable_temporal_smoothing,
            self.config.smoothing_switch_count)
        memory.occlusion_count += 1
        memory.occluded_by_track_id = int(occluder_id)
        memory.occlusion_ratio = float(overlap)
        memory.track_confidence = max(
            0.0, memory.track_confidence - 0.25 * self.config.confidence_decay)
        if memory.miss_count >= max(1, int(max_age)):
            memory.current_track_status = TrackStatus.TERMINATED
        elif memory.occlusion_count >= max(1, int(confirmation_frames)):
            memory.current_track_status = TrackStatus.OCCLUDED
        else:
            memory.current_track_status = TrackStatus.LOST
        return memory

    def missed(self, track, observation):
        memory = self._memory(track, observation)
        memory.occlusion_count = 0
        memory.occluded_by_track_id = -1
        memory.occlusion_ratio = 0.0
        memory.age += 1
        state = memory.observe(
            observation, self.config.enable_temporal_smoothing,
            self.config.smoothing_switch_count)
        if not self.config.enable_adaptive_lifecycle:
            memory.miss_count += 1
            memory.current_track_status = (
                TrackStatus.TERMINATED
                if memory.miss_count >= self.config.max_miss_count
                else TrackStatus.LOST)
            return memory
        memory.miss_count += 1
        sensor_support = (
            observation is not None and
            (observation.lidar_supported or observation.camera_supported))
        observable = (
            observation is not None and
            ((observation.lidar_available and observation.lidar_visible) or
             (observation.camera_available and observation.camera_visible)))
        if sensor_support:
            memory.track_confidence = max(
                memory.track_confidence, float(observation.observation_score))
        else:
            decay = self.config.confidence_decay * (
                1.0 if observable else 0.25)
            memory.track_confidence = max(
                0.0, memory.track_confidence - decay)
        terminate = (
            memory.miss_count >= self.config.max_miss_count or
            (memory.weak_count >= self.config.weak_observation_max_frames and
             memory.miss_count > self.config.recovery_max_age) or
            (memory.track_confidence <= self.config.min_track_confidence and
             memory.miss_count > self.config.recovery_max_age))
        if terminate:
            memory.current_track_status = TrackStatus.TERMINATED
        elif sensor_support:
            memory.current_track_status = TrackStatus.RECOVERING
        elif memory.miss_count > self.config.recovery_max_age:
            memory.current_track_status = TrackStatus.LOST
        elif state == ObservationClass.WEAK_OBSERVATION.value:
            memory.current_track_status = TrackStatus.LOST
        return memory

    def remove(self, track_id):
        self.memories.pop(track_id, None)
