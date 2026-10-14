"""Bounded temporal evidence retained independently for every track."""

from collections import Counter, deque
from dataclasses import dataclass, field

from .observation_state import (
    DepthState, ObservationClass, TrackStatus)


@dataclass
class TrackMemory:
    track_id: int
    class_id: int
    history_size: int = 7
    age: int = 1
    hit_count: int = 1
    miss_count: int = 0
    current_observation_state: str = ObservationClass.WEAK_OBSERVATION.value
    track_confidence: float = 0.5
    recovery_count: int = 0
    occlusion_count: int = 0
    occluded_by_track_id: int = -1
    occlusion_ratio: float = 0.0
    current_track_status: TrackStatus = TrackStatus.TENTATIVE
    consecutive_state_count: int = 0
    weak_count: int = 0
    observation_state_history: deque = field(init=False)
    lidar_support_history: deque = field(init=False)
    camera_support_history: deque = field(init=False)
    depth_history: deque = field(init=False)

    def __post_init__(self):
        size = max(1, int(self.history_size))
        self.observation_state_history = deque(maxlen=size)
        self.lidar_support_history = deque(maxlen=size)
        self.camera_support_history = deque(maxlen=size)
        self.depth_history = deque(maxlen=size)

    def observe(self, observation, smoothing=True, switch_count=2):
        if observation is None:
            return self.current_observation_state
        raw_state = str(observation.raw_state)
        previous_raw = (
            self.observation_state_history[-1]
            if self.observation_state_history else None)
        self.consecutive_state_count = (
            self.consecutive_state_count + 1 if raw_state == previous_raw else 1)
        self.observation_state_history.append(raw_state)
        self.lidar_support_history.append(bool(observation.lidar_supported))
        self.camera_support_history.append(bool(observation.camera_supported))
        self.depth_history.append(str(observation.depth_state.value))
        self.weak_count = (
            self.weak_count + 1
            if raw_state == ObservationClass.WEAK_OBSERVATION.value else 0)

        selected = raw_state
        if smoothing and raw_state != self.current_observation_state:
            counts = Counter(self.observation_state_history)
            majority, majority_count = counts.most_common(1)[0]
            if (self.consecutive_state_count < max(1, int(switch_count)) and
                    majority_count <= len(self.observation_state_history) // 2):
                selected = self.current_observation_state
            else:
                selected = majority
        self.current_observation_state = selected
        observation.state = selected
        alpha = 2.0 / (len(self.observation_state_history) + 1.0)
        self.track_confidence = (
            (1.0 - alpha) * self.track_confidence +
            alpha * float(observation.observation_score))
        return selected

    @property
    def recent_lidar_support(self):
        return bool(self.lidar_support_history) and sum(
            self.lidar_support_history) > len(self.lidar_support_history) // 2

    @property
    def recent_camera_support(self):
        return bool(self.camera_support_history) and sum(
            self.camera_support_history) > len(self.camera_support_history) // 2

    @property
    def last_depth_state(self):
        return (
            self.depth_history[-1] if self.depth_history
            else DepthState.UNAVAILABLE.value)
