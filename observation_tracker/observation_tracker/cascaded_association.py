"""Four-stage association that preserves Jay track prediction and IDs."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracking_core import (
    KittiHzTracker, TimeAwareTrack, bev_iou, compensate_body_frame_motion)

from .observation_state import (
    DepthState, ObservationClass, TrackStatus)
from .track_lifecycle import LifecycleConfig, TrackLifecycleManager


def mahalanobis_squared_2d(delta, covariance):
    """Evaluate a 2-D quadratic form without a general-purpose SVD."""
    x, y = map(float, delta)
    a, b = map(float, covariance[0])
    c, d = map(float, covariance[1])
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        inverse = np.linalg.pinv(covariance)
        return float(delta.T @ inverse @ delta)
    return float((d * x * x - (b + c) * x * y + a * y * y) /
                 determinant)


@dataclass
class AssociationConfig:
    iou_3d_weight: float = 1.0
    motion_weight: float = 0.5
    camera_iou_weight: float = 0.5
    depth_weight: float = 0.25
    mahalanobis_gate: float = 9.21
    maximum_cost: float = 5.0
    camera_recovery_position_gain: float = 0.25
    camera_recovery_max_correction: float = 1.0
    weak_position_gain: float = 0.25
    output_prediction_max_misses: int = 2
    enable_occlusion_reasoning: bool = True
    occlusion_overlap_threshold: float = 0.45
    occlusion_depth_margin: float = 0.30
    occlusion_confirmation_frames: int = 2
    occlusion_max_age: int = 8
    enable_static_fast_path: bool = True
    static_observation_interval: int = 3
    static_fast_match_distance: float = 0.75
    static_fast_max_yaw_rate: float = 0.20
    static_fast_max_ego_delta: float = 0.75


class StaticCvTrack(TimeAwareTrack):
    """Blend stationary and constant-velocity predictions from their evidence."""

    _transition = np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_mode_probabilities = np.asarray(
            [0.5, 0.5], dtype=np.float32)
        self._mode_predictions = np.vstack((self.box[:2], self.box[:2]))

    def predict(self, timestamp, ego_motion=None, ego_comp_gain=1.0):
        previous_xy = self.x[:2, 0].copy()
        previous_yaw = float(self.box[6])
        dt = super().predict(timestamp, ego_motion, ego_comp_gain)
        cv_xy = self.x[:2, 0].copy()
        static_xy, _ = compensate_body_frame_motion(
            previous_xy, previous_yaw, dt, ego_motion or {}, ego_comp_gain)
        self.motion_mode_probabilities = (
            self._transition.T @ self.motion_mode_probabilities)
        self.motion_mode_probabilities /= np.sum(self.motion_mode_probabilities)
        self._mode_predictions = np.vstack((static_xy, cv_xy))
        mixed_xy = (
            self.motion_mode_probabilities[:, None] *
            self._mode_predictions).sum(axis=0)
        spread = self._mode_predictions - mixed_xy
        self.P[:2, :2] += sum(
            probability * np.outer(delta, delta)
            for probability, delta in zip(
                self.motion_mode_probabilities, spread))
        self.x[:2, 0] = mixed_xy
        self.x[2:4, 0] *= float(self.motion_mode_probabilities[1])
        self.box[:2] = mixed_xy
        return dt

    def update(self, det, timestamp, static_hold_time, static_yaw_alpha,
               dynamic_yaw_alpha, ego_motion=None, observation=None,
               measurement_noise_scale=1.0):
        measurement = np.asarray(det[:2], dtype=np.float32)
        innovation_cov = (
            self.H @ self.P @ self.H.T +
            self.R * max(float(measurement_noise_scale), 1e-3))
        likelihoods = []
        for prediction in self._mode_predictions:
            residual = measurement - prediction
            distance_squared = mahalanobis_squared_2d(
                residual, innovation_cov)
            likelihoods.append(math.exp(-0.5 * min(distance_squared, 50.0)))
        posterior = self.motion_mode_probabilities * np.asarray(
            likelihoods, dtype=np.float32)
        total = float(np.sum(posterior))
        if total > 1e-12:
            self.motion_mode_probabilities = posterior / total
        super().update(
            det, timestamp, static_hold_time, static_yaw_alpha,
            dynamic_yaw_alpha, ego_motion, observation,
            measurement_noise_scale)
        self.x[2:4, 0] *= float(self.motion_mode_probabilities[1])


def iou_3d(box_a, box_b):
    bev = bev_iou(
        [box_a[0], box_a[1], box_a[3], box_a[4], box_a[6]],
        [box_b[0], box_b[1], box_b[3], box_b[4], box_b[6]])
    if bev <= 0.0:
        return 0.0
    a_low, a_high = box_a[2] - box_a[5] * 0.5, box_a[2] + box_a[5] * 0.5
    b_low, b_high = box_b[2] - box_b[5] * 0.5, box_b[2] + box_b[5] * 0.5
    intersection = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    height_union = max(a_high, b_high) - min(a_low, b_low)
    return float(bev * intersection / max(height_union, 1e-6))


class CascadedTracker(KittiHzTracker):
    """Observation-aware tracker with unified uncertainty-based association."""

    def __init__(self, *args, association_config=None, lifecycle_config=None,
                 **kwargs):
        kwargs.setdefault("enable_adaptive_kf", True)
        super().__init__(*args, adaptive_policy=None, **kwargs)
        self.association_config = association_config or AssociationConfig()
        self.lifecycle = TrackLifecycleManager(
            lifecycle_config or LifecycleConfig())
        self.last_debug_tracks = []
        self.last_detection_observations = []
        self.frame_index = 0
        self.previous_ego_motion = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}

    @staticmethod
    def _unavailable(observation):
        return observation is None or (
            not observation.lidar_available and
            not observation.camera_available)

    @staticmethod
    def _supports_prediction(observation):
        if observation is None:
            return False
        camera_support = (
            observation.camera_supported and
            observation.depth_state in {
                DepthState.CONSISTENT, DepthState.UNAVAILABLE})
        return bool(observation.lidar_supported or camera_support)

    def _pair_cost(self, track, detection, detection_observation,
                   projected_covariance=None, noise_scale=None):
        cfg = self.association_config
        if (track.cls_id > 0 and int(detection[7]) > 0 and
                track.cls_id != int(detection[7])):
            return np.inf
        innovation = np.asarray(
            detection[:2] - track.box[:2], dtype=np.float32)
        measurement_noise = track.R * (
            self._measurement_noise_scale(detection_observation)
            if noise_scale is None else noise_scale)
        innovation_covariance = (
            track.H @ track.P @ track.H.T
            if projected_covariance is None else projected_covariance)
        innovation_covariance = innovation_covariance + measurement_noise
        mahalanobis_squared = mahalanobis_squared_2d(
            innovation, innovation_covariance)
        if mahalanobis_squared > cfg.mahalanobis_gate:
            return np.inf
        overlap = iou_3d(track.box, detection)
        cost = (
            cfg.iou_3d_weight * (1.0 - overlap) +
            cfg.motion_weight * mahalanobis_squared /
            max(cfg.mahalanobis_gate, 1e-3))
        if (detection_observation is not None and
                detection_observation.camera_available and
                detection_observation.camera_visible):
            cost += cfg.camera_iou_weight * (
                1.0 - float(detection_observation.camera_iou))
        if (detection_observation is not None and
                detection_observation.depth_available):
            if detection_observation.depth_state == DepthState.CONSISTENT:
                depth_cost = 1.0 - float(np.clip(
                    detection_observation.depth_consistency, 0.0, 1.0))
            else:
                depth_cost = 1.0
            cost += cfg.depth_weight * depth_cost
        return float(cost) if cost <= cfg.maximum_cost else np.inf

    def _match(self, track_indices, detection_indices, detections,
               detection_observations):
        if not track_indices or not detection_indices:
            return []
        costs = np.full(
            (len(track_indices), len(detection_indices)), np.inf,
            dtype=np.float32)
        projected_covariances = {
            index: self.tracks[index].H @ self.tracks[index].P @
            self.tracks[index].H.T
            for index in track_indices}
        noise_scales = {
            index: self._measurement_noise_scale(detection_observations[index])
            for index in detection_indices}
        for row, track_index in enumerate(track_indices):
            for column, detection_index in enumerate(detection_indices):
                costs[row, column] = self._pair_cost(
                    self.tracks[track_index], detections[detection_index],
                    detection_observations[detection_index],
                    projected_covariances[track_index],
                    noise_scales[detection_index])
        finite = np.isfinite(costs)
        if not finite.any():
            return []
        safe = costs.copy()
        safe[~finite] = 1e6
        rows, columns = linear_sum_assignment(safe)
        return [
            (track_indices[row], detection_indices[column],
             float(costs[row, column]), "UNIFIED_ASSOCIATION")
            for row, column in zip(rows, columns)
            if np.isfinite(costs[row, column])]

    @staticmethod
    def _roi_coverage(front_roi, back_roi):
        left = max(float(front_roi[0]), float(back_roi[0]))
        top = max(float(front_roi[1]), float(back_roi[1]))
        right = min(float(front_roi[2]), float(back_roi[2]))
        bottom = min(float(front_roi[3]), float(back_roi[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        back_area = max(
            1e-3,
            (float(back_roi[2]) - float(back_roi[0])) *
            (float(back_roi[3]) - float(back_roi[1])))
        return float(intersection / back_area)

    def _find_pedestrian_occlusions(self, track_indices, track_observations):
        """Map hidden pedestrian indices to the nearer pedestrian and overlap."""
        cfg = self.association_config
        if not cfg.enable_occlusion_reasoning:
            return {}
        occlusions = {}
        for back_index in track_indices:
            back_track = self.tracks[back_index]
            back = track_observations[back_index]
            if (back_track.cls_id != 2 or back is None or
                    back.projected_roi is None or
                    back.depth_state != DepthState.OCCLUDED):
                continue
            best = None
            for front_index, front_track in enumerate(self.tracks):
                if front_index == back_index or front_track.cls_id != 2:
                    continue
                front = track_observations[front_index]
                if (front is None or front.projected_roi is None or
                        not (front.lidar_supported or front.camera_supported) or
                        front.object_range + cfg.occlusion_depth_margin >=
                        back.object_range):
                    continue
                coverage = self._roi_coverage(
                    front.projected_roi, back.projected_roi)
                if coverage < cfg.occlusion_overlap_threshold:
                    continue
                if best is None or coverage > best[1]:
                    best = (front_track.id, coverage)
            if best is not None:
                occlusions[back_index] = best
        return occlusions

    def _allow_spawn(self, observation):
        if observation is None or self._unavailable(observation):
            return True
        if observation.state in {
                ObservationClass.MULTIMODAL_STRONG.value,
                ObservationClass.LIDAR_DOMINANT.value}:
            return True
        # Camera-only evidence recovers 3-D tracks; it does not initialize them.
        if observation.state == ObservationClass.CAMERA_DOMINANT.value:
            return False
        both_observable = (
            observation.lidar_available and observation.lidar_visible and
            observation.camera_available and observation.camera_visible)
        return not both_observable

    def _static_fast_matches(self, detections):
        """Return unambiguous ego-compensated static track/detection pairs."""
        cfg = self.association_config
        if (not cfg.enable_static_fast_path or not len(detections) or
                self.frame_index % max(1, cfg.static_observation_interval) == 0):
            return []
        ego = self.current_ego_motion
        previous = self.previous_ego_motion
        ego_delta = math.hypot(
            float(ego.get("vx", 0.0)) - float(previous.get("vx", 0.0)),
            float(ego.get("vy", 0.0)) - float(previous.get("vy", 0.0)))
        if (abs(float(ego.get("yaw_rate", 0.0))) > cfg.static_fast_max_yaw_rate or
                ego_delta > cfg.static_fast_max_ego_delta):
            return []

        distances = np.full(
            (len(self.tracks), len(detections)), np.inf, dtype=np.float32)
        for track_index, track in enumerate(self.tracks):
            memory = self.lifecycle.memories.get(track.id)
            if (track.last_observation is None or track.is_dynamic or
                    track.static_time_sec < self.static_hold_time or
                    memory is None or memory.current_track_status !=
                    TrackStatus.CONFIRMED):
                continue
            gate = cfg.static_fast_match_distance * (
                0.65 if track.cls_id == 2 else 1.0)
            for detection_index, detection in enumerate(detections):
                if (track.cls_id > 0 and int(detection[7]) > 0 and
                        track.cls_id != int(detection[7])):
                    continue
                distance = float(np.linalg.norm(
                    track.box[:2] - detection[:2]))
                if distance <= gate:
                    distances[track_index, detection_index] = distance

        pairs = []
        for track_index, detection_index in enumerate(np.argmin(distances, axis=1)):
            distance = distances[track_index, detection_index]
            if (np.isfinite(distance) and
                    int(np.argmin(distances[:, detection_index])) == track_index):
                pairs.append((
                    track_index, int(detection_index), float(distance),
                    "STATIC_EGO_ASSOCIATION"))
        return pairs

    def update(self, detections, timestamp=None, ego_motion=None,
               observation_provider=None, detection_observations=None):
        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            detections = np.empty((0, 9), dtype=np.float32)
        self.frame_index += 1
        self.current_ego_motion = ego_motion or {
            "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        for track in self.tracks:
            track.predict(
                timestamp, ego_motion=self.current_ego_motion,
                ego_comp_gain=self.ego_motion_match_gain)
            track.was_matched = False

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        self.last_events = []
        static_matches = (
            self._static_fast_matches(detections)
            if observation_provider is not None and detection_observations is None
            else [])
        static_tracks = {item[0] for item in static_matches}
        static_detections = {item[1] for item in static_matches}
        unmatched_tracks = [i for i in unmatched_tracks if i not in static_tracks]
        unmatched_detections = [
            i for i in unmatched_detections if i not in static_detections]

        if detection_observations is None:
            detection_observations = [None] * len(detections)
            for track_index, detection_index, _, _ in static_matches:
                detection_observations[detection_index] = (
                    self.tracks[track_index].last_observation)
            if observation_provider is not None and unmatched_detections:
                observations = observation_provider(
                    detections[unmatched_detections, :7],
                    detections[unmatched_detections, 7].astype(np.int32))
                for detection_index, observation in zip(
                        unmatched_detections, observations):
                    detection_observations[detection_index] = observation

        matches = static_matches + self._match(
            unmatched_tracks, unmatched_detections, detections,
            detection_observations)
        matched_track_indices = {item[0] for item in matches}
        matched_detection_indices = {item[1] for item in matches}
        unmatched_tracks = [
            i for i in unmatched_tracks if i not in matched_track_indices]
        unmatched_detections = [
            i for i in unmatched_detections
            if i not in matched_detection_indices]

        track_observations = [None] * len(self.tracks)
        for track_index, detection_index, _, _ in matches:
            track_observations[track_index] = detection_observations[detection_index]
        if observation_provider is not None and unmatched_tracks:
            observations = observation_provider(
                np.asarray(
                    [self.tracks[index].box for index in unmatched_tracks],
                    dtype=np.float32),
                [self.tracks[index].cls_id for index in unmatched_tracks])
            for track_index, observation in zip(unmatched_tracks, observations):
                track_observations[track_index] = observation
        needs_occluders = any(
            self.tracks[index].cls_id == 2 and
            track_observations[index] is not None and
            track_observations[index].depth_state == DepthState.OCCLUDED
            for index in unmatched_tracks)
        if observation_provider is not None and needs_occluders:
            front_indices = [
                index for index in matched_track_indices
                if self.tracks[index].cls_id == 2]
            if front_indices:
                observations = observation_provider(
                    np.asarray(
                        [self.tracks[index].box for index in front_indices],
                        dtype=np.float32),
                    [self.tracks[index].cls_id for index in front_indices])
                for track_index, observation in zip(
                        front_indices, observations):
                    track_observations[track_index] = observation
        self.last_detection_observations = list(detection_observations)
        track_occlusions = self._find_pedestrian_occlusions(
            unmatched_tracks, track_observations)

        for track_index, detection_index, cost, stage in matches:
            track = self.tracks[track_index]
            observation = detection_observations[detection_index]
            weak = (
                observation is not None and observation.state ==
                ObservationClass.WEAK_OBSERVATION.value)
            if weak:
                detection = detections[detection_index]
                gain = float(np.clip(
                    self.association_config.weak_position_gain, 0.0, 1.0))
                track.box[:3] += gain * (detection[:3] - track.box[:3])
                track.x[:2, 0] = track.box[:2]
                track.record_observation(observation)
                memory = self.lifecycle.weak_matched(track, observation)
            else:
                track.update(
                    detections[detection_index], timestamp,
                    self.static_hold_time, self.static_yaw_alpha,
                    self.dynamic_yaw_alpha, ego_motion=self.current_ego_motion,
                    observation=observation,
                    measurement_noise_scale=self._measurement_noise_scale(
                        observation))
                memory = self.lifecycle.matched(track, observation)
            track.last_association_cost = cost
            self.last_events.append({
                "track_id": track.id,
                "detection_index": detection_index,
                "matched": True,
                "association_stage": stage,
                "association_cost": cost,
                "track_status": memory.current_track_status.value,
            })

        recovered_tracks = set()
        for track_index in unmatched_tracks:
            observation = track_observations[track_index]
            if (track_index not in track_occlusions and
                    self._supports_prediction(observation)):
                track = self.tracks[track_index]
                if observation.camera_depth_center is not None:
                    measured = np.asarray(
                        observation.camera_depth_center, dtype=np.float32)
                    delta = measured - track.box[:3]
                    distance = float(np.linalg.norm(delta))
                    maximum = self.association_config.camera_recovery_max_correction
                    if distance > maximum > 0.0:
                        delta *= maximum / distance
                    track.box[:3] += (
                        self.association_config.camera_recovery_position_gain * delta)
                    track.x[:2, 0] = track.box[:2]
                track.record_observation(observation)
                memory = self.lifecycle.recovered(track, observation)
                recovered_tracks.add(track_index)
                self.last_events.append({
                    "track_id": track.id,
                    "detection_index": -1,
                    "matched": False,
                    "association_stage": "SUPPORTED_PREDICTION",
                    "association_cost": float("nan"),
                    "track_status": memory.current_track_status.value,
                })

        for track_index in unmatched_tracks:
            if track_index in recovered_tracks:
                continue
            track = self.tracks[track_index]
            observation = track_observations[track_index]
            track.record_observation(observation)
            track.last_association_cost = float("nan")
            occlusion = track_occlusions.get(track_index)
            if occlusion is None:
                memory = self.lifecycle.missed(track, observation)
                stage = "PREDICTION_MISSED"
            else:
                memory = self.lifecycle.occluded(
                    track, observation, occlusion[0], occlusion[1],
                    self.association_config.occlusion_confirmation_frames,
                    self.association_config.occlusion_max_age)
                stage = "PREDICTION_OCCLUDED_BY_TRACK"
            self.last_events.append({
                "track_id": track.id,
                "detection_index": -1,
                "matched": False,
                "association_stage": stage,
                "association_cost": float("nan"),
                "track_status": memory.current_track_status.value,
            })

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            observation = detection_observations[detection_index]
            if (float(detection[8]) < self.min_spawn_score or
                    not self._allow_spawn(observation)):
                continue
            track = StaticCvTrack(
                detection, timestamp, self.expected_dt, self.confirm_time,
                self.static_speed_thresh, observation,
                self.observation_history_size)
            self.tracks.append(track)
            memory = self.lifecycle.create(track, observation)
            self.last_events.append({
                "track_id": track.id,
                "detection_index": detection_index,
                "matched": False,
                "spawned": True,
                "association_stage": "INITIALIZATION",
                "association_cost": float("nan"),
                "track_status": memory.current_track_status.value,
            })

        survivors = []
        for track in self.tracks:
            memory = self.lifecycle.memories.get(track.id)
            if (memory is not None and
                    memory.current_track_status == TrackStatus.TERMINATED):
                self.lifecycle.remove(track.id)
            else:
                survivors.append(track)
        self.tracks = survivors

        event_by_track = {event["track_id"]: event for event in self.last_events}
        results = []
        debug_results = []
        for track in self.tracks:
            memory = self.lifecycle.memories.get(track.id)
            if memory is None:
                continue
            event = event_by_track.get(track.id, {})
            result = {
                "id": track.id,
                "box": track.box.copy(),
                "label": track.cls_id,
                "score": track.score,
                "miss_age": track.time_since_update_sec,
                "age_sec": track.age_sec,
                "speed": track.speed,
                "is_dynamic": track.is_dynamic,
                "is_static": track.static_time_sec >= self.static_hold_time,
                "was_detected": bool(
                    event.get("matched", False) or event.get("spawned", False)),
                "observation": track.last_observation,
                "observation_state": memory.current_observation_state,
                "observation_confidence": memory.track_confidence,
                "track_confidence": memory.track_confidence,
                "track_status": memory.current_track_status.value,
                "hit_count": memory.hit_count,
                "miss_count": memory.miss_count,
                "recovery_count": memory.recovery_count,
                "occlusion_count": memory.occlusion_count,
                "occluded_by_track_id": memory.occluded_by_track_id,
                "occlusion_ratio": memory.occlusion_ratio,
                "camera_support_age": track.camera_support_age,
                "lidar_support_age": track.lidar_support_age,
                "association_cost": event.get("association_cost", float("nan")),
                "association_stage": event.get("association_stage", "NONE"),
                "matched_detection_id": event.get("detection_index", -1),
                "matched": bool(event.get("matched", False)),
            }
            debug_results.append(result)
            observation = track.last_observation
            camera_depth_verified = (
                observation is not None and
                observation.camera_supported and
                observation.depth_state == DepthState.CONSISTENT)
            output_supported = (
                result["was_detected"] or camera_depth_verified or
                memory.miss_count <=
                self.association_config.output_prediction_max_misses)
            if (memory.current_track_status in {
                    TrackStatus.CONFIRMED, TrackStatus.RECOVERING,
                    TrackStatus.OCCLUDED} and
                    output_supported):
                results.append(result)
        self.last_debug_tracks = debug_results
        self.previous_ego_motion = dict(self.current_ego_motion)
        return results
