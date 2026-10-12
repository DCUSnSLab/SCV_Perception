"""Four-stage association that preserves Jay track prediction and IDs."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from pcdet_tracker.jay_tracker import KittiHzTracker, TimeAwareTrack, bev_iou

from .observation_state import (
    DepthState, ObservationClass, TrackStatus)
from .track_lifecycle import LifecycleConfig, TrackLifecycleManager


@dataclass
class AssociationConfig:
    center_distance_weight: float = 1.0
    iou_3d_weight: float = 1.0
    motion_weight: float = 0.5
    camera_iou_weight: float = 0.5
    depth_weight: float = 0.25
    class_penalty: float = 10.0
    center_distance_gate: float = 4.0
    iou_3d_gate: float = 0.0
    motion_gate: float = 6.0
    maximum_cost: float = 5.0
    enable_cascaded_association: bool = True
    camera_recovery_position_gain: float = 0.25
    camera_recovery_max_correction: float = 1.0
    enable_weak_track_continuity: bool = True
    weak_center_distance_gate: float = 1.5
    weak_iou_3d_gate: float = 0.10
    weak_continuity_max_age: int = 3


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
    """Reuse the baseline KF/track object and replace only association/lifecycle."""

    def __init__(self, *args, association_config=None, lifecycle_config=None,
                 **kwargs):
        super().__init__(*args, adaptive_policy=None, **kwargs)
        self.association_config = association_config or AssociationConfig()
        self.lifecycle = TrackLifecycleManager(
            lifecycle_config or LifecycleConfig())
        self.last_debug_tracks = []
        self.last_detection_observations = []

    @staticmethod
    def _unavailable(observation):
        return observation is None or (
            not observation.lidar_available and
            not observation.camera_available)

    def _pair_cost(self, track, detection, track_observation,
                   detection_observation, multimodal):
        cfg = self.association_config
        class_mismatch = (
            track.cls_id > 0 and int(detection[7]) > 0 and
            track.cls_id != int(detection[7]))
        if class_mismatch and cfg.class_penalty >= cfg.maximum_cost:
            return np.inf
        center = float(np.linalg.norm(track.box[:3] - detection[:3]))
        if center > cfg.center_distance_gate:
            return np.inf
        overlap = iou_3d(track.box, detection)
        if overlap < cfg.iou_3d_gate:
            return np.inf
        innovation = detection[:2] - track.box[:2]
        covariance = np.maximum(np.diag(track.P)[:2], 1e-3)
        motion = float(math.sqrt(np.sum(innovation * innovation / covariance)))
        if motion > cfg.motion_gate:
            return np.inf
        cost = (
            cfg.center_distance_weight * center / max(cfg.center_distance_gate, 1e-3) +
            cfg.iou_3d_weight * (1.0 - overlap) +
            cfg.motion_weight * motion / max(cfg.motion_gate, 1e-3))
        if class_mismatch:
            cost += cfg.class_penalty
        if multimodal and detection_observation is not None:
            cost += cfg.camera_iou_weight * (
                1.0 - float(detection_observation.camera_iou))
            if detection_observation.depth_state == DepthState.CONSISTENT:
                depth_cost = 1.0 - float(np.clip(
                    detection_observation.depth_consistency, 0.0, 1.0))
            elif detection_observation.depth_state == DepthState.UNAVAILABLE:
                depth_cost = 0.5
            else:
                depth_cost = 1.0
            cost += cfg.depth_weight * depth_cost
            if (track_observation is not None and
                    track_observation.camera_detection_index >= 0 and
                    detection_observation.camera_detection_index >= 0 and
                    track_observation.camera_detection_index !=
                    detection_observation.camera_detection_index):
                return np.inf
        return float(cost) if cost <= cfg.maximum_cost else np.inf

    def _match_stage(self, track_indices, detection_indices, detections,
                     track_observations, detection_observations, multimodal,
                     stage_name):
        if not track_indices or not detection_indices:
            return []
        costs = np.full(
            (len(track_indices), len(detection_indices)), np.inf,
            dtype=np.float32)
        for row, track_index in enumerate(track_indices):
            for column, detection_index in enumerate(detection_indices):
                costs[row, column] = self._pair_cost(
                    self.tracks[track_index], detections[detection_index],
                    track_observations[track_index],
                    detection_observations[detection_index], multimodal)
        finite = np.isfinite(costs)
        if not finite.any():
            return []
        safe = costs.copy()
        safe[~finite] = 1e6
        rows, columns = linear_sum_assignment(safe)
        return [
            (track_indices[row], detection_indices[column],
             float(costs[row, column]), stage_name)
            for row, column in zip(rows, columns)
            if np.isfinite(costs[row, column])]

    def _match_weak_continuity(self, track_indices, detection_indices,
                               detections, detection_observations):
        """Attach weak detections to recent established tracks without a KF hit."""
        cfg = self.association_config
        if not cfg.enable_weak_track_continuity:
            return []
        eligible_tracks = []
        for track_index in track_indices:
            memory = self.lifecycle.memories.get(self.tracks[track_index].id)
            if (memory is not None and
                    memory.current_track_status in {
                        TrackStatus.CONFIRMED, TrackStatus.RECOVERING} and
                    memory.miss_count <= cfg.weak_continuity_max_age):
                eligible_tracks.append(track_index)
        weak_detections = [
            index for index in detection_indices
            if (detection_observations[index] is not None and
                detection_observations[index].state ==
                ObservationClass.WEAK_OBSERVATION.value)]
        if not eligible_tracks or not weak_detections:
            return []

        costs = np.full(
            (len(eligible_tracks), len(weak_detections)), np.inf,
            dtype=np.float32)
        for row, track_index in enumerate(eligible_tracks):
            track = self.tracks[track_index]
            for column, detection_index in enumerate(weak_detections):
                detection = detections[detection_index]
                if track.cls_id != int(detection[7]):
                    continue
                distance = float(np.linalg.norm(
                    track.box[:3] - detection[:3]))
                overlap = iou_3d(track.box, detection)
                if (distance > cfg.weak_center_distance_gate or
                        overlap < cfg.weak_iou_3d_gate):
                    continue
                costs[row, column] = (
                    distance / max(cfg.weak_center_distance_gate, 1e-3) +
                    1.0 - overlap)
        finite = np.isfinite(costs)
        if not finite.any():
            return []
        safe = costs.copy()
        safe[~finite] = 1e6
        rows, columns = linear_sum_assignment(safe)
        return [
            (eligible_tracks[row], weak_detections[column],
             float(costs[row, column]), "STAGE_4_WEAK_CONTINUITY")
            for row, column in zip(rows, columns)
            if np.isfinite(costs[row, column])]

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

    def update(self, detections, timestamp=None, ego_motion=None,
               observation_provider=None, detection_observations=None):
        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            detections = np.empty((0, 9), dtype=np.float32)
        self.current_ego_motion = ego_motion or {
            "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        for track in self.tracks:
            track.predict(
                timestamp, ego_motion=self.current_ego_motion,
                ego_comp_gain=self.ego_motion_match_gain)
            track.was_matched = False

        if observation_provider is not None:
            boxes = np.asarray(
                [track.box for track in self.tracks], dtype=np.float32
            ) if self.tracks else np.empty((0, 7), dtype=np.float32)
            track_observations = observation_provider(
                boxes, [track.cls_id for track in self.tracks])
            if detection_observations is None:
                detection_observations = observation_provider(
                    detections[:, :7], detections[:, 7].astype(np.int32))
        else:
            track_observations = [None] * len(self.tracks)
        if detection_observations is None:
            detection_observations = [None] * len(detections)
        self.last_detection_observations = list(detection_observations)

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        matches = []
        self.last_events = []
        if self.association_config.enable_cascaded_association:
            stage1_detections = [
                index for index in unmatched_detections
                if (detection_observations[index] is not None and
                    detection_observations[index].state ==
                    ObservationClass.MULTIMODAL_STRONG.value)]
            stage1 = self._match_stage(
                unmatched_tracks, stage1_detections, detections,
                track_observations, detection_observations, True, "STAGE_1")
            matches.extend(stage1)
            used_tracks = {item[0] for item in stage1}
            used_detections = {item[1] for item in stage1}
            unmatched_tracks = [i for i in unmatched_tracks if i not in used_tracks]
            unmatched_detections = [
                i for i in unmatched_detections if i not in used_detections]

            stage2_detections = [
                index for index in unmatched_detections
                if (self._unavailable(detection_observations[index]) or
                    detection_observations[index].state in {
                        ObservationClass.MULTIMODAL_STRONG.value,
                        ObservationClass.LIDAR_DOMINANT.value})]
            stage2 = self._match_stage(
                unmatched_tracks, stage2_detections, detections,
                track_observations, detection_observations, False, "STAGE_2")
            matches.extend(stage2)
            used_tracks = {item[0] for item in stage2}
            used_detections = {item[1] for item in stage2}
            unmatched_tracks = [i for i in unmatched_tracks if i not in used_tracks]
            unmatched_detections = [
                i for i in unmatched_detections if i not in used_detections]
        else:
            matches = self._match_stage(
                unmatched_tracks, unmatched_detections, detections,
                track_observations, detection_observations, False, "BASELINE")
            used_tracks = {item[0] for item in matches}
            used_detections = {item[1] for item in matches}
            unmatched_tracks = [i for i in unmatched_tracks if i not in used_tracks]
            unmatched_detections = [
                i for i in unmatched_detections if i not in used_detections]

        weak_matches = self._match_weak_continuity(
            unmatched_tracks, unmatched_detections, detections,
            detection_observations)
        weak_track_indices = {item[0] for item in weak_matches}
        weak_detection_indices = {item[1] for item in weak_matches}
        unmatched_tracks = [
            i for i in unmatched_tracks if i not in weak_track_indices]
        unmatched_detections = [
            i for i in unmatched_detections if i not in weak_detection_indices]
        for track_index, detection_index, cost, stage in weak_matches:
            track = self.tracks[track_index]
            observation = detection_observations[detection_index]
            track.record_observation(observation)
            track.last_association_cost = cost
            memory = self.lifecycle.missed(track, observation)
            self.last_events.append({
                "track_id": track.id,
                "detection_index": detection_index,
                "matched": False,
                "association_stage": stage,
                "association_cost": cost,
                "track_status": memory.current_track_status.value,
            })

        for track_index, detection_index, cost, stage in matches:
            track = self.tracks[track_index]
            observation = detection_observations[detection_index]
            track.update(
                detections[detection_index], timestamp,
                self.static_hold_time, self.static_yaw_alpha,
                self.dynamic_yaw_alpha, ego_motion=self.current_ego_motion,
                observation=observation,
                measurement_noise_scale=self._measurement_noise_scale(observation))
            track.last_association_cost = cost
            memory = self.lifecycle.matched(track, observation)
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
            if (observation is not None and
                    observation.state == ObservationClass.CAMERA_DOMINANT.value and
                    observation.camera_supported and
                    observation.depth_state == DepthState.CONSISTENT):
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
                    "association_stage": "STAGE_3",
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
            memory = self.lifecycle.missed(track, observation)
            self.last_events.append({
                "track_id": track.id,
                "detection_index": -1,
                "matched": False,
                "association_stage": "STAGE_4",
                "association_cost": float("nan"),
                "track_status": memory.current_track_status.value,
            })

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            observation = detection_observations[detection_index]
            if (float(detection[8]) < self.min_spawn_score or
                    not self._allow_spawn(observation) or
                    not self._should_spawn_track(detection, observation)):
                continue
            track = TimeAwareTrack(
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
                "camera_support_age": track.camera_support_age,
                "lidar_support_age": track.lidar_support_age,
                "association_cost": event.get("association_cost", float("nan")),
                "association_stage": event.get("association_stage", "NONE"),
                "matched_detection_id": event.get("detection_index", -1),
                "matched": bool(event.get("matched", False)),
            }
            debug_results.append(result)
            if memory.current_track_status not in {
                    TrackStatus.TENTATIVE, TrackStatus.SUPPRESSED,
                    TrackStatus.TERMINATED}:
                results.append(result)
        self.last_debug_tracks = debug_results
        return results
