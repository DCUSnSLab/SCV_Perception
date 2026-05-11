#!/usr/bin/env python3
import argparse
import csv
from collections import deque
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
from tracking_msgs.msg import DetectedObject, DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray

from pcdet_tracker.tracker_csv_writer import TrackerCsvWriter


TRACK_COLORS = [
    (0.10, 0.80, 1.00),
    (1.00, 0.85, 0.20),
    (0.20, 1.00, 0.35),
    (1.00, 0.35, 0.25),
    (0.85, 0.25, 1.00),
]

DETECTION_COLOR = (0.95, 0.95, 0.95)

DEFAULT_TRACK_BOX_SIZES = {
    1: (4.67, 2.04, 1.80),   # Vehicle / Car: length, width, height
    2: (0.98, 0.77, 1.70),   # Pedestrian
    3: (1.76, 0.60, 1.73),   # Cyclist
}


class PerfMonitor:
    def __init__(self, node, name, log_interval_sec, warmup_frames, csv_output_dir):
        self.node = node
        self.name = name
        self.log_interval_sec = max(float(log_interval_sec), 0.1)
        self.warmup_frames = max(int(warmup_frames), 0)
        self.csv_file = None
        self.csv_writer = None
        csv_dir = Path(csv_output_dir).expanduser().resolve()
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_dir / f"{name}_perf.csv"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_index",
            "msg_stamp",
            "total_ms",
            "input_count",
            "output_count",
            "parse_ms",
            "csv_det_ms",
            "ego_ms",
            "track_ms",
            "csv_track_ms",
            "history_ms",
            "markers_ms",
        ])
        self.reset()

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

    def reset(self):
        self.window_start = time.perf_counter()
        self.frames = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.stage_totals = {}
        self.input_total = 0
        self.output_total = 0
        self.first_msg_stamp = None
        self.last_msg_stamp = None

    def add(self, frame_index, total_ms, stage_ms, input_count=0, output_count=0, msg_stamp=None):
        if self.warmup_frames > 0:
            self.warmup_frames -= 1
            return

        self.csv_writer.writerow([
            int(frame_index),
            "" if msg_stamp is None else f"{float(msg_stamp):.9f}",
            f"{float(total_ms):.6f}",
            int(input_count),
            int(output_count),
            f"{float(stage_ms.get('parse', 0.0)):.6f}",
            f"{float(stage_ms.get('csv_det', 0.0)):.6f}",
            f"{float(stage_ms.get('ego', 0.0)):.6f}",
            f"{float(stage_ms.get('track', 0.0)):.6f}",
            f"{float(stage_ms.get('csv_track', 0.0)):.6f}",
            f"{float(stage_ms.get('history', 0.0)):.6f}",
            f"{float(stage_ms.get('markers', 0.0)):.6f}",
        ])
        self.csv_file.flush()
        self.frames += 1
        self.total_ms += float(total_ms)
        self.max_ms = max(self.max_ms, float(total_ms))
        self.input_total += int(input_count)
        self.output_total += int(output_count)
        for key, value in stage_ms.items():
            self.stage_totals[key] = self.stage_totals.get(key, 0.0) + float(value)

        if msg_stamp is not None:
            if self.first_msg_stamp is None:
                self.first_msg_stamp = float(msg_stamp)
            self.last_msg_stamp = float(msg_stamp)

        elapsed = time.perf_counter() - self.window_start
        if elapsed < self.log_interval_sec:
            return

        avg_ms = self.total_ms / max(self.frames, 1)
        wall_fps = self.frames / max(elapsed, 1e-6)
        msg_fps = 0.0
        if self.first_msg_stamp is not None and self.last_msg_stamp is not None and self.last_msg_stamp > self.first_msg_stamp:
            msg_fps = (self.frames - 1) / max(self.last_msg_stamp - self.first_msg_stamp, 1e-6)
        stage_summary = " ".join(
            f"{key}={self.stage_totals[key] / max(self.frames, 1):.1f}ms"
            for key in stage_ms.keys()
        )
        avg_in = self.input_total / max(self.frames, 1)
        avg_out = self.output_total / max(self.frames, 1)
        self.node.get_logger().info(
            f"[perf:{self.name}] frames={self.frames} wall_fps={wall_fps:.2f} "
            f"msg_fps={msg_fps:.2f} avg_total={avg_ms:.1f}ms max_total={self.max_ms:.1f}ms "
            f"avg_in={avg_in:.1f} avg_out={avg_out:.1f} {stage_summary}"
        )
        self.reset()


def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def angle_diff(target_angle, reference_angle):
    return wrap_angle(target_angle - reference_angle)


def closest_yaw(reference_angle, measured_angle):
    candidates = (
        wrap_angle(measured_angle),
        wrap_angle(measured_angle + np.pi),
    )
    return min(candidates, key=lambda angle: abs(angle_diff(angle, reference_angle)))


def angle_lerp(prev_angle, new_angle, alpha):
    return wrap_angle(prev_angle + alpha * angle_diff(new_angle, prev_angle))


def rotation_matrix_2d(angle):
    cos_yaw = math.cos(angle)
    sin_yaw = math.sin(angle)
    return np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)


def compensate_body_frame_motion(center_xy, yaw, dt, ego_motion, gain=1.0):
    vx = float(ego_motion.get("vx", 0.0))
    vy = float(ego_motion.get("vy", 0.0))
    yaw_rate = float(ego_motion.get("yaw_rate", 0.0))
    delta_local = np.array([vx * dt, vy * dt], dtype=np.float32) * float(gain)
    delta_yaw = yaw_rate * dt * float(gain)
    compensated_xy = rotation_matrix_2d(-delta_yaw) @ (center_xy.astype(np.float32) - delta_local)
    compensated_yaw = wrap_angle(float(yaw) - delta_yaw)
    return compensated_xy, compensated_yaw


def box_corners_bev(center_x, center_y, length, width, yaw):
    half_l = length * 0.5
    half_w = width * 0.5
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return corners @ rot.T + np.array([center_x, center_y], dtype=np.float32)


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside_edge(point, edge_start, edge_end):
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= 0.0


def segment_intersection(p1, p2, q1, q2):
    s = np.vstack([p1, p2, q1, q2]).astype(np.float32)
    h = np.hstack([s, np.ones((4, 1), dtype=np.float32)])
    line1 = np.cross(h[0], h[1])
    line2 = np.cross(h[2], h[3])
    x, y, z = np.cross(line1, line2)
    if abs(z) < 1e-6:
        return p2
    return np.array([x / z, y / z], dtype=np.float32)


def polygon_clip(subject, clipper):
    output = subject.copy()
    for idx in range(len(clipper)):
        edge_start = clipper[idx]
        edge_end = clipper[(idx + 1) % len(clipper)]
        input_list = output
        if len(input_list) == 0:
            break
        output = []
        prev = input_list[-1]
        for curr in input_list:
            curr_inside = inside_edge(curr, edge_start, edge_end)
            prev_inside = inside_edge(prev, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(segment_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(segment_intersection(prev, curr, edge_start, edge_end))
            prev = curr
        output = np.asarray(output, dtype=np.float32)
    return output


def bev_iou(box_a, box_b):
    poly_a = box_corners_bev(box_a[0], box_a[1], box_a[2], box_a[3], box_a[4])
    poly_b = box_corners_bev(box_b[0], box_b[1], box_b[2], box_b[3], box_b[4])
    inter_poly = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    if inter_area <= 0.0:
        return 0.0
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    union = max(area_a + area_b - inter_area, 1e-6)
    return float(inter_area / union)


def apply_center_nms(boxes, scores, labels, dist_thresh):
    if len(boxes) == 0:
        return boxes, scores, labels

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        survivors = []
        for next_idx in order[1:]:
            if labels[next_idx] != labels[idx]:
                survivors.append(next_idx)
                continue
            dist = np.linalg.norm(boxes[next_idx, :2] - boxes[idx, :2])
            iou = bev_iou(
                [boxes[idx, 0], boxes[idx, 1], boxes[idx, 3], boxes[idx, 4], boxes[idx, 6]],
                [boxes[next_idx, 0], boxes[next_idx, 1], boxes[next_idx, 3], boxes[next_idx, 4], boxes[next_idx, 6]],
            )
            if dist > dist_thresh and iou < 0.15:
                survivors.append(next_idx)
        order = np.asarray(survivors, dtype=np.int64)

    return boxes[keep], scores[keep], labels[keep]


class RosModelDataset:
    def __init__(self, cfg_obj, pc_range):
        self.class_names = cfg_obj.CLASS_NAMES
        self.point_feature_encoder = type("Encoder", (), {"num_point_features": 4})()
        self.point_cloud_range = pc_range
        self.depth_downsample_factor = None
        self.dataset_cfg = cfg_obj.DATA_CONFIG
        self.voxel_size = None
        for processor in cfg_obj.DATA_CONFIG.DATA_PROCESSOR:
            if processor["NAME"] == "transform_points_to_voxels":
                self.voxel_size = np.array(processor["VOXEL_SIZE"], dtype=np.float32)
                break
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int64)


class TimeAwareTrack:
    _next_id = 0

    def __init__(self, det, timestamp, expected_dt, confirm_time, static_speed_thresh):
        self.id = TimeAwareTrack._next_id
        TimeAwareTrack._next_id += 1

        self.expected_dt = float(expected_dt)
        self.confirm_time = float(confirm_time)
        self.static_speed_thresh = float(static_speed_thresh)

        self.last_timestamp = timestamp
        self.age_sec = 0.0
        self.time_since_update_sec = 0.0
        self.hits = 1
        self.state = "tentative"

        self.x = np.array([det[0], det[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        self.P = np.eye(4, dtype=np.float32)
        self.F = np.eye(4, dtype=np.float32)
        self.H = np.zeros((2, 4), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = np.eye(4, dtype=np.float32) * 0.02
        self.R = np.eye(2, dtype=np.float32) * 0.20

        self.box = det[:7].astype(np.float32).copy()
        self.cls_id = int(det[7])
        self.score = float(det[8])
        self.stable_yaw = wrap_angle(float(det[6]))
        self.static_time_sec = 0.0
        self.class_scores = {self.cls_id: self.score}
        self.anchor_xy = self.box[:2].astype(np.float32).copy()
        self.last_ego_motion = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        self.center_history = deque(maxlen=6)
        self.relative_velocity = np.zeros(2, dtype=np.float32)
        self.expected_static_relative_velocity = np.zeros(2, dtype=np.float32)
        self.object_velocity = np.zeros(2, dtype=np.float32)
        self.static_residual_velocity = np.zeros(2, dtype=np.float32)
        self.static_residual_speed = 0.0
        self.dynamic_speed_thresh = max(1.2, 2.0 * self.static_speed_thresh)
        self.dynamic_votes = 0
        self._push_center(timestamp, self.box[:2])

    def _push_center(self, timestamp, center_xy):
        if timestamp is None:
            return
        self.center_history.append((float(timestamp), np.asarray(center_xy, dtype=np.float32).copy()))

    def _update_motion_state(self, timestamp, ego_motion):
        if len(self.center_history) < 2:
            self.relative_velocity = self.x[2:4, 0].astype(np.float32).copy()
            self.expected_static_relative_velocity = np.zeros(2, dtype=np.float32)
            self.object_velocity = self.relative_velocity.copy()
            self.static_residual_velocity = self.object_velocity.copy()
            self.static_residual_speed = float(np.linalg.norm(self.static_residual_velocity))
            return

        t0, p0 = self.center_history[-2]
        t1, p1 = self.center_history[-1]
        dt = float(t1 - t0)
        if dt <= 1e-3:
            return

        self.relative_velocity = (p1 - p0) / dt
        ego_motion = ego_motion or {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        ego_v = np.array(
            [float(ego_motion.get("vx", 0.0)), float(ego_motion.get("vy", 0.0))],
            dtype=np.float32,
        )
        ego_speed = float(np.linalg.norm(ego_v))
        yaw_rate = float(ego_motion.get("yaw_rate", 0.0))
        rot_comp = np.array([yaw_rate * p1[1], -yaw_rate * p1[0]], dtype=np.float32)
        self.expected_static_relative_velocity = -ego_v + rot_comp
        measured_object_velocity = self.relative_velocity - self.expected_static_relative_velocity
        self.static_residual_velocity = measured_object_velocity.astype(np.float32)
        self.static_residual_speed = float(np.linalg.norm(self.static_residual_velocity))
        self.object_velocity = 0.65 * self.object_velocity + 0.35 * measured_object_velocity

        object_speed = float(np.linalg.norm(self.object_velocity))
        dynamic_threshold = max(self.dynamic_speed_thresh, self.static_speed_thresh + 0.20 * ego_speed)
        static_threshold = max(self.static_speed_thresh, 0.20 + 0.08 * ego_speed)
        if object_speed >= dynamic_threshold:
            self.dynamic_votes = min(self.dynamic_votes + 1, 6)
        elif self.static_residual_speed < static_threshold:
            self.dynamic_votes = max(self.dynamic_votes - 1, 0)

    def _set_dt(self, dt):
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.Q = np.eye(4, dtype=np.float32) * 0.02
        self.Q[0, 0] = self.Q[1, 1] = max(0.02, 0.08 * dt)
        self.Q[2, 2] = self.Q[3, 3] = max(0.02, 0.18 * dt)

    def predict(self, timestamp, ego_motion=None, ego_comp_gain=1.0):
        if self.last_timestamp is None or timestamp is None:
            dt = self.expected_dt
        else:
            dt = float(timestamp - self.last_timestamp)
            if dt <= 1e-3:
                dt = self.expected_dt
            dt = min(max(dt, 1e-2), max(1.0, self.expected_dt * 3.0))

        self._set_dt(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        ego_motion = ego_motion or {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        compensated_xy, compensated_yaw = compensate_body_frame_motion(
            self.x[:2, 0], self.box[6], dt, ego_motion, gain=ego_comp_gain
        )
        if self.is_dynamic:
            compensated_xy = compensated_xy + self.object_velocity * dt
        self.x[0, 0] = float(compensated_xy[0])
        self.x[1, 0] = float(compensated_xy[1])
        rotated_velocity = rotation_matrix_2d(
            -float(ego_motion.get("yaw_rate", 0.0)) * dt * float(ego_comp_gain)
        ) @ self.x[2:4, 0]
        if self.is_dynamic:
            self.x[2:4, 0] = 0.45 * rotated_velocity + 0.55 * self.object_velocity
        else:
            self.x[2:4, 0] = rotated_velocity
        self.anchor_xy, _ = compensate_body_frame_motion(
            self.anchor_xy, self.stable_yaw, dt, ego_motion, gain=ego_comp_gain
        )

        self.box[0] = float(self.x[0, 0])
        self.box[1] = float(self.x[1, 0])
        self.box[6] = compensated_yaw
        self.stable_yaw = compensated_yaw
        self.age_sec += dt
        self.time_since_update_sec += dt
        self.last_timestamp = timestamp
        self.last_ego_motion = dict(ego_motion)
        return dt

    def update(self, det, timestamp, static_hold_time, static_yaw_alpha, dynamic_yaw_alpha, ego_motion=None):
        measurement = np.array([det[0], det[1]], dtype=np.float32).reshape(2, 1)
        innovation = measurement - (self.H @ self.x)
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)

        prev_xy = self.x[:2, 0].astype(np.float32).copy()
        self.x = self.x + kalman_gain @ innovation
        self.P = (np.eye(4, dtype=np.float32) - kalman_gain @ self.H) @ self.P

        self.box[0] = float(self.x[0, 0])
        self.box[1] = float(self.x[1, 0])
        self.box[2] = 0.70 * self.box[2] + 0.30 * float(det[2])
        self.box[3:6] = 0.80 * self.box[3:6] + 0.20 * det[3:6]
        self._push_center(timestamp, det[:2])
        self._update_motion_state(timestamp, ego_motion)
        self.x[2:4, 0] = 0.35 * self.x[2:4, 0] + 0.65 * self.object_velocity

        vel = self.x[2:4, 0].astype(np.float32)
        speed = float(np.linalg.norm(vel))
        center_shift = float(np.linalg.norm(self.x[:2, 0] - prev_xy))
        ego_motion = ego_motion or {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        ego_speed = math.hypot(
            float(ego_motion.get("vx", 0.0)),
            float(ego_motion.get("vy", 0.0)),
        )
        adaptive_static_thresh = max(self.static_speed_thresh, 0.20 + 0.08 * ego_speed)
        if ego_speed > 0.25 or abs(float(ego_motion.get("yaw_rate", 0.0))) > 0.12:
            is_motion_static = self.static_residual_speed < adaptive_static_thresh
        else:
            is_motion_static = self.static_residual_speed < adaptive_static_thresh and center_shift < 0.35

        if is_motion_static and speed < max(self.dynamic_speed_thresh, adaptive_static_thresh * 2.0):
            self.static_time_sec += self.expected_dt
        else:
            self.static_time_sec = 0.0

        measured_yaw = closest_yaw(self.stable_yaw, float(det[6]))
        is_static = self.static_time_sec >= static_hold_time
        yaw_alpha = static_yaw_alpha if is_static else dynamic_yaw_alpha
        self.box[6] = angle_lerp(self.stable_yaw if is_static else float(self.box[6]), measured_yaw, yaw_alpha)
        self.stable_yaw = self.box[6]
        anchor_alpha = 0.05 if is_static else 0.20
        self.anchor_xy = (1.0 - anchor_alpha) * self.anchor_xy + anchor_alpha * self.box[:2]

        self.cls_id = self._update_semantics(int(det[7]), float(det[8]))
        self.score = max(float(det[8]), 0.75 * self.score + 0.25 * float(det[8]))
        self.hits += 1
        self.time_since_update_sec = 0.0
        self.last_timestamp = timestamp
        if self.age_sec >= self.confirm_time or self.hits >= 2:
            self.state = "confirmed"

    def _update_semantics(self, cls_id, score):
        self.class_scores[cls_id] = self.class_scores.get(cls_id, 0.0) * 0.8 + score
        best_cls = max(self.class_scores.items(), key=lambda item: item[1])[0]
        return int(best_cls)

    @property
    def is_confirmed(self):
        return self.state == "confirmed"

    @property
    def speed(self):
        return float(np.linalg.norm(self.x[2:4, 0]))

    @property
    def is_dynamic(self):
        return self.dynamic_votes >= 2


class KittiHzTracker:
    def __init__(
        self,
        expected_dt=0.116,
        max_age_frames=5,
        confirm_frames=2,
        match_distance=2.4,
        min_spawn_score=0.20,
        static_speed_thresh=0.60,
        static_yaw_alpha=0.04,
        dynamic_yaw_alpha=0.20,
        ego_motion_match_gain=1.0,
        ego_motion_assoc_bonus=0.35,
    ):
        self.expected_dt = float(expected_dt)
        self.max_age_sec = float(max_age_frames) * self.expected_dt
        self.confirm_time = float(confirm_frames) * self.expected_dt
        self.static_hold_time = float(confirm_frames) * self.expected_dt
        self.match_distance = float(match_distance)
        self.min_spawn_score = float(min_spawn_score)
        self.static_speed_thresh = float(static_speed_thresh)
        self.static_yaw_alpha = float(static_yaw_alpha)
        self.dynamic_yaw_alpha = float(dynamic_yaw_alpha)
        self.ego_motion_match_gain = float(ego_motion_match_gain)
        self.ego_motion_assoc_bonus = float(ego_motion_assoc_bonus)
        self.tracks = []
        self.current_ego_motion = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}

    def _cost(self, track, det):
        ego_speed = math.hypot(
            float(self.current_ego_motion.get("vx", 0.0)),
            float(self.current_ego_motion.get("vy", 0.0)),
        )
        yaw_rate = abs(float(self.current_ego_motion.get("yaw_rate", 0.0)))
        dynamic_bonus = max(0.0, track.speed - self.static_speed_thresh) * 0.45
        effective_match_distance = (
            self.match_distance
            + self.ego_motion_assoc_bonus * ego_speed
            + 1.5 * yaw_rate
            + dynamic_bonus
        )
        det_cls = int(det[7])
        if track.cls_id > 0 and det_cls > 0 and track.cls_id != det_cls:
            class_penalty = 0.6
        else:
            class_penalty = 0.0

        xy_dist = float(np.linalg.norm(track.box[:2] - det[:2]))
        if xy_dist > effective_match_distance:
            return np.inf

        yaw_det = closest_yaw(track.stable_yaw, float(det[6]))
        yaw_diff = abs(angle_diff(yaw_det, track.stable_yaw))
        size_diff = float(np.linalg.norm(track.box[3:6] - det[3:6]) / max(np.linalg.norm(track.box[3:6]), 1e-3))
        anchor_dist = float(np.linalg.norm(track.anchor_xy - det[:2]))
        iou = bev_iou(
            [track.box[0], track.box[1], track.box[3], track.box[4], track.box[6]],
            [det[0], det[1], det[3], det[4], yaw_det],
        )
        motion_dir_penalty = 0.0
        if track.is_dynamic and track.speed > 1e-3:
            motion_yaw = math.atan2(float(track.x[3, 0]), float(track.x[2, 0]))
            motion_dir_penalty = 0.18 * abs(angle_diff(yaw_det, motion_yaw))

        if size_diff > 0.50:
            return np.inf
        if track.speed < self.static_speed_thresh and anchor_dist > max(1.2, 0.5 * effective_match_distance):
            return np.inf

        return (
            xy_dist
            + 0.6 * anchor_dist
            + 0.8 * (1.0 - iou)
            + 0.20 * yaw_diff
            + 0.25 * size_diff
            + motion_dir_penalty
            + class_penalty
        )

    def _should_spawn_track(self, det):
        for track in self.tracks:
            xy_dist = float(np.linalg.norm(track.box[:2] - det[:2]))
            iou = bev_iou(
                [track.box[0], track.box[1], track.box[3], track.box[4], track.box[6]],
                [det[0], det[1], det[3], det[4], det[6]],
            )
            if xy_dist < max(1.0, self.match_distance * 0.5) or iou > 0.10:
                return False
        return True

    def update(self, detections, timestamp=None, ego_motion=None):
        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            detections = np.empty((0, 9), dtype=np.float32)
        self.current_ego_motion = ego_motion or {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}

        for track in self.tracks:
            track.predict(
                timestamp,
                ego_motion=self.current_ego_motion,
                ego_comp_gain=self.ego_motion_match_gain,
            )

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))

        if self.tracks and len(detections) > 0:
            cost_matrix = np.full((len(self.tracks), len(detections)), np.inf, dtype=np.float32)
            for track_idx, track in enumerate(self.tracks):
                for det_idx, det in enumerate(detections):
                    cost_matrix[track_idx, det_idx] = self._cost(track, det)

            finite_mask = np.isfinite(cost_matrix)
            if finite_mask.any():
                safe_cost = cost_matrix.copy()
                safe_cost[~finite_mask] = 1e6
                row_ind, col_ind = linear_sum_assignment(safe_cost)
                used_rows = set()
                used_cols = set()
                for row_idx, col_idx in zip(row_ind, col_ind):
                    if not np.isfinite(cost_matrix[row_idx, col_idx]):
                        continue
                    matched.append((row_idx, col_idx))
                    used_rows.add(row_idx)
                    used_cols.add(col_idx)
                unmatched_trks = [idx for idx in range(len(self.tracks)) if idx not in used_rows]
                unmatched_dets = [idx for idx in range(len(detections)) if idx not in used_cols]

        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(
                detections[det_idx],
                timestamp=timestamp,
                static_hold_time=self.static_hold_time,
                static_yaw_alpha=self.static_yaw_alpha,
                dynamic_yaw_alpha=self.dynamic_yaw_alpha,
                ego_motion=self.current_ego_motion,
            )

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if float(det[8]) < self.min_spawn_score:
                continue
            if not self._should_spawn_track(det):
                continue
            self.tracks.append(
                TimeAwareTrack(
                    det,
                    timestamp=timestamp,
                    expected_dt=self.expected_dt,
                    confirm_time=self.confirm_time,
                    static_speed_thresh=self.static_speed_thresh,
                )
            )

        self.tracks = [
            track for idx, track in enumerate(self.tracks)
            if idx not in unmatched_trks or track.time_since_update_sec <= self.max_age_sec
        ]

        results = []
        for track in self.tracks:
            if not track.is_confirmed and track.hits < 2:
                continue
            results.append(
                {
                    "id": track.id,
                    "box": track.box.copy(),
                    "label": track.cls_id,
                    "score": track.score,
                    "miss_age": track.time_since_update_sec,
                    "speed": track.speed,
                    "is_dynamic": track.is_dynamic,
                    "is_static": track.static_time_sec >= self.static_hold_time,
                    "was_detected": track.time_since_update_sec <= 1e-4,
                }
            )
        return results


class JayTrackerNode(Node):
    def __init__(self, args):
        super().__init__("jay_tracker")
        self.args = args
        self.frame_count = 0
        self.last_stamp = None
        self.track_histories = {}
        self.track_history_maxlen = max(4, int(round(args.track_history_seconds / args.lidar_frame_dt)))
        self.prev_detection_marker_ids = set()
        self.prev_track_marker_ids = set()
        self.prev_text_marker_ids = set()
        self.prev_path_marker_ids = set()
        self.ego_vx = 0.0
        self.ego_vy = 0.0
        self.ego_yaw_rate = 0.0
        self.odom_history = deque(maxlen=256)
        self.imu_history = deque(maxlen=256)
        self.csv_writer = None
        if args.enable_csv_logging:
            self.csv_writer = TrackerCsvWriter(
                sequence_id=args.sequence_id,
                detection_output_dir=args.detection_output_dir,
                track_output_dir=args.track_output_dir,
            )
        self.class_names = ["Vehicle", "Pedestrian", "Cyclist"]
        self.label_aliases = {
            "car": 1,
            "vehicle": 1,
            "pedestrian": 2,
            "cyclist": 3,
        }
        self.fixed_box_sizes = {
            1: np.asarray(args.vehicle_box_size, dtype=np.float32),
            2: np.asarray(args.pedestrian_box_size, dtype=np.float32),
            3: np.asarray(args.cyclist_box_size, dtype=np.float32),
        }
        self.perf_monitor = PerfMonitor(
            self,
            "tracker",
            args.perf_log_interval_sec,
            args.perf_warmup_frames,
            args.perf_output_dir,
        )

        self.tracker = KittiHzTracker(
            expected_dt=args.lidar_frame_dt,
            max_age_frames=args.track_max_age_frames,
            confirm_frames=args.track_confirm_frames,
            match_distance=args.track_match_distance,
            min_spawn_score=args.track_min_spawn_score,
            static_speed_thresh=args.track_static_speed_thresh,
            static_yaw_alpha=args.track_static_yaw_alpha,
            dynamic_yaw_alpha=args.track_dynamic_yaw_alpha,
            ego_motion_match_gain=args.ego_motion_match_gain,
            ego_motion_assoc_bonus=args.ego_motion_assoc_bonus,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(DetectedObjectArray, args.detection_topic, self.detection_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, args.odom_topic, self.odometry_callback, 10)
        self.imu_sub = self.create_subscription(Imu, args.imu_topic, self.imu_callback, 10)
        self.pub_tracks = self.create_publisher(DetectedObjectArray, args.tracked_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, args.bbox_topic, 10)

        self.get_logger().info(f"listening detections: {args.detection_topic}")
        self.get_logger().info(f"publishing tracked objects: {args.tracked_topic}")
        self.get_logger().info(f"publishing: {args.bbox_topic}")
        self.get_logger().info(f"odom: {args.odom_topic}")
        self.get_logger().info(f"imu: {args.imu_topic}")
        if self.csv_writer is not None:
            self.get_logger().info(f"detection csv: {self.csv_writer.detection_output_path}")
            self.get_logger().info(f"track csv: {self.csv_writer.track_output_path}")
        self.get_logger().info(f"performance csv: {self.perf_monitor.csv_path}")
        self.get_logger().info(
            "fixed track box sizes: "
            f"Vehicle={self.fixed_box_sizes[1].tolist()} "
            f"Pedestrian={self.fixed_box_sizes[2].tolist()} "
            f"Cyclist={self.fixed_box_sizes[3].tolist()}"
        )
        self.get_logger().info(
            f"LiDAR timing: {args.lidar_hz:.1f} Hz ({args.lidar_frame_dt:.3f} s/frame)"
        )
        self.get_logger().info(
            f"tracker: max_age={args.track_max_age_frames} frames "
            f"({args.track_max_age_frames * args.lidar_frame_dt:.3f} s), "
            f"confirm={args.track_confirm_frames} frames"
        )
        self.get_logger().info(
            "ego compensation: "
            f"gain={args.ego_motion_match_gain:.2f}, "
            f"assoc_bonus={args.ego_motion_assoc_bonus:.2f}"
        )

    def destroy_node(self):
        if self.csv_writer is not None:
            self.csv_writer.close()
        self.perf_monitor.close()
        super().destroy_node()

    def odometry_callback(self, msg):
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.ego_vx = float(msg.twist.twist.linear.x)
        self.ego_vy = float(msg.twist.twist.linear.y)
        yaw_rate = float(msg.twist.twist.angular.z)
        if abs(yaw_rate) > 1e-4:
            self.ego_yaw_rate = yaw_rate
        self.odom_history.append((timestamp, self.ego_vx, self.ego_vy, yaw_rate))

    def imu_callback(self, msg):
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.ego_yaw_rate = float(msg.angular_velocity.z)
        self.imu_history.append((timestamp, self.ego_yaw_rate))

    def _interpolate_series(self, history, timestamp):
        if not history:
            return None
        if len(history) == 1:
            return history[0][1:]
        if timestamp <= history[0][0]:
            return history[0][1:]
        if timestamp >= history[-1][0]:
            return history[-1][1:]

        for idx in range(1, len(history)):
            t1 = history[idx][0]
            if timestamp > t1:
                continue
            t0 = history[idx - 1][0]
            v0 = np.asarray(history[idx - 1][1:], dtype=np.float32)
            v1 = np.asarray(history[idx][1:], dtype=np.float32)
            dt = max(t1 - t0, 1e-6)
            alpha = float((timestamp - t0) / dt)
            return tuple(((1.0 - alpha) * v0 + alpha * v1).tolist())
        return history[-1][1:]

    def get_ego_motion_for_stamp(self, timestamp):
        odom_values = self._interpolate_series(self.odom_history, timestamp)
        imu_values = self._interpolate_series(self.imu_history, timestamp)

        vx = float(odom_values[0]) if odom_values is not None else self.ego_vx
        vy = float(odom_values[1]) if odom_values is not None and len(odom_values) > 1 else self.ego_vy
        yaw_rate = self.ego_yaw_rate
        if imu_values is not None:
            yaw_rate = float(imu_values[0])
        elif odom_values is not None and len(odom_values) > 2:
            yaw_rate = float(odom_values[2])

        return {"vx": vx, "vy": vy, "yaw_rate": yaw_rate}

    def detection_callback(self, msg):
        stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if stamp == self.last_stamp:
            return
        self.last_stamp = stamp

        t0 = time.perf_counter()
        msg_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        detections = self.detected_objects_to_array(msg)
        t_parse = time.perf_counter()

        timestamp = msg_stamp
        if self.csv_writer is not None:
            self.csv_writer.save_detections(frame_id=self.frame_count, timestamp=timestamp, detections=detections)
        t_csv_det = time.perf_counter()
        ego_motion = self.get_ego_motion_for_stamp(timestamp)
        t_ego = time.perf_counter()
        tracked_objects = self.tracker.update(detections, timestamp=timestamp, ego_motion=ego_motion)
        t_track = time.perf_counter()
        if self.csv_writer is not None:
            self.csv_writer.save_tracks(frame_id=self.frame_count, timestamp=timestamp, tracked_objects=tracked_objects)
        t_csv_track = time.perf_counter()
        self.update_track_histories(tracked_objects)
        t_history = time.perf_counter()
        self.publish_tracked_objects(tracked_objects, msg.header)
        t_publish_tracks = time.perf_counter()
        self.publish_markers(detections, tracked_objects, msg.header.frame_id, msg.header.stamp)
        t_markers = time.perf_counter()

        self.frame_count += 1
        latency_ms = (t_markers - t0) * 1000.0
        self.perf_monitor.add(
            frame_index=self.frame_count,
            total_ms=latency_ms,
            stage_ms={
                "parse": (t_parse - t0) * 1000.0,
                "csv_det": (t_csv_det - t_parse) * 1000.0,
                "ego": (t_ego - t_csv_det) * 1000.0,
                "track": (t_track - t_ego) * 1000.0,
                "csv_track": (t_csv_track - t_track) * 1000.0,
                "history": (t_history - t_csv_track) * 1000.0,
                "markers": (t_markers - t_publish_tracks) * 1000.0,
            },
            input_count=len(detections),
            output_count=len(tracked_objects),
            msg_stamp=msg_stamp,
        )
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f"frame={self.frame_count} dets={len(detections)} "
                f"tracks={len(tracked_objects)} latency={latency_ms:.1f}ms "
                f"ego={math.hypot(ego_motion['vx'], ego_motion['vy']) * 3.6:.1f}km/h"
            )

    def parse_label(self, label_value):
        label_str = str(label_value)
        try:
            return int(label_str)
        except ValueError:
            return self.label_aliases.get(label_str.strip().lower(), 0)

    def get_fixed_box_size(self, label_id):
        return self.fixed_box_sizes.get(int(label_id), None)

    def get_track_render_box(self, track):
        box = track["box"].copy()
        fixed_size = self.get_fixed_box_size(track["label"])
        if fixed_size is not None:
            box[3:6] = fixed_size
        return box

    def detected_objects_to_array(self, msg):
        detections = []
        for obj in msg.objects:
            score = float(obj.score)
            if score < self.args.score_thresh:
                continue
            if len(obj.pose) < 3 or len(obj.dimensions) < 3:
                continue
            detections.append([
                float(obj.pose[0]),
                float(obj.pose[1]),
                float(obj.pose[2]),
                float(obj.dimensions[0]),
                float(obj.dimensions[1]),
                float(obj.dimensions[2]),
                float(obj.yaw),
                float(self.parse_label(obj.label)),
                score,
            ])

        if not detections:
            return np.empty((0, 9), dtype=np.float32)

        det_array = np.asarray(detections, dtype=np.float32)
        det_array[:, :7], det_array[:, 8], det_array[:, 7] = (
            det_array[:, :7],
            det_array[:, 8],
            det_array[:, 7],
        )
        boxes, scores, labels = apply_center_nms(
            det_array[:, :7],
            det_array[:, 8],
            det_array[:, 7].astype(np.int32),
            dist_thresh=self.args.det_nms_distance,
        )
        return np.hstack([
            boxes.astype(np.float32, copy=False),
            labels.reshape(-1, 1).astype(np.float32),
            scores.reshape(-1, 1).astype(np.float32),
        ])

    def update_track_histories(self, tracked_objects):
        active_track_ids = set()
        for track in tracked_objects:
            track_id = int(track["id"])
            active_track_ids.add(track_id)
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=self.track_history_maxlen)

            if track["was_detected"]:
                box = track["box"]
                self.track_histories[track_id].append(
                    (float(box[0]), float(box[1]), float(box[2]))
                )

        stale_track_ids = [
            track_id for track_id in self.track_histories
            if track_id not in active_track_ids
        ]
        for track_id in stale_track_ids:
            del self.track_histories[track_id]

    def publish_tracked_objects(self, tracked_objects, header):
        out_msg = DetectedObjectArray()
        out_msg.header = header

        for track in tracked_objects:
            box = self.get_track_render_box(track)
            label = int(track["label"])
            cls_name = self.class_names[label - 1] if 0 < label <= len(self.class_names) else "Unknown"
            obj = DetectedObject()
            obj.header = header
            obj.id = int(track["id"])
            obj.label = cls_name
            obj.score = float(track["score"])
            obj.pose = [float(box[0]), float(box[1]), float(box[2])]
            obj.dimensions = [float(box[3]), float(box[4]), float(box[5])]
            obj.yaw = float(box[6])
            out_msg.objects.append(obj)

        self.pub_tracks.publish(out_msg)

    def make_delete_marker(self, frame_id, stamp, namespace, marker_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.action = Marker.DELETE
        return marker

    def publish_markers(self, detections, tracked_objects, frame_id, stamp):
        marker_array = MarkerArray()

        lifetime = Duration(seconds=self.args.marker_lifetime).to_msg()
        current_detection_ids = set()
        current_track_ids = set()
        current_text_ids = set()
        current_path_ids = set()

        for det_idx, det in enumerate(detections):
            current_detection_ids.add(det_idx)
            det_marker = Marker()
            det_marker.header.frame_id = frame_id
            det_marker.header.stamp = stamp
            det_marker.ns = "kitti_based_detections"
            det_marker.id = det_idx
            det_marker.type = Marker.CUBE
            det_marker.action = Marker.ADD
            det_marker.pose.position.x = float(det[0])
            det_marker.pose.position.y = float(det[1])
            det_marker.pose.position.z = float(det[2])
            det_marker.pose.orientation.z = math.sin(float(det[6]) * 0.5)
            det_marker.pose.orientation.w = math.cos(float(det[6]) * 0.5)
            det_marker.scale.x = float(det[3])
            det_marker.scale.y = float(det[4])
            det_marker.scale.z = float(det[5])
            det_marker.color.r = DETECTION_COLOR[0]
            det_marker.color.g = DETECTION_COLOR[1]
            det_marker.color.b = DETECTION_COLOR[2]
            det_marker.color.a = self.args.detection_marker_alpha
            det_marker.lifetime = lifetime
            marker_array.markers.append(det_marker)

        for idx, track in enumerate(tracked_objects):
            box = self.get_track_render_box(track)
            label = int(track["label"])
            score = float(track["score"])
            track_id = int(track["id"])
            miss_age = float(track["miss_age"])
            cls_name = self.class_names[label - 1] if 0 < label <= len(self.class_names) else "Unknown"
            color = TRACK_COLORS[(track_id % len(TRACK_COLORS))]
            alpha = self.args.marker_alpha
            current_track_ids.add(track_id)
            current_text_ids.add(track_id + 10000)

            box_marker = Marker()
            box_marker.header.frame_id = frame_id
            box_marker.header.stamp = stamp
            box_marker.ns = "kitti_based_tracks"
            box_marker.id = track_id
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = float(box[0])
            box_marker.pose.position.y = float(box[1])
            box_marker.pose.position.z = float(box[2])
            box_marker.pose.orientation.z = math.sin(float(box[6]) * 0.5)
            box_marker.pose.orientation.w = math.cos(float(box[6]) * 0.5)
            box_marker.scale.x = float(box[3])
            box_marker.scale.y = float(box[4])
            box_marker.scale.z = float(box[5])
            box_marker.color.r = color[0]
            box_marker.color.g = color[1]
            box_marker.color.b = color[2]
            box_marker.color.a = alpha
            box_marker.lifetime = lifetime
            marker_array.markers.append(box_marker)

            history = self.track_histories.get(track_id, ())
            if len(history) >= 2:
                current_path_ids.add(track_id + 20000)
                path_marker = Marker()
                path_marker.header.frame_id = frame_id
                path_marker.header.stamp = stamp
                path_marker.ns = "kitti_based_track_paths"
                path_marker.id = track_id + 20000
                path_marker.type = Marker.LINE_STRIP
                path_marker.action = Marker.ADD
                path_marker.scale.x = self.args.track_line_width
                path_marker.color.r = color[0]
                path_marker.color.g = color[1]
                path_marker.color.b = color[2]
                path_marker.color.a = max(0.35, alpha)
                path_marker.lifetime = lifetime
                path_marker.points = []
                for hx, hy, hz in history:
                    point = Point()
                    point.x = hx
                    point.y = hy
                    point.z = hz + 0.1
                    path_marker.points.append(point)
                marker_array.markers.append(path_marker)

            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = "kitti_based_track_ids"
            text_marker.id = track_id + 10000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(box[0])
            text_marker.pose.position.y = float(box[1])
            text_marker.pose.position.z = float(box[2] + box[5] * 0.7)
            text_marker.scale.z = 0.45
            motion_tag = "static" if track["is_static"] else "track"
            text_marker.text = f"{cls_name} {track_id} {motion_tag} {score:.2f}"
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.lifetime = lifetime
            marker_array.markers.append(text_marker)

        for marker_id in self.prev_detection_marker_ids - current_detection_ids:
            marker_array.markers.append(
                self.make_delete_marker(frame_id, stamp, "kitti_based_detections", marker_id)
            )
        for marker_id in self.prev_track_marker_ids - current_track_ids:
            marker_array.markers.append(
                self.make_delete_marker(frame_id, stamp, "kitti_based_tracks", marker_id)
            )
        for marker_id in self.prev_text_marker_ids - current_text_ids:
            marker_array.markers.append(
                self.make_delete_marker(frame_id, stamp, "kitti_based_track_ids", marker_id)
            )
        for marker_id in self.prev_path_marker_ids - current_path_ids:
            marker_array.markers.append(
                self.make_delete_marker(frame_id, stamp, "kitti_based_track_paths", marker_id)
            )

        self.prev_detection_marker_ids = current_detection_ids
        self.prev_track_marker_ids = current_track_ids
        self.prev_text_marker_ids = current_text_ids
        self.prev_path_marker_ids = current_path_ids

        self.pub_markers.publish(marker_array)


def parse_config():
    parser = argparse.ArgumentParser(description="Jay tracker using fixed LiDAR timing")
    parser.add_argument(
        "--detection_topic",
        type=str,
        default="/detected_objects_3d",
        help="ROS2 detected object topic from detector or coda_detector",
    )
    parser.add_argument(
        "--odom_topic",
        type=str,
        default="/odometry/wheel",
        help="ROS2 odometry topic used for ego-motion compensation",
    )
    parser.add_argument(
        "--imu_topic",
        type=str,
        default="/vectornav/imu",
        help="ROS2 IMU topic used for yaw-rate compensation",
    )
    parser.add_argument(
        "--bbox_topic",
        type=str,
        default="/pcdet/kitti_based_tracks",
        help="MarkerArray topic for tracked boxes",
    )
    parser.add_argument(
        "--tracked_topic",
        type=str,
        default="/tracked_objects_3d",
        help="DetectedObjectArray topic for tracked objects",
    )
    parser.add_argument(
        "--enable_csv_logging",
        type=lambda value: str(value).lower() in ("1", "true", "yes", "on"),
        default=False,
        help="enable CSV export for detections and tracks",
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default="seq01",
        help="sequence id used in CSV filenames and rows",
    )
    parser.add_argument(
        "--detection_output_dir",
        type=str,
        default="results/detections",
        help="directory where detection CSV files are written",
    )
    parser.add_argument(
        "--track_output_dir",
        type=str,
        default="results/tracks",
        help="directory where track CSV files are written",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.65,
        help="minimum detection score before tracking",
    )
    parser.add_argument(
        "--lidar_hz",
        type=float,
        default=8.6,
        help="LiDAR rate from the ROS bag",
    )
    parser.add_argument(
        "--lidar_frame_dt",
        type=float,
        default=0.116,
        help="fixed LiDAR frame interval in seconds used by the tracker",
    )
    parser.add_argument(
        "--marker_alpha",
        type=float,
        default=0.45,
        help="track marker alpha",
    )
    parser.add_argument(
        "--detection_marker_alpha",
        type=float,
        default=0.16,
        help="raw detector box marker alpha",
    )
    parser.add_argument(
        "--marker_lifetime",
        type=float,
        default=0.0,
        help="marker lifetime in seconds, 0 keeps markers until explicitly updated or deleted",
    )
    parser.add_argument(
        "--vehicle_box_size",
        type=float,
        nargs=3,
        metavar=("L", "W", "H"),
        default=DEFAULT_TRACK_BOX_SIZES[1],
        help="fixed tracked box size for Vehicle/Car as length width height",
    )
    parser.add_argument(
        "--pedestrian_box_size",
        type=float,
        nargs=3,
        metavar=("L", "W", "H"),
        default=DEFAULT_TRACK_BOX_SIZES[2],
        help="fixed tracked box size for Pedestrian as length width height",
    )
    parser.add_argument(
        "--cyclist_box_size",
        type=float,
        nargs=3,
        metavar=("L", "W", "H"),
        default=DEFAULT_TRACK_BOX_SIZES[3],
        help="fixed tracked box size for Cyclist as length width height",
    )
    parser.add_argument(
        "--track_max_age_frames",
        type=int,
        default=5,
        help="how many LiDAR frames a track survives without a detection",
    )
    parser.add_argument(
        "--track_confirm_frames",
        type=int,
        default=2,
        help="how many LiDAR frames are needed before a track is fully confirmed",
    )
    parser.add_argument(
        "--track_match_distance",
        type=float,
        default=2.4,
        help="maximum XY distance for detection-to-track matching",
    )
    parser.add_argument(
        "--track_min_spawn_score",
        type=float,
        default=0.20,
        help="minimum score needed to spawn a new track",
    )
    parser.add_argument(
        "--track_static_speed_thresh",
        type=float,
        default=0.60,
        help="speed threshold below which yaw is treated as stationary",
    )
    parser.add_argument(
        "--track_static_yaw_alpha",
        type=float,
        default=0.04,
        help="yaw update gain for stationary tracks",
    )
    parser.add_argument(
        "--track_dynamic_yaw_alpha",
        type=float,
        default=0.20,
        help="yaw update gain for moving tracks",
    )
    parser.add_argument(
        "--ego_motion_match_gain",
        type=float,
        default=1.0,
        help="how strongly ego-motion compensation is applied before track association",
    )
    parser.add_argument(
        "--ego_motion_assoc_bonus",
        type=float,
        default=0.35,
        help="extra association tolerance added under ego motion",
    )
    parser.add_argument(
        "--det_nms_distance",
        type=float,
        default=1.2,
        help="same-class detection center NMS distance before tracking",
    )
    parser.add_argument(
        "--track_history_seconds",
        type=float,
        default=2.0,
        help="how many seconds of per-ID trajectory history to draw",
    )
    parser.add_argument(
        "--track_line_width",
        type=float,
        default=0.12,
        help="LINE_STRIP width for track trajectories",
    )
    parser.add_argument(
        "--perf_log_interval_sec",
        type=float,
        default=1.0,
        help="seconds between tracker performance summaries",
    )
    parser.add_argument(
        "--perf_warmup_frames",
        type=int,
        default=5,
        help="number of initial frames to skip for tracker performance summaries",
    )
    parser.add_argument(
        "--perf_output_dir",
        type=str,
        default="results/perf",
        help="directory where tracker performance CSV is written",
    )
    args, _ = parser.parse_known_args()

    args.detection_output_dir = str((Path.cwd() / args.detection_output_dir).resolve())
    args.track_output_dir = str((Path.cwd() / args.track_output_dir).resolve())
    args.perf_output_dir = str((Path.cwd() / args.perf_output_dir).resolve())
    args.lidar_frame_dt = float(args.lidar_frame_dt)
    if args.lidar_frame_dt <= 0.0:
        parser.error("--lidar_frame_dt must be positive.")
    args.lidar_hz = float(args.lidar_hz)
    if args.lidar_hz <= 0.0:
        args.lidar_hz = 1.0 / args.lidar_frame_dt

    return args


def main():
    args = parse_config()

    rclpy.init()
    node = JayTrackerNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
