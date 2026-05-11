#!/usr/bin/env python3
import csv
import gc
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

OPENPCDET_PATH = "/home/jay/OpenPCDet"
sys.path.insert(0, OPENPCDET_PATH)
warnings.filterwarnings("ignore", category=UserWarning)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.models import build_network


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class SimpleKFTrack:
    _next_id = 0

    def __init__(self, det, timestamp, n_init=2, default_dt=0.1):
        self.id = SimpleKFTrack._next_id
        SimpleKFTrack._next_id += 1

        self.default_dt = default_dt
        self.last_timestamp = timestamp
        self.n_init = n_init

        # [x, y, vx, vy]
        self.x = np.array([det[0], det[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        self.P = np.eye(4, dtype=np.float32)
        self.F = np.eye(4, dtype=np.float32)
        self.H = np.zeros((2, 4), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 0.1

        self.z = float(det[2])
        self.dim = det[3:6].astype(np.float32).copy()
        self.yaw = float(det[6])
        self.cls_id = int(det[7])
        self.score = float(det[8])
        self.min_speed_for_yaw = 1.0
        self.max_yaw_jump = np.deg2rad(25.0)
        self.yaw_alpha = 0.15
        self.early_yaw_alpha = 0.25

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"
        if self.hits >= self.n_init:
            self.state = "confirmed"

    def _set_dt(self, dt):
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.Q[0, 0] = self.Q[1, 1] = max(0.01, 0.05 * dt)
        self.Q[2, 2] = self.Q[3, 3] = max(0.01, 0.2 * dt)

    def predict(self, timestamp, ego_motion=None):
        if self.last_timestamp is None or timestamp is None:
            dt = self.default_dt
        else:
            dt = float(timestamp - self.last_timestamp)
            if dt <= 0.0:
                dt = self.default_dt
            dt = min(max(dt, 1e-2), 1.0)

        self._set_dt(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        speed = float(np.linalg.norm(self.x[2:4, 0]))
        ego_yaw_rate = abs(float((ego_motion or {}).get("yaw_rate", 0.0)))
        if speed >= self.min_speed_for_yaw and ego_yaw_rate < 0.15:
            vel_yaw = np.arctan2(self.x[3, 0], self.x[2, 0])
            self.yaw = wrap_angle(0.95 * self.yaw + 0.05 * vel_yaw)
        self.age += 1
        self.time_since_update += 1
        self.last_timestamp = timestamp

    def update(self, det, timestamp, ego_motion=None):
        measurement = np.array([det[0], det[1]], dtype=np.float32).reshape(2, 1)
        residual = measurement - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

        self.z = 0.8 * self.z + 0.2 * float(det[2])
        self.dim = 0.8 * self.dim + 0.2 * det[3:6].astype(np.float32)
        yaw_delta = wrap_angle(float(det[6] - self.yaw))
        if abs(yaw_delta) > np.pi / 2:
            flip_delta = yaw_delta - np.sign(yaw_delta) * np.pi
            if abs(flip_delta) < abs(yaw_delta):
                yaw_delta = flip_delta
        speed = float(np.linalg.norm(self.x[2:4, 0]))
        ego_motion = ego_motion or {}
        ego_yaw_rate = abs(float(ego_motion.get("yaw_rate", 0.0)))
        ego_speed = abs(float(ego_motion.get("speed", 0.0)))
        dynamic_scene = ego_yaw_rate > 0.15 or ego_speed > 1.0
        allow_early_yaw = self.hits < self.n_init + 1 and abs(yaw_delta) <= np.deg2rad(50.0)
        use_yaw_update = (speed >= self.min_speed_for_yaw and abs(yaw_delta) <= self.max_yaw_jump) or allow_early_yaw
        if dynamic_scene and speed < (self.min_speed_for_yaw + 0.5):
            use_yaw_update = False
        if ego_yaw_rate > 0.25:
            use_yaw_update = False
        if use_yaw_update:
            alpha = self.early_yaw_alpha if allow_early_yaw and speed < self.min_speed_for_yaw else self.yaw_alpha
            if ego_yaw_rate > 0.1:
                alpha *= 0.5
            self.yaw = wrap_angle(self.yaw + alpha * yaw_delta)
        self.cls_id = int(det[7])
        self.score = float(det[8])

        self.hits += 1
        self.time_since_update = 0
        self.last_timestamp = timestamp
        if self.hits >= self.n_init:
            self.state = "confirmed"

    def mark_missed(self, max_age):
        if self.time_since_update > max_age:
            self.state = "deleted"

    @property
    def is_confirmed(self):
        return self.state == "confirmed"

    @property
    def is_deleted(self):
        return self.state == "deleted"


class SimpleKFTracker:
    def __init__(self, max_age=6, min_hits=2, dist_threshold=2.5, score_threshold=0.35):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.score_threshold = score_threshold
        self.tracks = []

    def _match_cost(self, track, det):
        det_cls = int(det[7])
        if track.cls_id != det_cls:
            return np.inf
        xy_dist = np.linalg.norm(track.x[:2, 0] - det[:2])
        if xy_dist > self.dist_threshold:
            return np.inf
        yaw_diff = abs(wrap_angle(float(track.yaw - det[6])))
        size_diff = np.linalg.norm(track.dim - det[3:6]) / max(np.linalg.norm(track.dim), 1e-3)
        if size_diff > 0.5:
            return np.inf
        return xy_dist + 0.25 * yaw_diff + 0.25 * size_diff

    def update(self, detections, timestamp=None, ego_motion=None):
        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            detections = np.empty((0, 9), dtype=np.float32)

        for track in self.tracks:
            track.predict(timestamp, ego_motion=ego_motion)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))

        if self.tracks and len(detections) > 0:
            cost_matrix = np.full((len(self.tracks), len(detections)), np.inf, dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = self._match_cost(track, det)

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
                unmatched_trks = [i for i in range(len(self.tracks)) if i not in used_rows]
                unmatched_dets = [j for j in range(len(detections)) if j not in used_cols]

        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx], timestamp, ego_motion=ego_motion)

        for track_idx in unmatched_trks:
            self.tracks[track_idx].mark_missed(self.max_age)

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if float(det[8]) < self.score_threshold:
                continue
            self.tracks.append(SimpleKFTrack(det, timestamp, n_init=self.min_hits))

        self.tracks = [track for track in self.tracks if not track.is_deleted]

        outputs = []
        for track in self.tracks:
            recently_seen_tentative = (not track.is_confirmed) and track.hits >= max(1, self.min_hits - 1) and track.time_since_update <= 1
            if not track.is_confirmed and not recently_seen_tentative:
                continue
            outputs.append([
                track.x[0, 0],
                track.x[1, 0],
                track.z,
                track.id,
                track.yaw,
                track.cls_id,
                track.dim[0],
                track.dim[1],
                track.dim[2],
                track.time_since_update,
                track.score,
            ])
        return outputs


class OpenPCDetKFNode(Node):
    def __init__(self):
        super().__init__("pcdet_ros2_kf_node")

        self.cfg_file = os.path.join(OPENPCDET_PATH, "tools/cfgs/kitti_models/pv_rcnn_my_ver.yaml")
        self.ckpt_file = os.path.join(OPENPCDET_PATH, "output/coda32_allclass_bestoracle.pth")
        self.lidar_topic = "/no_ground_points"
        self.z_offset = 0.64
        self.score_thresh = 0.4
        self.nms_dist_thresh = 1.5
        self.max_detection_range = 50.0

        self.logger_ros = self.get_logger()
        model_name = os.path.basename(self.cfg_file).split(".")[0]
        self.log_filename = f"perf_log_kf_{model_name}_{time.strftime('%m%d_%H%M')}.csv"
        self.csv_file = open(self.log_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame_id", "pre_proc_ms", "inference_ms", "post_proc_ms", "total_ms", "obj_count", "avg_score"])

        cfg_from_yaml_file(self.cfg_file, cfg)
        self.class_names = cfg.CLASS_NAMES
        pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        self.data_processor = DataProcessor(
            processor_configs=cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=pc_range,
            training=False,
            num_point_features=4,
        )

        self.logger_ros.info("Building Model...")
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(self.class_names),
            dataset=self.build_dummy_dataset(cfg, pc_range),
        )
        self.model.load_params_from_file(filename=self.ckpt_file, logger=self.logger_ros, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        self.tracker = SimpleKFTracker(max_age=6, min_hits=2, dist_threshold=2.5, score_threshold=self.score_thresh)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/pcdet/detections_kf", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odom_sub = self.create_subscription(Odometry, "/odometry/wheel", self.odometry_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/vectornav/imu", self.imu_callback, 10)
        self.ego_speed = 0.0
        self.ego_yaw_rate = 0.0
        self.frame_count = 0
        self.logger_ros.info(f"Node Ready! Logging to {self.log_filename}")

    def build_dummy_dataset(self, cfg_obj, pc_range):
        class Dummy:
            def __init__(self):
                self.class_names = cfg_obj.CLASS_NAMES
                self.point_feature_encoder = type("Encoder", (), {"num_point_features": 4})()
                self.point_cloud_range = pc_range
                self.depth_downsample_factor = None
                self.dataset_cfg = cfg_obj.DATA_CONFIG
                self.voxel_size = None
                for proc in cfg_obj.DATA_CONFIG.DATA_PROCESSOR:
                    if proc["NAME"] == "transform_points_to_voxels":
                        self.voxel_size = np.array(proc["VOXEL_SIZE"], dtype=np.float32)
                        break
                self.grid_size = np.round((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(np.int64)
        return Dummy()

    def cleanup_memory(self):
        self.logger_ros.info("종료 중: 로그 파일을 저장하고 메모리를 해제합니다.")
        self.csv_file.close()
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def odometry_callback(self, msg):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.ego_speed = math.hypot(vx, vy)

    def imu_callback(self, msg):
        self.ego_yaw_rate = float(msg.angular_velocity.z)

    def lidar_callback(self, msg):
        start_total = time.perf_counter()
        self.frame_count += 1

        start_pre = time.perf_counter()
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points = np.array([[p[0], p[1], p[2], p[3]] for p in gen], dtype=np.float32)
        if points.shape[0] < 10:
            return
        points[:, 2] -= self.z_offset
        if points[:, 3].max() > 1.0:
            points[:, 3] /= 255.0

        input_dict = {"points": points, "frame_id": msg.header.frame_id, "use_lead_xyz": True}
        data_dict = self.data_processor.forward(data_dict=input_dict)
        data_dict["points"] = torch.from_numpy(np.pad(data_dict["points"], ((0, 0), (1, 0)), mode="constant")).float().cuda()
        if "voxels" in data_dict:
            for key in ["voxels", "voxel_num_points", "voxel_coords"]:
                data_dict[key] = torch.from_numpy(data_dict[key]).cuda()
                if key == "voxel_coords":
                    data_dict[key] = torch.nn.functional.pad(data_dict[key], (1, 0), mode="constant", value=0)
        data_dict["batch_size"] = 1
        pre_time = (time.perf_counter() - start_pre) * 1000

        start_inf = time.perf_counter()
        with torch.no_grad():
            torch.cuda.synchronize()
            pred_dicts, _ = self.model.forward(data_dict)
            torch.cuda.synchronize()
        inf_time = (time.perf_counter() - start_inf) * 1000

        start_post = time.perf_counter()
        obj_count, avg_score = self.process_and_publish(pred_dicts[0], msg.header)
        post_time = (time.perf_counter() - start_post) * 1000

        total_time = (time.perf_counter() - start_total) * 1000
        self.csv_writer.writerow([self.frame_count, pre_time, inf_time, post_time, total_time, obj_count, avg_score])
        if self.frame_count % 20 == 0:
            self.logger_ros.info(f"FPS: {1000/total_time:.1f} | Latency: {inf_time:.1f}ms | Tracks: {obj_count}")

    def apply_nms(self, boxes, scores, labels):
        if len(boxes) == 0:
            return boxes, scores, labels
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx = order[0]
            keep.append(idx)
            dists = np.linalg.norm(boxes[order[1:], :2] - boxes[idx, :2], axis=1)
            order = order[np.where(dists > self.nms_dist_thresh)[0] + 1]
        return boxes[keep], scores[keep], labels[keep]

    def process_and_publish(self, pred_dict, header):
        boxes = pred_dict["pred_boxes"].cpu().numpy()
        scores = pred_dict["pred_scores"].cpu().numpy()
        labels = pred_dict["pred_labels"].cpu().numpy()
        timestamp = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

        mask = (scores > self.score_thresh) & (np.linalg.norm(boxes[:, :2], axis=1) < self.max_detection_range)
        boxes, scores, labels = self.apply_nms(boxes[mask], scores[mask], labels[mask])
        if len(boxes) == 0:
            detections = np.empty((0, 9), dtype=np.float32)
        else:
            detections = np.hstack([boxes, labels.reshape(-1, 1), scores.reshape(-1, 1)]).astype(np.float32, copy=False)

        target_frame = "odom"
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, header.frame_id, Time.from_msg(header.stamp), timeout=Duration(seconds=0.05))
            rot = R.from_quat([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ])
            if len(detections) > 0:
                xyz = detections[:, 0:3]
                detections[:, 0:3] = np.dot(xyz, rot.as_matrix().T) + np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ])
                detections[:, 6] += rot.as_euler("zyx")[0]
            frame_id = target_frame
        except Exception:
            frame_id = header.frame_id

        tracked_objects = self.tracker.update(
            detections,
            timestamp=timestamp,
            ego_motion={"speed": self.ego_speed, "yaw_rate": self.ego_yaw_rate},
        )
        self.publish_markers(tracked_objects, frame_id, header.stamp)
        avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        return len(tracked_objects), avg_score

    def publish_markers(self, tracked_objects, frame_id, stamp):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for obj in tracked_objects:
            x, y, z, track_id, yaw, label, dx, dy, dz, age, score = obj
            x, y, z, dx, dy, dz, yaw = map(float, [x, y, z, dx, dy, dz, yaw])
            track_id = int(track_id)
            label = int(label)
            cls_name = self.class_names[label - 1] if 0 < label <= len(self.class_names) else "Unknown"

            alpha = 0.8 if int(age) == 0 else max(0.2, 0.7 - 0.1 * int(age))

            box_marker = Marker()
            box_marker.header.frame_id = frame_id
            box_marker.header.stamp = stamp
            box_marker.ns = "kf_tracks"
            box_marker.id = track_id
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = x
            box_marker.pose.position.y = y
            box_marker.pose.position.z = z
            box_marker.pose.orientation.z = math.sin(yaw / 2.0)
            box_marker.pose.orientation.w = math.cos(yaw / 2.0)
            box_marker.scale.x = dx
            box_marker.scale.y = dy
            box_marker.scale.z = dz
            box_marker.color.r = 0.0
            box_marker.color.g = 1.0
            box_marker.color.b = 0.0
            box_marker.color.a = alpha
            box_marker.lifetime = Duration(seconds=1.0).to_msg()
            marker_array.markers.append(box_marker)

            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = "kf_track_ids"
            text_marker.id = track_id + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + dz / 2.0 + 0.5
            text_marker.scale.z = 0.5
            text_marker.text = f"{cls_name} {track_id} ({score:.2f})"
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.lifetime = Duration(seconds=1.0).to_msg()
            marker_array.markers.append(text_marker)

        self.pub_markers.publish(marker_array)


def main():
    rclpy.init()
    node = OpenPCDetKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup_memory()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
