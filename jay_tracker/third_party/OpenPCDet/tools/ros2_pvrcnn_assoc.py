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
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
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


def box_corners_bev(center_x, center_y, length, width, yaw):
    half_l = length * 0.5
    half_w = width * 0.5
    corners = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ], dtype=np.float32)
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
    return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
            (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0.0


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


class AssocTrack:
    _next_id = 0

    def __init__(self, det):
        self.id = AssocTrack._next_id
        AssocTrack._next_id += 1
        self.box = det[:7].astype(np.float32).copy()
        self.cls_id = int(det[7])
        self.score = float(det[8])
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = "tentative"

    def update(self, det):
        self.box = det[:7].astype(np.float32).copy()
        self.cls_id = int(det[7])
        self.score = float(det[8])
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def predict_only(self):
        self.time_since_update += 1
        self.age += 1

    @property
    def is_confirmed(self):
        return self.state == "confirmed"


class AssocTracker:
    def __init__(self, max_age=6, min_hits=2, dist_threshold=2.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.tracks = []

    def _cost(self, track, det):
        if track.cls_id != int(det[7]):
            return np.inf
        xy_dist = np.linalg.norm(track.box[:2] - det[:2])
        if xy_dist > self.dist_threshold:
            return np.inf
        iou = bev_iou(
            [track.box[0], track.box[1], track.box[3], track.box[4], track.box[6]],
            [det[0], det[1], det[3], det[4], det[6]],
        )
        yaw_diff = abs(wrap_angle(float(track.box[6] - det[6])))
        size_diff = np.linalg.norm(track.box[3:6] - det[3:6]) / max(np.linalg.norm(track.box[3:6]), 1e-3)
        if size_diff > 0.6:
            return np.inf
        return 1.0 * xy_dist + 1.25 * (1.0 - iou) + 0.2 * yaw_diff + 0.25 * size_diff

    def update(self, detections):
        detections = np.asarray(detections, dtype=np.float32)
        if detections.size == 0:
            detections = np.empty((0, 9), dtype=np.float32)

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
            self.tracks[track_idx].update(detections[det_idx])

        for track_idx in unmatched_trks:
            self.tracks[track_idx].predict_only()

        for det_idx in unmatched_dets:
            self.tracks.append(AssocTrack(detections[det_idx]))

        alive_tracks = []
        for track in self.tracks:
            if track.time_since_update > self.max_age:
                continue
            if track.hits >= self.min_hits:
                track.state = "confirmed"
            alive_tracks.append(track)
        self.tracks = alive_tracks

        results = []
        for track in self.tracks:
            recently_seen_tentative = (not track.is_confirmed) and track.hits >= max(1, self.min_hits - 1) and track.time_since_update <= 1
            if not track.is_confirmed and not recently_seen_tentative:
                continue
            results.append([
                track.box[0],
                track.box[1],
                track.box[2],
                track.id,
                track.box[6],
                track.cls_id,
                track.box[3],
                track.box[4],
                track.box[5],
                track.time_since_update,
                track.score,
            ])
        return results


class OpenPCDetAssocNode(Node):
    def __init__(self):
        super().__init__("pcdet_ros2_assoc_node")

        self.cfg_file = os.path.join(OPENPCDET_PATH, "tools/cfgs/kitti_models/pv_rcnn_my_ver.yaml")
        self.ckpt_file = os.path.join(OPENPCDET_PATH, "output/coda32_allclass_bestoracle.pth")
        self.lidar_topic = "/no_ground_points"
        self.z_offset = 0.64
        self.score_thresh = 0.4
        self.nms_dist_thresh = 1.5
        self.max_detection_range = 50.0

        self.logger_ros = self.get_logger()
        model_name = os.path.basename(self.cfg_file).split(".")[0]
        self.log_filename = f"perf_log_assoc_{model_name}_{time.strftime('%m%d_%H%M')}.csv"
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

        self.tracker = AssocTracker(max_age=6, min_hits=2, dist_threshold=2.5)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/pcdet/detections_assoc", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
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

        tracked_objects = self.tracker.update(detections)
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
            box_marker.ns = "assoc_tracks"
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
            text_marker.ns = "assoc_track_ids"
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
    node = OpenPCDetAssocNode()
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
