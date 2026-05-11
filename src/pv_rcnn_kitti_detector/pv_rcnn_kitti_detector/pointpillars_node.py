#!/usr/bin/env python3

import csv
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from tracking_msgs.msg import DetectedObject, DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPCDET_PATH = Path(__file__).resolve().parents[3] / "third_party" / "OpenPCDet"
if str(OPENPCDET_PATH) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_PATH))

warnings.filterwarnings("ignore", category=UserWarning)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.models import build_network


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
            "read_ms",
            "batch_ms",
            "infer_ms",
            "post_ms",
            "pub_msg_ms",
            "pub_marker_ms",
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
            f"{float(stage_ms.get('read', 0.0)):.6f}",
            f"{float(stage_ms.get('batch', 0.0)):.6f}",
            f"{float(stage_ms.get('infer', 0.0)):.6f}",
            f"{float(stage_ms.get('post', 0.0)):.6f}",
            f"{float(stage_ms.get('pub_msg', 0.0)):.6f}",
            f"{float(stage_ms.get('pub_marker', 0.0)):.6f}",
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
            f"avg_in={avg_in:.0f} avg_out={avg_out:.1f} {stage_summary}"
        )
        self.reset()


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


class KittiDetectorNode(Node):
    def __init__(self):
        super().__init__("kitti_detector_node")

        self.declare_parameter("input_topic", "/no_ground_points")
        self.declare_parameter("detection_topic", "/detected_objects_3d")
        self.declare_parameter("marker_topic", "/detections/visual_markers")
        self.declare_parameter("score_thresh", 0.30)
        self.declare_parameter("z_offset", 0.64)
        self.declare_parameter("det_nms_distance", 1.2)
        self.declare_parameter("debug_every_n_frames", 10)
        self.declare_parameter("perf_log_interval_sec", 1.0)
        self.declare_parameter("perf_warmup_frames", 5)
        self.declare_parameter("perf_output_dir", "results/perf")

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.score_thresh = float(self.get_parameter("score_thresh").value)
        self.z_offset = float(self.get_parameter("z_offset").value)
        self.det_nms_distance = float(self.get_parameter("det_nms_distance").value)
        self.debug_every_n_frames = int(self.get_parameter("debug_every_n_frames").value)
        self.frame_count = 0
        self.perf_monitor = PerfMonitor(
            self,
            "detector",
            self.get_parameter("perf_log_interval_sec").value,
            self.get_parameter("perf_warmup_frames").value,
            self.get_parameter("perf_output_dir").value,
        )

        self.cfg_file = str(
            Path(get_package_share_directory("pv_rcnn_kitti_detector")) / "config" / "pv_rcnn_my_ver.yaml"
        )
        self.ckpt_file = str(Path.home() / "pcdet_ros2_ws" / "models" / "pv-rcnn_kitti.pth")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        tools_dir = OPENPCDET_PATH / "tools"
        if tools_dir.exists():
            os.chdir(tools_dir)

        self.get_logger().info("Loading KITTI PV-RCNN config...")
        cfg_from_yaml_file(self.cfg_file, cfg)
        pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

        self.class_names = cfg.CLASS_NAMES
        self.dataset = RosModelDataset(cfg, pc_range)
        self.data_processor = DataProcessor(
            processor_configs=cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=pc_range,
            training=False,
            num_point_features=4,
        )

        self.get_logger().info("Building KITTI PV-RCNN model...")
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(self.class_names),
            dataset=self.dataset,
        )
        self.model.load_params_from_file(filename=self.ckpt_file, logger=self.get_logger(), to_cpu=False)
        self.model.cuda()
        self.model.eval()

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.lidar_callback, qos)
        self.pub = self.create_publisher(DetectedObjectArray, self.detection_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.get_logger().info(
            f"detector ready: input={self.input_topic}, detections={self.detection_topic}, "
            f"score_thresh={self.score_thresh:.2f}, z_offset={self.z_offset:.2f}"
        )
        self.get_logger().info(f"performance csv: {self.perf_monitor.csv_path}")

    def destroy_node(self):
        self.perf_monitor.close()
        super().destroy_node()

    def read_points(self, msg):
        field_names = [field.name for field in msg.fields]
        intensity_field = "intensity" if "intensity" in field_names else ("i" if "i" in field_names else None)
        read_fields = ("x", "y", "z", intensity_field) if intensity_field else ("x", "y", "z")

        gen = pc2.read_points(msg, field_names=read_fields, skip_nans=True)
        try:
            if intensity_field:
                points = np.array([[p[0], p[1], p[2], p[3]] for p in gen], dtype=np.float32)
            else:
                points = np.array([[p[0], p[1], p[2], 0.0] for p in gen], dtype=np.float32)
        except Exception as exc:
            self.get_logger().error(f"point cloud conversion failed: {exc}")
            return None

        if len(points) == 0:
            return None

        points[:, 2] -= self.z_offset
        if float(points[:, 3].max()) > 1.0:
            points[:, 3] /= 255.0
        return points

    def prepare_batch(self, points, frame_id):
        input_dict = {
            "points": points,
            "frame_id": frame_id,
            "use_lead_xyz": True,
        }
        data_dict = self.data_processor.forward(data_dict=input_dict)
        data_dict["points"] = torch.from_numpy(
            np.pad(data_dict["points"], ((0, 0), (1, 0)), mode="constant")
        ).float().cuda()

        if "voxels" in data_dict:
            for key in ["voxels", "voxel_num_points", "voxel_coords"]:
                data_dict[key] = torch.from_numpy(data_dict[key]).cuda()
                if key == "voxel_coords":
                    data_dict[key] = torch.nn.functional.pad(
                        data_dict[key], (1, 0), mode="constant", value=0
                    )
        data_dict["batch_size"] = 1
        return data_dict

    def lidar_callback(self, msg):
        self.frame_count += 1
        t0 = time.perf_counter()
        msg_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        points = self.read_points(msg)
        t_read = time.perf_counter()
        if points is None or len(points) < 10:
            return

        batch_dict = self.prepare_batch(points, msg.header.frame_id)
        t_batch = time.perf_counter()
        with torch.no_grad():
            torch.cuda.synchronize()
            pred_dicts, _ = self.model.forward(batch_dict)
            torch.cuda.synchronize()
        t_infer = time.perf_counter()

        pred_boxes = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        pred_scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        pred_labels = pred_dicts[0]["pred_labels"].detach().cpu().numpy()
        raw_det_count = len(pred_scores)
        raw_max_score = float(pred_scores.max()) if raw_det_count > 0 else 0.0

        mask = pred_scores >= self.score_thresh
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]
        pred_boxes, pred_scores, pred_labels = apply_center_nms(
            pred_boxes,
            pred_scores,
            pred_labels,
            dist_thresh=self.det_nms_distance,
        )
        t_post = time.perf_counter()

        if self.debug_every_n_frames > 0 and self.frame_count % self.debug_every_n_frames == 0:
            self.get_logger().info(
                f"frame={self.frame_count} points={len(points)} raw_det={raw_det_count} "
                f"raw_max_score={raw_max_score:.3f} kept={len(pred_scores)}"
            )

        self.publish_results(pred_boxes, pred_scores, pred_labels, msg.header)
        t_publish_results = time.perf_counter()
        self.publish_markers(pred_boxes, pred_scores, pred_labels, msg.header)
        t_publish_markers = time.perf_counter()
        self.perf_monitor.add(
            frame_index=self.frame_count,
            total_ms=(t_publish_markers - t0) * 1000.0,
            stage_ms={
                "read": (t_read - t0) * 1000.0,
                "batch": (t_batch - t_read) * 1000.0,
                "infer": (t_infer - t_batch) * 1000.0,
                "post": (t_post - t_infer) * 1000.0,
                "pub_msg": (t_publish_results - t_post) * 1000.0,
                "pub_marker": (t_publish_markers - t_publish_results) * 1000.0,
            },
            input_count=len(points),
            output_count=len(pred_scores),
            msg_stamp=msg_stamp,
        )

    def publish_results(self, boxes, scores, labels, header):
        out_msg = DetectedObjectArray()
        out_msg.header = header

        for i in range(len(boxes)):
            obj = DetectedObject()
            obj.header = header
            obj.id = 0
            obj.label = str(int(labels[i]))
            obj.score = float(scores[i])
            obj.pose = [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2])]
            obj.dimensions = [float(boxes[i][3]), float(boxes[i][4]), float(boxes[i][5])]
            obj.yaw = float(boxes[i][6])
            out_msg.objects.append(obj)

        self.pub.publish(out_msg)

    def publish_markers(self, boxes, scores, labels, header):
        marker_array = MarkerArray()

        for i, box in enumerate(boxes):
            marker = Marker()
            marker.header = header
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            yaw = float(box[6])
            marker.pose.orientation.z = math.sin(yaw / 2.0)
            marker.pose.orientation.w = math.cos(yaw / 2.0)
            marker.scale.x = float(box[3])
            marker.scale.y = float(box[4])
            marker.scale.z = float(box[5])
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "detection_info"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 200000000
            text_marker.pose.position.x = float(box[0])
            text_marker.pose.position.y = float(box[1])
            text_marker.pose.position.z = float(box[2]) + float(box[5]) / 2.0 + 0.5
            text_marker.scale.z = 0.5
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            label_idx = int(labels[i]) - 1
            cls_name = self.class_names[label_idx] if 0 <= label_idx < len(self.class_names) else "Unknown"
            text_marker.text = f"{cls_name}: {float(scores[i]):.2f}"
            marker_array.markers.append(text_marker)

        self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = KittiDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
