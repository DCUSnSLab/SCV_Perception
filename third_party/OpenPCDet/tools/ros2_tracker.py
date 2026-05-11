#!/usr/bin/env python3
import sys
import os
import time
import csv
import math
import warnings
import numpy as np
import torch
import gc 

# ---------------- CONFIGURATION ----------------
# OpenPCDet 루트 경로 (사용자 환경에 맞게 수정 필요)
OPENPCDET_PATH = "/home/jay/OpenPCDet"
sys.path.insert(0, OPENPCDET_PATH)
# -----------------------------------------------

# PyTorch 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, cKDTree

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets.processor.data_processor import DataProcessor

# 같은 폴더에 있는 tracker.py 임포트
try:
    from tracker import GlobalTracker, bev_iou, wrap_angle
except ImportError:
    print("Error: 'tracker.py' not found in the same directory.")
    sys.exit(1)

class OpenPCDetNode(Node):
    def __init__(self):
        super().__init__('pcdet_ros2_node')
        
        # 1. 파일 경로 및 파라미터 설정
        self.cfg_file = os.path.join(OPENPCDET_PATH, 'tools/cfgs/kitti_models/pv_rcnn_my_ver.yaml')
        self.ckpt_file = os.path.join(OPENPCDET_PATH, 'output/coda32_allclass_bestoracle.pth')
        self.lidar_topic = '/no_ground_points' 
        self.fallback_lidar_topic = '/velodyne_points'
        self.z_offset = 0.64 
        self.SCORE_THRESH = 0.3
        self.TRACK_SCORE_THRESH = 0.2
        self.NMS_DIST_THRESH = 1.75
        self.NMS_IOU_THRESH = 0.2
        self.MAX_DETECTION_RANGE = 50.0
        self.BEV_FIT_MIN_POINTS = 12
        self.BEV_CLUSTER_TOL = 0.45
        self.BEV_CLUSTER_MIN_POINTS = 8
        self.BEV_BOX_MARGIN = np.array([0.35, 0.20], dtype=np.float32)
        self.BEV_Z_MARGIN = 0.15
        self.DET_CLUSTER_MATCH_DIST = 2.5
        self.BEV_RECT_MIN_FILL = 0.45
        self.BEV_RECT_MAX_ASPECT = 6.0
        self.BEV_SPLIT_MIN_GAP = 1.0
        self.BEV_SPLIT_MIN_ASPECT = 2.2
        self.CLUSTER_STABILIZE_DIST = 1.5
        self.CLUSTER_STABILIZE_ALPHA = 0.35

        self.logger_ros = self.get_logger()
        
        # 2. 성능 기록용 CSV 설정
        model_name = os.path.basename(self.cfg_file).split('.')[0]
        self.log_filename = f"perf_log_{model_name}_{time.strftime('%m%d_%H%M')}.csv"
        self.csv_file = open(self.log_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame_id', 'pre_proc_ms', 'inference_ms', 'post_proc_ms', 'total_ms', 'obj_count', 'avg_score'])

        # 3. Config 로드
        if not os.path.exists(self.cfg_file):
            self.logger_ros.error(f"Config file not found: {self.cfg_file}")
            sys.exit(1)
        cfg_from_yaml_file(self.cfg_file, cfg)
        self.class_names = cfg.CLASS_NAMES
        pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

        # 4. Data Processor 초기화
        self.data_processor = DataProcessor(
            processor_configs=cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=pc_range,
            training=False,
            num_point_features=4
        )

        # 5. 모델 빌드
        self.logger_ros.info("Building Model...")
        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(self.class_names), 
            dataset=self.build_dummy_dataset(cfg, pc_range)
        )
        self.model.load_params_from_file(filename=self.ckpt_file, logger=self.logger_ros, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        # 6. 트래커 초기화 (3D Kalman Filter 기반)
        self.tracker = GlobalTracker(
            max_age=10,
            min_hits=3,
            dist_threshold=3.0,
            score_threshold=self.TRACK_SCORE_THRESH,
        )

        # 7. ROS Setup
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, qos)
        self.pub_markers = self.create_publisher(MarkerArray, '/pcdet/detections', 10)
        self.pub_bev_points = self.create_publisher(PointCloud2, '/pcdet/bev_points', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ego_speed = 0.0
        self.ego_yaw_rate = 0.0
        self.ego_motion_stamp = None
        self.last_lidar_stamp = None
        self.last_fallback_warn_frame = 0
        self.prev_cluster_observations = np.empty((0, 9), dtype=np.float32)
        self.odom_sub = self.create_subscription(Odometry, '/odometry/wheel', self.odometry_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/vectornav/imu', self.imu_callback, 10)
        self.sub_fallback = self.create_subscription(PointCloud2, self.fallback_lidar_topic, self.fallback_lidar_callback, qos)
        
        self.frame_count = 0
        self.logger_ros.info(f'Node Ready! Tracking & Logging Enabled.')

    def build_dummy_dataset(self, cfg, pc_range):
        class Dummy:
            def __init__(self):
                self.class_names = cfg.CLASS_NAMES
                self.point_feature_encoder = type('Encoder', (), {'num_point_features': 4})()
                self.point_cloud_range = pc_range
                self.depth_downsample_factor = None 
                self.dataset_cfg = cfg.DATA_CONFIG
                self.voxel_size = None
                for p in cfg.DATA_CONFIG.DATA_PROCESSOR:
                    if p['NAME'] == 'transform_points_to_voxels':
                        self.voxel_size = np.array(p['VOXEL_SIZE'], dtype=np.float32)
                        break
                self.grid_size = np.round((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(np.int64)
        return Dummy()

    def cleanup_memory(self):
        self.logger_ros.info("종료 시퀀스 시작: 로그를 저장하고 GPU 메모리를 해제합니다.")
        self.csv_file.close()
        if hasattr(self, 'model'): del self.model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def odometry_callback(self, msg):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.ego_speed = math.hypot(vx, vy)
        self.ego_motion_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def imu_callback(self, msg):
        self.ego_yaw_rate = float(msg.angular_velocity.z)
        self.ego_motion_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def fallback_lidar_callback(self, msg):
        if self.frame_count - self.last_fallback_warn_frame >= 50:
            self.last_fallback_warn_frame = self.frame_count
            self.logger_ros.warning(
                f"Ignoring {self.fallback_lidar_topic}; BEV compression is fixed to {self.lidar_topic}."
            )

    def lidar_callback(self, msg):
        stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if self.last_lidar_stamp == stamp:
            return
        self.last_lidar_stamp = stamp

        start_total = time.perf_counter()
        self.frame_count += 1

        # --- [1. Pre-processing] ---
        start_pre = time.perf_counter()
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points = np.array([[p[0], p[1], p[2], p[3]] for p in gen], dtype=np.float32)
        
        if points.shape[0] < 10: return
        points[:, 2] -= self.z_offset
        if points[:, 3].max() > 1.0: points[:, 3] /= 255.0

        input_dict = {'points': points, 'frame_id': msg.header.frame_id, 'use_lead_xyz': True}
        data_dict = self.data_processor.forward(data_dict=input_dict)
        
        data_dict['points'] = torch.from_numpy(np.pad(data_dict['points'], ((0,0),(1,0)), mode='constant')).float().cuda()
        if 'voxels' in data_dict:
            for k in ['voxels', 'voxel_num_points', 'voxel_coords']:
                data_dict[k] = torch.from_numpy(data_dict[k]).cuda()
                if k == 'voxel_coords':
                    data_dict[k] = torch.nn.functional.pad(data_dict[k], (1,0), mode='constant', value=0)
        data_dict['batch_size'] = 1
        pre_time = (time.perf_counter() - start_pre) * 1000

        # --- [2. Inference] ---
        start_inf = time.perf_counter()
        with torch.no_grad():
            torch.cuda.synchronize()
            pred_dicts, _ = self.model.forward(data_dict)
            torch.cuda.synchronize()
        inf_time = (time.perf_counter() - start_inf) * 1000

        # --- [3. Post-processing & Tracking] ---
        start_post = time.perf_counter()
        obj_count, avg_score = self.process_and_publish(pred_dicts[0], msg.header, points[:, :3])
        post_time = (time.perf_counter() - start_post) * 1000
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        # CSV 기록
        self.csv_writer.writerow([self.frame_count, pre_time, inf_time, post_time, total_time, obj_count, avg_score])
        if self.frame_count % 20 == 0:
            self.logger_ros.info(f"FPS: {1000/total_time:.1f} | Latency: {inf_time:.1f}ms | Tracked: {obj_count}")

    def apply_nms(self, boxes, scores, labels):
        if len(boxes) == 0:
            return boxes, scores, labels
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            survivors = []
            for j in order[1:]:
                same_class = labels[j] == labels[i]
                dists = np.linalg.norm(boxes[j, :2] - boxes[i, :2])
                iou = bev_iou(
                    [boxes[i, 0], boxes[i, 1], boxes[i, 3], boxes[i, 4], boxes[i, 6]],
                    [boxes[j, 0], boxes[j, 1], boxes[j, 3], boxes[j, 4], boxes[j, 6]],
                )
                if same_class and (dists < self.NMS_DIST_THRESH or iou > self.NMS_IOU_THRESH):
                    continue
                survivors.append(j)
            order = np.asarray(survivors, dtype=np.int64)
        return boxes[keep], scores[keep], labels[keep]

    def compress_points_to_bev(self, points_xyz):
        if len(points_xyz) == 0:
            return points_xyz
        return points_xyz

    def cluster_bev_points(self, slice_points):
        if len(slice_points) < self.BEV_CLUSTER_MIN_POINTS:
            return []

        tree = cKDTree(slice_points[:, :2])
        visited = np.zeros(len(slice_points), dtype=bool)
        clusters = []
        for idx in range(len(slice_points)):
            if visited[idx]:
                continue
            queue = [idx]
            visited[idx] = True
            cluster = []
            while queue:
                current = queue.pop()
                cluster.append(current)
                neighbors = tree.query_ball_point(slice_points[current, :2], r=self.BEV_CLUSTER_TOL)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(cluster) >= self.BEV_CLUSTER_MIN_POINTS:
                clusters.append(slice_points[np.asarray(cluster, dtype=np.int64)])
        return clusters

    def split_cluster_if_needed(self, cluster_points):
        if len(cluster_points) < self.BEV_CLUSTER_MIN_POINTS * 2:
            return [cluster_points]

        xy = cluster_points[:, :2]
        mean_xy = xy.mean(axis=0)
        centered_xy = xy - mean_xy
        cov = centered_xy.T @ centered_xy / max(len(centered_xy) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        secondary = eigvecs[:, np.argmin(eigvals)]
        proj_major = centered_xy @ principal
        proj_minor = centered_xy @ secondary
        major_span = float(proj_major.max() - proj_major.min())
        minor_span = max(float(proj_minor.max() - proj_minor.min()), 1e-3)
        if major_span / minor_span < self.BEV_SPLIT_MIN_ASPECT:
            return [cluster_points]

        order = np.argsort(proj_major)
        sorted_proj = proj_major[order]
        gaps = np.diff(sorted_proj)
        if len(gaps) == 0:
            return [cluster_points]
        gap_idx = int(np.argmax(gaps))
        max_gap = float(gaps[gap_idx])
        if max_gap < self.BEV_SPLIT_MIN_GAP:
            return [cluster_points]

        left_idx = order[:gap_idx + 1]
        right_idx = order[gap_idx + 1:]
        if len(left_idx) < self.BEV_CLUSTER_MIN_POINTS or len(right_idx) < self.BEV_CLUSTER_MIN_POINTS:
            return [cluster_points]
        return [cluster_points[left_idx], cluster_points[right_idx]]

    def fit_min_area_rect(self, xy):
        if len(xy) < 3:
            return None
        try:
            hull = ConvexHull(xy)
            hull_pts = xy[hull.vertices]
        except Exception:
            hull_pts = xy
        if len(hull_pts) < 3:
            return None

        best = None
        best_area = np.inf
        edges = np.roll(hull_pts, -1, axis=0) - hull_pts
        for edge in edges:
            angle = math.atan2(edge[1], edge[0])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)
            proj = hull_pts @ rot.T
            min_proj = proj.min(axis=0)
            max_proj = proj.max(axis=0)
            dims = max_proj - min_proj
            area = float(dims[0] * dims[1])
            if area < best_area:
                best_area = area
                best = (angle, min_proj, max_proj)

        if best is None:
            return None

        angle, min_proj, max_proj = best
        dims_xy = np.maximum(max_proj - min_proj, np.array([0.6, 0.6], dtype=np.float32))
        center_proj = 0.5 * (min_proj + max_proj)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        inv_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        center_xy = center_proj @ inv_rot.T
        yaw = angle
        if dims_xy[0] < dims_xy[1]:
            dims_xy = dims_xy[::-1]
            yaw = wrap_angle(yaw + np.pi / 2)
        return center_xy, dims_xy, yaw

    def fit_cluster_box(self, cluster_points, default_label=0, default_score=1.0):
        if len(cluster_points) < self.BEV_FIT_MIN_POINTS:
            return None

        xy = cluster_points[:, :2]
        label = int(default_label)
        cls_name = self.class_names[label - 1].lower() if 0 < label <= len(self.class_names) else ""
        is_vehicle_like = any(token in cls_name for token in ("car", "truck", "bus", "van"))

        rect_fit = self.fit_min_area_rect(xy) if is_vehicle_like else None
        if rect_fit is not None:
            fitted_center_xy, dims_xy, yaw = rect_fit
        else:
            mean_xy = xy.mean(axis=0)
            centered_xy = xy - mean_xy
            cov = centered_xy.T @ centered_xy / max(len(centered_xy) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, np.argmax(eigvals)]
            yaw = math.atan2(principal[1], principal[0])

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            rot = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float32)
            proj = centered_xy @ rot.T
            min_proj = proj.min(axis=0)
            max_proj = proj.max(axis=0)
            dims_xy = np.maximum(max_proj - min_proj, np.array([0.6, 0.6], dtype=np.float32))
            center_proj = 0.5 * (min_proj + max_proj)
            fitted_center_xy = mean_xy + center_proj @ np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32).T

        rect_area = max(float(dims_xy[0] * dims_xy[1]), 1e-3)
        try:
            hull = ConvexHull(xy)
            hull_area = float(hull.volume)
        except Exception:
            hull_area = rect_area
        fill_ratio = min(hull_area / rect_area, 1.0)
        aspect_ratio = max(float(dims_xy[0] / max(dims_xy[1], 1e-3)), float(dims_xy[1] / max(dims_xy[0], 1e-3)))
        if fill_ratio < self.BEV_RECT_MIN_FILL or aspect_ratio > self.BEV_RECT_MAX_ASPECT:
            return None

        z_min = np.percentile(cluster_points[:, 2], 5)
        z_max = np.percentile(cluster_points[:, 2], 95)
        dims_xy = dims_xy + self.BEV_BOX_MARGIN
        dz = max(float(z_max - z_min) + self.BEV_Z_MARGIN, 0.6)
        z_center = 0.5 * (z_min + z_max)

        return np.array([
            fitted_center_xy[0], fitted_center_xy[1], z_center,
            dims_xy[0], dims_xy[1], dz, yaw,
            default_label, default_score
        ], dtype=np.float32)

    def build_cluster_observations(self, points_xyz):
        bev_points = self.compress_points_to_bev(points_xyz)
        clusters = self.cluster_bev_points(bev_points)
        if not clusters:
            return np.empty((0, 9), dtype=np.float32), bev_points

        observations = []
        for cluster_points in clusters:
            for sub_cluster in self.split_cluster_if_needed(cluster_points):
                fitted = self.fit_cluster_box(sub_cluster)
                if fitted is not None:
                    observations.append(fitted)

        if not observations:
            return np.empty((0, 9), dtype=np.float32), bev_points
        return np.asarray(observations, dtype=np.float32), bev_points

    def assign_detector_semantics(self, cluster_observations, detections):
        if len(cluster_observations) == 0:
            return cluster_observations
        if len(detections) == 0:
            return cluster_observations

        enriched = cluster_observations.copy()
        used_det_indices = set()
        for cluster_idx, cluster_box in enumerate(enriched):
            best_det_idx = None
            best_cost = np.inf
            for det_idx, det in enumerate(detections):
                if det_idx in used_det_indices:
                    continue
                center_dist = np.linalg.norm(cluster_box[:2] - det[:2])
                if center_dist > self.DET_CLUSTER_MATCH_DIST:
                    continue
                yaw_diff = abs(wrap_angle(float(cluster_box[6] - det[6])))
                cost = center_dist + 0.3 * yaw_diff
                if cost < best_cost:
                    best_cost = cost
                    best_det_idx = det_idx

            if best_det_idx is None:
                continue

            used_det_indices.add(best_det_idx)
            enriched[cluster_idx, 7] = detections[best_det_idx, 7]
            enriched[cluster_idx, 8] = detections[best_det_idx, 8]
        return enriched

    def stabilize_cluster_observations(self, cluster_observations):
        if len(cluster_observations) == 0:
            self.prev_cluster_observations = cluster_observations
            return cluster_observations
        if len(self.prev_cluster_observations) == 0:
            self.prev_cluster_observations = cluster_observations.copy()
            return cluster_observations

        stabilized = cluster_observations.copy()
        used_prev = set()
        for idx, obs in enumerate(stabilized):
            best_prev = None
            best_cost = np.inf
            for prev_idx, prev_obs in enumerate(self.prev_cluster_observations):
                if prev_idx in used_prev:
                    continue
                center_dist = np.linalg.norm(obs[:2] - prev_obs[:2])
                if center_dist > self.CLUSTER_STABILIZE_DIST:
                    continue
                yaw_diff = abs(wrap_angle(float(obs[6] - prev_obs[6])))
                cost = center_dist + 0.25 * yaw_diff
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev_idx

            if best_prev is None:
                continue

            used_prev.add(best_prev)
            prev_obs = self.prev_cluster_observations[best_prev]
            alpha = self.CLUSTER_STABILIZE_ALPHA
            stabilized[idx, 0:3] = (1.0 - alpha) * prev_obs[0:3] + alpha * obs[0:3]
            stabilized[idx, 3:6] = np.maximum(
                0.6,
                (1.0 - alpha) * prev_obs[3:6] + alpha * obs[3:6],
            )
            yaw_delta = wrap_angle(float(obs[6] - prev_obs[6]))
            stabilized[idx, 6] = wrap_angle(float(prev_obs[6] + alpha * yaw_delta))
            if int(stabilized[idx, 7]) <= 0 and int(prev_obs[7]) > 0:
                stabilized[idx, 7] = prev_obs[7]
                stabilized[idx, 8] = prev_obs[8]

        self.prev_cluster_observations = stabilized.copy()
        return stabilized

    def publish_bev_points(self, points_xyz, frame_id, stamp):
        if len(points_xyz) == 0:
            return

        bev_points = np.zeros((len(points_xyz), 4), dtype=np.float32)
        bev_points[:, 0:2] = points_xyz[:, 0:2]
        bev_points[:, 2] = 0.0
        if points_xyz.shape[1] > 2:
            z = points_xyz[:, 2]
            z_range = max(float(z.max() - z.min()), 1e-3)
            bev_points[:, 3] = (z - z.min()) / z_range

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp
        msg = pc2.create_cloud(header, fields, bev_points.tolist())
        self.pub_bev_points.publish(msg)

    def process_and_publish(self, pred_dict, header, points_xyz):
        boxes = pred_dict['pred_boxes'].cpu().numpy()
        scores = pred_dict['pred_scores'].cpu().numpy()
        labels = pred_dict['pred_labels'].cpu().numpy()
        timestamp = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        
        mask = (scores > self.SCORE_THRESH) & (np.linalg.norm(boxes[:, :2], axis=1) < self.MAX_DETECTION_RANGE)
        boxes, scores, labels = self.apply_nms(boxes[mask], scores[mask], labels[mask])
        
        if len(boxes) == 0:
            detections = np.empty((0, 9), dtype=np.float32)
        else:
            detections = np.hstack([boxes, labels.reshape(-1,1), scores.reshape(-1,1)]).astype(np.float32, copy=False)

        ego_motion = {
            'speed': self.ego_speed,
            'yaw_rate': self.ego_yaw_rate,
        }

        # TF Transform & Tracking
        target_frame = 'odom'
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                header.frame_id,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.05)
            )
            r = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            rot_mat = r.as_matrix()
            
            if len(detections) > 0:
                xyz = detections[:, 0:3]
                detections[:, 0:3] = np.dot(xyz, rot_mat.T) + np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
                detections[:, 6] += r.as_euler('zyx')[0]
            if len(points_xyz) > 0:
                points_xyz = (points_xyz @ rot_mat.T) + np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])

            cluster_observations, bev_slice_points = self.build_cluster_observations(points_xyz)
            cluster_observations = self.assign_detector_semantics(cluster_observations, detections)
            detections = self.stabilize_cluster_observations(cluster_observations)
            self.publish_bev_points(bev_slice_points, target_frame, header.stamp)
            
            # odom 좌표계에서 트래킹 업데이트
            tracked_objects = self.tracker.update(detections, timestamp=timestamp, ego_motion=ego_motion)
            self.publish_markers(tracked_objects, target_frame, header.stamp)
            
        except Exception as e:
            # TF 실패 시 lidar 좌표계에서 트래킹 (ID 유지가 불안정할 수 있음)
            if self.frame_count % 20 == 0:
                self.logger_ros.warning(f"TF lookup failed, falling back to sensor frame tracking: {e}")
            cluster_observations, bev_slice_points = self.build_cluster_observations(points_xyz)
            cluster_observations = self.assign_detector_semantics(cluster_observations, detections)
            detections = self.stabilize_cluster_observations(cluster_observations)
            self.publish_bev_points(bev_slice_points, header.frame_id, header.stamp)
            tracked_objects = self.tracker.update(detections, timestamp=timestamp, ego_motion=ego_motion)
            self.publish_markers(tracked_objects, header.frame_id, header.stamp)

        # 통계용 데이터 반환
        avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        return len(tracked_objects), avg_score

    def publish_markers(self, tracked_objects, frame_id, stamp):
        ma = MarkerArray()

        for obj in tracked_objects:
            # tracker output: [x, y, z, id, yaw, label, dx, dy, dz, time_since_update]
            x, y, z, tid, yaw, label, dx, dy, dz, age = obj
            x, y, z, dx, dy, dz, yaw = map(float, [x, y, z, dx, dy, dz, yaw])
            tid, label = int(tid), int(label)

            cls_name = self.class_names[label-1] if 0 < label <= len(self.class_names) else "Unknown"
            r, g, b = (0.0, 1.0, 0.0) if 'car' in cls_name.lower() else (1.0, 1.0, 0.0) if 'ped' in cls_name.lower() else (1.0, 1.0, 1.0)
            
            # 고스트 박스(예측값) 시각화
            alpha = 0.8 if age == 0 else max(0.1, 0.6 - (age * 0.1))

            # Box
            m = Marker()
            m.header.frame_id, m.header.stamp = frame_id, stamp
            m.ns, m.id, m.type, m.action = "tracked_obj", tid, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            m.scale.x, m.scale.y, m.scale.z = dx, dy, dz
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, alpha
            m.lifetime = Duration(seconds=0.6).to_msg()
            ma.markers.append(m)

            # Text (ID & Class)
            t = Marker()
            t.header.frame_id, t.header.stamp = frame_id, stamp
            t.ns, t.id, t.type, t.action = "ids", tid + 1000, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, z + dz/2 + 0.5
            t.scale.z, t.text = 0.5, f"{cls_name} {tid}"
            t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
            t.lifetime = Duration(seconds=0.6).to_msg()
            ma.markers.append(t)

        self.pub_markers.publish(ma)

def main():
    rclpy.init()
    node = OpenPCDetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup_memory()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
