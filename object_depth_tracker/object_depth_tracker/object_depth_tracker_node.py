#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from cv_bridge import CvBridge
import message_filters
import time
import struct
import ctypes
import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from perception_interface.msg import DetectionArray
import tf2_ros

DEFAULT_QUEUE_SIZE = 40

class ObjectDepthTracker(Node):
    def __init__(self):
        super().__init__("object_depth_tracker_node")
        self.bridge = CvBridge()

        # --------------------
        # Parameters (dynamic-friendly)
        # --------------------
        self.declare_parameter("track_class_ids", [0, 2, 5, 7])  # person, car, bus, truck
        self.declare_parameter("detection_topic", "yolo/detections")
        self.declare_parameter("depth_topic", "/depth_anything/depth_registered/image_rect")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/left/camera_info")
        self.declare_parameter("min_mask_pixels", 30)
        self.declare_parameter("min_cluster_points", 50)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("slop_sec", 0.08)            # sync slop (초) 기본 80ms
        self.declare_parameter("queue_size", DEFAULT_QUEUE_SIZE)
        self.declare_parameter("sample_points_max", 300)
        self.declare_parameter("dbscan_min_samples", 10)
        self.declare_parameter("dbscan_eps_per_meter", 0.02)  # eps ≈ 0.02 * median_z
        self.declare_parameter("bev_frame", "base_link")      # BEV 퍼블리시용 프레임
        self.declare_parameter("output_frame", "odom")        # 출력 좌표계
        self.declare_parameter("use_morphology", True)
        self.declare_parameter("use_median_blur_depth", True)
        self.declare_parameter("median_blur_ksize", 3)

        # Parse track_class_ids parameter
        track_class_ids_param = self.get_parameter("track_class_ids").value
        if isinstance(track_class_ids_param, list):
            self.track_ids = [int(x) for x in track_class_ids_param]
        elif isinstance(track_class_ids_param, str):
            try:
                self.track_ids = ast.literal_eval(track_class_ids_param)
                if not isinstance(self.track_ids, list):
                    self.track_ids = [int(track_class_ids_param)]
            except (ValueError, SyntaxError):
                self.get_logger().error(
                    f"Invalid track_class_ids format: {track_class_ids_param}. Using default [0, 2, 5, 7]"
                )
                self.track_ids = [0, 2, 5, 7]
        else:
            self.get_logger().error(
                f"Unexpected track_class_ids type: {type(track_class_ids_param)}. Using default [0, 2, 5, 7]"
            )
            self.track_ids = [0, 2, 5, 7]

        detection_topic = self.get_parameter("detection_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        self.min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.slop_sec = float(self.get_parameter("slop_sec").value)
        self.sample_points_max = int(self.get_parameter("sample_points_max").value)
        self.dbscan_min_samples = int(self.get_parameter("dbscan_min_samples").value)
        self.dbscan_eps_per_meter = float(self.get_parameter("dbscan_eps_per_meter").value)
        self.bev_frame = str(self.get_parameter("bev_frame").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.use_morphology = bool(self.get_parameter("use_morphology").value)
        self.use_median_blur_depth = bool(self.get_parameter("use_median_blur_depth").value)
        self.median_blur_ksize = int(self.get_parameter("median_blur_ksize").value)
        self.queue_size = int(self.get_parameter("queue_size").value)

        # Class name → COCO ID
        self._init_class_mapping()

        # --------------------
        # TF
        # --------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --------------------
        # QoS profiles
        # --------------------
        # Depth/CameraInfo는 일반적으로 BEST_EFFORT
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        # 감지(Detections)는 신뢰성 있는 RELIABLE가 흔함
        detection_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # --------------------
        # Subscribers (message_filters sync)
        # --------------------
        self.sub_detections = message_filters.Subscriber(
            self, DetectionArray, detection_topic, qos_profile=detection_qos
        )
        self.sub_depth = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=sensor_qos
        )
        self.sub_cinfo = message_filters.Subscriber(
            self, CameraInfo, camera_info_topic, qos_profile=sensor_qos
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_detections, self.sub_depth, self.sub_cinfo],
            queue_size=self.queue_size,
            slop=self.slop_sec,
            allow_headerless=False
        )
        self.sync.registerCallback(self.sync_cb)

        # --------------------
        # Publishers
        # --------------------
        self.marker_pub = self.create_publisher(MarkerArray, "tracked_points", QoSProfile(depth=2))
        self.bev_pub = self.create_publisher(MarkerArray, "tracked_points_bev", QoSProfile(depth=2))
        self.pointcloud_pub = self.create_publisher(PointCloud2, "debug_pointclouds", QoSProfile(depth=1))

        self.get_logger().info(f"3D converter ready (track ids={self.track_ids})")

        # Debug pointclouds per frame
        self.debug_pointclouds = []

        # Class color map (reused)
        self.class_colors = {
            'person': (1.0, 0.0, 0.0, 0.3),
            'car': (0.0, 1.0, 0.0, 0.3),
            'truck': (0.0, 0.0, 1.0, 0.3),
            'bus': (1.0, 1.0, 0.0, 0.3),
            'bicycle': (1.0, 0.0, 1.0, 0.3),
            'motorcycle': (0.0, 1.0, 1.0, 0.3)
        }

    # --------------------
    # Utility: RGB packing for PointCloud2 'rgb' field (float32)
    # --------------------
    @staticmethod
    def _rgb_to_packed_float(r: float, g: float, b: float) -> float:
        ri = int(max(0, min(255, r * 255.0)))
        gi = int(max(0, min(255, g * 255.0)))
        bi = int(max(0, min(255, b * 255.0)))
        rgb_uint32 = (ri << 16) | (gi << 8) | bi
        # reinterpret uint32 bits as float32
        return ctypes.c_float.from_buffer(ctypes.c_uint32(rgb_uint32)).value

    # --------------------
    # Depth & mask preprocessing
    # --------------------
    def _preprocess_depth_image(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        depth[(depth <= 0) | (depth > self.max_depth_m)] = 0.0
        if self.use_median_blur_depth and self.median_blur_ksize >= 3 and self.median_blur_ksize % 2 == 1:
            try:
                depth = cv2.medianBlur(depth, self.median_blur_ksize)
            except Exception as e:
                self.get_logger().warning(f"Median blur failed: {e}")
        return depth

    def _preprocess_segmentation_mask(self, mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8)
        if self.use_morphology:
            try:
                k = np.ones((3, 3), np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            except Exception as e:
                self.get_logger().warning(f"Morphology failed: {e}")
        # Largest component only
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = int(np.argmax(areas)) + 1
            return (labels == largest_label)
        return m > 0

    @staticmethod
    def _sample_points(points_3d: np.ndarray, max_points=300) -> np.ndarray:
        if len(points_3d) <= max_points:
            return points_3d
        idx = np.random.choice(len(points_3d), size=max_points, replace=False)
        return points_3d[idx]

    @staticmethod
    def _basic_outlier_removal(points_3d: np.ndarray) -> np.ndarray:
        if len(points_3d) < 10:
            return points_3d
        centroid = np.mean(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)
        q75 = np.percentile(distances, 75)
        thr = q75 * 2.0
        return points_3d[distances <= thr]

    def _dbscan_clustering(self, points_3d: np.ndarray,
                           min_samples: int,
                           eps_per_meter: float) -> np.ndarray:
        if len(points_3d) < max(2 * min_samples, self.min_cluster_points):
            return points_3d
        try:
            from sklearn.cluster import DBSCAN
            median_z = float(np.median(points_3d[:, 2]))
            eps = max(0.05, eps_per_meter * max(0.5, median_z))  # 최소 5cm
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clustering.fit_predict(points_3d)

            valid = labels != -1
            uniq, counts = np.unique(labels[valid], return_counts=True)
            if len(uniq) == 0:
                return self._basic_outlier_removal(points_3d)
            main_label = uniq[int(np.argmax(counts))]
            main_points = points_3d[labels == main_label]
            if len(main_points) < self.min_cluster_points:
                # 느슨하게 한 번 더
                return self._dbscan_clustering(points_3d, max(5, min_samples // 2), eps_per_meter * 1.5)
            return main_points
        except ImportError:
            self.get_logger().warning("sklearn not available, using fallback outlier removal")
            return self._basic_outlier_removal(points_3d)
        except Exception as e:
            self.get_logger().warning(f"DBSCAN clustering failed: {e}, using fallback")
            return self._basic_outlier_removal(points_3d)

    def _compute_oriented_bbox(self, points_3d: np.ndarray):
        try:
            pts = points_3d - np.mean(points_3d, axis=0, keepdims=True)
            # PCA via SVD
            _, _, Vt = np.linalg.svd(pts, full_matrices=False)
            R = Vt.T  # (3x3)
            # project points into principal axes
            proj = pts @ R
            mins = np.min(proj, axis=0)
            maxs = np.max(proj, axis=0)
            sizes = (maxs - mins)
            sizes = np.maximum(sizes, 0.1)
            center_local = 0.5 * (maxs + mins)
            centroid = np.mean(points_3d, axis=0)
            center_world = centroid + R @ center_local

            # Convert rotation matrix -> quaternion for RViz Marker (we'll only store R here; marker later)
            return {
                'center': center_world,
                'size': sizes,
                'rotation': R,   # 3x3
                'point_count': len(points_3d)
            }
        except Exception as e:
            # Fallback AABB in camera coords
            x_min, y_min, z_min = np.min(points_3d, axis=0)
            x_max, y_max, z_max = np.max(points_3d, axis=0)
            center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            size = np.maximum(np.array([x_max - x_min, y_max - y_min, z_max - z_min]), 0.1)
            self.get_logger().warning(f"PCA OBB failed, using AABB: {e}")
            return {
                'center': center,
                'size': size,
                'rotation': np.eye(3),
                'point_count': len(points_3d)
            }

    def _estimate_3d_bbox_from_mask(self, mask: np.ndarray, depth: np.ndarray, fx, fy, cx, cy):
        mask_clean = self._preprocess_segmentation_mask(mask)
        depth_clean = self._preprocess_depth_image(depth)

        vv, uu = np.nonzero(mask_clean)
        if vv.size < self.min_mask_pixels:
            return None

        valid_mask = (vv < depth_clean.shape[0]) & (uu < depth_clean.shape[1])
        if valid_mask.sum() < self.min_mask_pixels:
            return None

        vv_valid, uu_valid = vv[valid_mask], uu[valid_mask]
        z_vals = depth_clean[vv_valid, uu_valid]
        good = (z_vals > 0.1) & (z_vals < self.max_depth_m)
        if good.sum() < self.min_mask_pixels:
            return None

        uu_good, vv_good = uu_valid[good], vv_valid[good]
        z_good = z_vals[good]

        x_3d = (uu_good - cx) * z_good / fx
        y_3d = (vv_good - cy) * z_good / fy
        z_3d = z_good
        points_3d_original = np.column_stack([x_3d, y_3d, z_3d])

        points_3d_sampled = self._sample_points(points_3d_original, max_points=self.sample_points_max)
        points_3d_filtered = self._dbscan_clustering(
            points_3d_sampled, self.dbscan_min_samples, self.dbscan_eps_per_meter
        )
        if len(points_3d_filtered) < self.min_cluster_points:
            return None

        debug_pc = {
            'points_original': points_3d_original.copy(),
            'points_sampled': points_3d_sampled.copy(),
            'points_clustered': points_3d_filtered.copy(),
            'center': np.mean(points_3d_filtered, axis=0),
            'preprocessing_stats': {
                'original_points': len(points_3d_original),
                'sampled_points': len(points_3d_sampled),
                'clustered_points': len(points_3d_filtered)
            }
        }

        bbox_result = self._compute_oriented_bbox(points_3d_filtered)
        if bbox_result:
            bbox_result['debug_pointcloud'] = debug_pc
        return bbox_result

    def _estimate_3d_bbox_from_2d_bbox(self, bbox_points, depth, fx, fy, cx, cy, class_name):
        try:
            # bbox_points: 4 points(순서 불문)
            xs = [int(p.x) for p in bbox_points[:4]]
            ys = [int(p.y) for p in bbox_points[:4]]
            x1, x2 = max(0, min(xs)), min(depth.shape[1] - 1, max(xs))
            y1, y2 = max(0, min(ys)), min(depth.shape[0] - 1, max(ys))
            if x2 <= x1 or y2 <= y1:
                return None

            depth_roi = depth[y1:y2+1, x1:x2+1]
            if depth_roi.size == 0:
                return None

            # 샘플링 & 필터링
            h, w = depth_roi.shape
            u_samples = np.linspace(0, w - 1, 10, dtype=int)
            v_samples = np.linspace(0, h - 1, 10, dtype=int)
            ds = []
            for u in u_samples:
                for v in v_samples:
                    d = float(depth_roi[v, u])
                    if 0.1 < d < 50.0:
                        ds.append(d)
            ds = np.array(ds, dtype=np.float32)
            if ds.size < 10:
                return None
            ds.sort()
            lo, hi = int(0.1 * ds.size), int(0.9 * ds.size)
            center_depth = float(np.median(ds[lo:hi])) if hi > lo else float(np.median(ds))

            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0
            cx3 = (u_center - cx) * center_depth / fx
            cy3 = (v_center - cy) * center_depth / fy
            cz3 = center_depth

            # Class-specific rough priors [w, h, l]
            size_priors = {
                'person': [0.6, 1.7, 0.3],
                'car': [1.8, 1.5, 4.5],
                'truck': [2.5, 3.0, 8.0],
                'bus': [2.8, 3.2, 12.0],
                'bicycle': [0.6, 1.2, 1.8],
                'motorcycle': [0.8, 1.3, 2.2]
            }
            default_size = [1.0, 1.0, 1.0]
            prior = size_priors.get(class_name.lower(), default_size)

            scale_factor = max((x2 - x1) / 100.0, (y2 - y1) / 100.0) * (center_depth / 5.0)
            scaled = np.array(prior, dtype=np.float32) * max(0.5, min(2.0, scale_factor))

            return {
                'center': np.array([cx3, cy3, cz3], dtype=np.float32),
                'size': scaled,
                'rotation': np.eye(3, dtype=np.float32)
            }
        except Exception as e:
            self.get_logger().debug(f"3D bbox estimation from 2D failed: {e}")
            return None

    def _init_class_mapping(self):
        self.class_map = {
            "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
            "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
            "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
            "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
            "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
            "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
            "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34,
            "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38,
            "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43,
            "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48,
            "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
            "donut": 54, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58,
            "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63,
            "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
            "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
            "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78,
            "toothbrush": 79
        }

    # --------------------
    # TF helper
    # --------------------
    def _transform_point(self, xyz: np.ndarray, from_frame: str, to_frame: str, stamp) -> np.ndarray:
        ps = PointStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = from_frame
        ps.point.x, ps.point.y, ps.point.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        try:
            out = self.tf_buffer.transform(ps, to_frame, timeout=Duration(seconds=0.1))
            return np.array([out.point.x, out.point.y, out.point.z], dtype=np.float32)
        except Exception as e:
            self.get_logger().warning(f"TF transform {from_frame}->{to_frame} failed: {e}")
            return xyz.astype(np.float32)

    # --------------------
    # Main sync callback
    # --------------------
    def sync_cb(self, detection_msg, depth_msg, caminfo_msg):
        t0 = time.perf_counter()

        # Clear debug data from previous frame
        self.debug_pointclouds.clear()

        # Camera parameters
        fx, fy = caminfo_msg.k[0], caminfo_msg.k[4]
        cx, cy = caminfo_msg.k[2], caminfo_msg.k[5]

        # Convert depth image
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        stamp = depth_msg.header.stamp
        camera_frame = depth_msg.header.frame_id or "camera_frame"

        tracked_objects = []

        for det in detection_msg.detections:
            if det.track_id < 0:
                continue

            class_name = det.class_name.lower()
            cls_id = self.class_map.get(class_name, -1)
            if cls_id == -1:
                self.get_logger().debug(f"Unknown class '{det.class_name}', skipping", throttle_duration_sec=5.0)
                continue
            if cls_id not in self.track_ids:
                continue

            bbox_3d_info = None
            X = Y = Z = None

            # Mask path (preferred)
            if getattr(det, "mask", None) and det.mask.width > 0 and det.mask.height > 0:
                try:
                    mask = self.bridge.imgmsg_to_cv2(det.mask, "mono8") > 0
                    bbox_3d_info = self._estimate_3d_bbox_from_mask(mask, depth, fx, fy, cx, cy)
                    if bbox_3d_info is None:
                        continue
                    center_3d = bbox_3d_info['center']
                    X, Y, Z = float(center_3d[0]), float(center_3d[1]), float(center_3d[2])
                except Exception as e:
                    self.get_logger().warning(f"Mask processing failed: {e}")
                    continue
            else:
                # 2D bbox fallback
                try:
                    if len(det.bounding_box.points) >= 4:
                        bbox = det.bounding_box.points
                        bbox_3d_info = self._estimate_3d_bbox_from_2d_bbox(bbox, depth, fx, fy, cx, cy, det.class_name)
                        if bbox_3d_info is None:
                            # Simple centroid fallback
                            xs = [int(p.x) for p in bbox[:4]]
                            ys = [int(p.y) for p in bbox[:4]]
                            x1, x2 = max(0, min(xs)), min(depth.shape[1] - 1, max(xs))
                            y1, y2 = max(0, min(ys)), min(depth.shape[0] - 1, max(ys))
                            u_bar = int((x1 + x2) / 2)
                            v_bar = int((y1 + y2) / 2)
                            Z = float(depth[v_bar, u_bar])
                            if Z <= 0 or Z > 50.0:
                                continue
                            X = (u_bar - cx) * Z / fx
                            Y = (v_bar - cy) * Z / fy
                        else:
                            center_3d = bbox_3d_info['center']
                            X, Y, Z = float(center_3d[0]), float(center_3d[1]), float(center_3d[2])
                    else:
                        continue
                except (IndexError, ValueError) as e:
                    self.get_logger().warning(f"Bounding box processing failed: {e}")
                    continue

            if any(abs(v) > 100.0 for v in (X, Y, Z)):
                self.get_logger().debug(
                    f"Unrealistic 3D coordinates: ({X:.2f}, {Y:.2f}, {Z:.2f})",
                    throttle_duration_sec=5.0
                )
                continue

            # Transform to odom frame
            xyz_camera = np.array([X, Y, Z], dtype=np.float32)
            xyz_odom = self._transform_point(xyz_camera, camera_frame, self.output_frame, stamp)
            
            # Transform bbox center to odom frame if available
            bbox_3d_info_transformed = None
            if bbox_3d_info is not None:
                bbox_3d_info_transformed = bbox_3d_info.copy()
                center_camera = bbox_3d_info['center']
                center_odom = self._transform_point(center_camera, camera_frame, self.output_frame, stamp)
                bbox_3d_info_transformed['center'] = center_odom
            
            tracked_objects.append({
                'id': det.track_id,
                'xyz': xyz_odom,
                'class_name': det.class_name,
                'cls_id': cls_id,
                'bbox': bbox_3d_info_transformed,  # include center/size/rotation for marker
                'confidence': det.confidence,
                'frame': self.output_frame  # Changed to output_frame
            })

            if bbox_3d_info and 'debug_pointcloud' in bbox_3d_info:
                debug_pc = bbox_3d_info['debug_pointcloud']
                debug_pc['class_name'] = det.class_name
                debug_pc['track_id'] = det.track_id
                self.debug_pointclouds.append(debug_pc)

        # Debug: track IDs
        self.get_logger().info(
            f"Publishing tracks with IDs: {[obj['id'] for obj in tracked_objects]}",
            throttle_duration_sec=2.0
        )

        # Publish markers (camera frame)
        marker_arr = self._to_markers(tracked_objects, stamp)
        self.marker_pub.publish(marker_arr)

        # Publish BEV markers (transform to bev_frame, Z=0)
        bev_arr = self._create_bev_markers(tracked_objects, stamp)
        self.bev_pub.publish(bev_arr)

        # Publish debug point clouds
        if self.debug_pointclouds:
            debug_pc_msg = self._create_debug_pointcloud_msg(stamp, camera_frame)
            self.pointcloud_pub.publish(debug_pc_msg)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"[3d_converter] Frame: {elapsed_ms:.1f}ms, Detections: {len(detection_msg.detections)}, "
            f"Valid 3D objects: {len(tracked_objects)}",
            throttle_duration_sec=1.0
        )

    # --------------------
    # Marker builders
    # --------------------
    def _to_markers(self, tracked_objects, stamp):
        arr = MarkerArray()

        # Clear all
        clear = Marker()
        clear.header.stamp = stamp
        # 모든 오브젝트는 각자의 frame을 가질 수 있지만, RViz의 DeleteAll은 frame 무시로 동작함.
        # 안전하게 depth_msg 프레임 기준 사용
        clear.header.frame_id = tracked_objects[0]['frame'] if tracked_objects else "camera_frame"
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for obj in tracked_objects:
            frame = obj['frame']
            x, y, z = obj['xyz'].tolist()
            cls = obj['class_name'].lower()
            color = self.class_colors.get(cls, (0.5, 0.5, 0.5, 0.8))

            # Center sphere
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "tracked_pts"
            m.id = obj['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 0.8
            arr.markers.append(m)

            # 3D Bounding Box (with rotation)
            if obj['bbox'] is not None:
                bbox_info = obj['bbox']
                size = bbox_info['size']
                R = bbox_info['rotation']

                bbox = Marker()
                bbox.header.stamp = stamp
                bbox.header.frame_id = frame
                bbox.ns = "bbox_3d"
                bbox.id = obj['id']
                bbox.type = Marker.CUBE
                bbox.action = Marker.ADD
                bbox.pose.position.x = float(x)
                bbox.pose.position.y = float(y)
                bbox.pose.position.z = float(z)

                # rotation matrix -> quaternion
                q = self._quat_from_rotmat(R)
                bbox.pose.orientation.x = float(q[0])
                bbox.pose.orientation.y = float(q[1])
                bbox.pose.orientation.z = float(q[2])
                bbox.pose.orientation.w = float(q[3])

                bbox.scale.x = max(0.1, float(size[0]))
                bbox.scale.y = max(0.1, float(size[1]))
                bbox.scale.z = max(0.1, float(size[2]))
                bbox.color.r, bbox.color.g, bbox.color.b, bbox.color.a = color
                arr.markers.append(bbox)

            # Text label
            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = frame
            text.ns = "labels"
            text.id = obj['id']
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x)
            text.pose.position.y = float(y)
            text.pose.position.z = float(z + 0.4)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            size_str = ""
            if obj['bbox'] is not None:
                s = obj['bbox']['size']
                size_str = f" ({s[0]:.1f}×{s[1]:.1f}×{s[2]:.1f}m)"
            text.text = f"ID:{obj['id']} {obj['class_name']}{size_str}"
            arr.markers.append(text)

        return arr

    def _create_bev_markers(self, tracked_objects, stamp):
        arr = MarkerArray()

        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = self.bev_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for obj in tracked_objects:
            # Transform to bev_frame
            xyz_bev = self._transform_point(obj['xyz'], obj['frame'], self.bev_frame, stamp)
            x, y = float(xyz_bev[0]), float(xyz_bev[1])
            cls = obj['class_name'].lower()
            color = self.class_colors.get(cls, (0.5, 0.5, 0.5, 0.8))

            # BEV center (cylinder)
            bev = Marker()
            bev.header.stamp = stamp
            bev.header.frame_id = self.bev_frame
            bev.ns = "bev_tracked_pts"
            bev.id = obj['id']
            bev.type = Marker.CYLINDER
            bev.action = Marker.ADD
            bev.pose.position.x = x
            bev.pose.position.y = y
            bev.pose.position.z = 0.0
            bev.pose.orientation.w = 1.0
            bev.scale.x = bev.scale.y = 0.3
            bev.scale.z = 0.1
            bev.color.r, bev.color.g, bev.color.b, bev.color.a = 0.0, 1.0, 0.0, 0.8
            arr.markers.append(bev)

            # BEV text
            bev_text = Marker()
            bev_text.header.stamp = stamp
            bev_text.header.frame_id = self.bev_frame
            bev_text.ns = "bev_labels"
            bev_text.id = obj['id']
            bev_text.type = Marker.TEXT_VIEW_FACING
            bev_text.action = Marker.ADD
            bev_text.pose.position.x = x
            bev_text.pose.position.y = y
            bev_text.pose.position.z = 0.3
            bev_text.pose.orientation.w = 1.0
            bev_text.scale.z = 0.15
            bev_text.color.r = bev_text.color.g = bev_text.color.b = bev_text.color.a = 1.0
            bev_text.text = f"ID:{obj['id']}"
            arr.markers.append(bev_text)

        return arr

    @staticmethod
    def _quat_from_rotmat(R: np.ndarray):
        # Convert 3x3 rotation matrix to quaternion (x,y,z,w) with Python floats
        m = R
        t = float(np.trace(m))
        if t > 0.0:
            S = float(np.sqrt(t + 1.0) * 2.0)
            qw = 0.25 * S
            qx = (m[2,1] - m[1,2]) / S
            qy = (m[0,2] - m[2,0]) / S
            qz = (m[1,0] - m[0,1]) / S
        elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = float(np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0)
            qw = (m[2,1] - m[1,2]) / S
            qx = 0.25 * S
            qy = (m[0,1] + m[1,0]) / S
            qz = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = float(np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0)
            qw = (m[0,2] - m[2,0]) / S
            qx = (m[0,1] + m[1,0]) / S
            qy = 0.25 * S
            qz = (m[1,2] + m[2,1]) / S
        else:
            S = float(np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0)
            qw = (m[1,0] - m[0,1]) / S
            qx = (m[0,2] + m[2,0]) / S
            qy = (m[1,2] + m[2,1]) / S
            qz = 0.25 * S

        # normalize & sanitize
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        n = float(np.linalg.norm(q))
        if n > 0.0 and np.isfinite(n):
            q /= n
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        # return Python floats (not numpy scalars)
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    # --------------------
    # Debug PointCloud2 (with 'rgb' float field)
    # --------------------
    def _create_debug_pointcloud_msg(self, stamp, frame):
        all_points = []  # list of (x,y,z,rgb_float)

        for i, debug_pc in enumerate(self.debug_pointclouds):
            # Original points (light red)
            if 'points_original' in debug_pc:
                for p in debug_pc['points_original'][::5]:
                    rgbf = self._rgb_to_packed_float(1.0, 0.3, 0.3)
                    all_points.append((float(p[0]), float(p[1]), float(p[2]), rgbf))
            # Sampled (yellow)
            if 'points_sampled' in debug_pc:
                for p in debug_pc['points_sampled'][::2]:
                    rgbf = self._rgb_to_packed_float(1.0, 1.0, 0.0)
                    all_points.append((float(p[0]), float(p[1]), float(p[2]), rgbf))
            # Clustered (green)
            if 'points_clustered' in debug_pc:
                for p in debug_pc['points_clustered']:
                    rgbf = self._rgb_to_packed_float(0.0, 1.0, 0.0)
                    all_points.append((float(p[0]), float(p[1]), float(p[2]), rgbf))

            if 'preprocessing_stats' in debug_pc:
                s = debug_pc['preprocessing_stats']
                self.get_logger().info(
                    f"Object {i}: {s['original_points']} -> {s['sampled_points']} -> {s['clustered_points']} points",
                    throttle_duration_sec=2.0
                )

        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = stamp
        pc2_msg.header.frame_id = frame
        pc2_msg.height = 1
        pc2_msg.width = len(all_points)
        pc2_msg.is_bigendian = False
        pc2_msg.is_dense = True

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg.fields = fields
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width

        buf = bytearray(pc2_msg.row_step)
        off = 0
        for (x, y, z, rgbf) in all_points:
            struct.pack_into('<ffff', buf, off, x, y, z, rgbf)
            off += 16
        pc2_msg.data = bytes(buf)

        return pc2_msg

    # --------------------
    # main
    # --------------------
def main(args=None):
    rclpy.init(args=args)
    node = ObjectDepthTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
