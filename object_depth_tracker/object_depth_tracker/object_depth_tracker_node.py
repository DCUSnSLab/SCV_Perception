#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from cv_bridge import CvBridge
import message_filters
import time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from perception_interface.msg import DetectionArray
# Removed tracker filters - using Memory-SORT track IDs directly
import struct
import cv2

QUEUE_SIZE = 40
SLOP = 0.06  # 60 ms

class ObjectDepthTracker(Node):
    def __init__(self):
        super().__init__("object_depth_tracker_node")
        
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter("track_class_ids", [0, 2, 5, 7])  # person, car, bus, truck
        self.declare_parameter("detection_topic", "yolo/detections")
        self.declare_parameter("depth_topic", "/depth_anything/depth_registered/image_rect")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/left/camera_info")
        # Removed filter_type and tracking parameters - using Memory-SORT directly
        self.declare_parameter("min_mask_pixels", 30)
        
        # Get track_class_ids parameter
        track_class_ids_param = self.get_parameter("track_class_ids").value
        if isinstance(track_class_ids_param, list):
            self.track_ids = [int(x) for x in track_class_ids_param]
        elif isinstance(track_class_ids_param, str):
            try:
                self.track_ids = ast.literal_eval(track_class_ids_param)
                if not isinstance(self.track_ids, list):
                    self.track_ids = [int(track_class_ids_param)]
            except (ValueError, SyntaxError):
                self.get_logger().error(f"Invalid track_class_ids format: {track_class_ids_param}. Using default [0, 2, 5, 7]")
                self.track_ids = [0, 2, 5, 7]
        else:
            self.get_logger().error(f"Unexpected track_class_ids type: {type(track_class_ids_param)}. Using default [0, 2, 5, 7]")
            self.track_ids = [0, 2, 5, 7]
        detection_topic = self.get_parameter("detection_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        self.min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)
        
        # Create comprehensive class mapping for YOLO models
        self._init_class_mapping()

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # Synchronized subscribers
        self.sub_detections = message_filters.Subscriber(
            self, DetectionArray, detection_topic, qos_profile=sensor_qos)
        self.sub_depth = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=sensor_qos)
        self.sub_cinfo = message_filters.Subscriber(
            self, CameraInfo, camera_info_topic, qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_detections, self.sub_depth, self.sub_cinfo],
            queue_size=QUEUE_SIZE, slop=SLOP, allow_headerless=False)
        self.sync.registerCallback(self.sync_cb)

        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray, "tracked_points", QoSProfile(depth=2))
        self.bev_pub = self.create_publisher(
            MarkerArray, "tracked_points_bev", QoSProfile(depth=2))
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "debug_pointclouds", QoSProfile(depth=1))
        
        # No internal tracker needed - using Memory-SORT track IDs
        self.get_logger().info(f"3D converter ready (track ids={self.track_ids})")
        
        # Debug: store point clouds for visualization
        self.debug_pointclouds = []
        
    def _preprocess_depth_image(self, depth):
        """Basic depth preprocessing - remove invalid values only"""
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        depth[depth <= 0] = 0
        depth[depth > 20.0] = 0  # Remove far outliers
        return depth
        
    def _preprocess_segmentation_mask(self, mask):
        """Keep only largest connected component"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            return labels == largest_label
        return mask > 0
        
    def _sample_points(self, points_3d, max_points=300):
        """Simple random sampling"""
        if len(points_3d) <= max_points:
            return points_3d
        indices = np.random.choice(len(points_3d), size=max_points, replace=False)
        return points_3d[indices]
        
    def _estimate_3d_bbox_from_mask(self, mask, depth, fx, fy, cx, cy):
        """Estimate 3D bounding box from segmentation mask point cloud with improved preprocessing"""
        
        # Step 1: Preprocess segmentation mask
        mask_clean = self._preprocess_segmentation_mask(mask)
        
        # Step 2: Preprocess depth image
        depth_clean = self._preprocess_depth_image(depth)
        
        # Step 3: Get valid points
        vv, uu = np.nonzero(mask_clean)
        if vv.size < self.min_mask_pixels:
            return None
            
        valid_mask = (vv < depth_clean.shape[0]) & (uu < depth_clean.shape[1])
        if valid_mask.sum() < self.min_mask_pixels:
            return None
            
        vv_valid, uu_valid = vv[valid_mask], uu[valid_mask]
        z_vals = depth_clean[vv_valid, uu_valid]
        
        # More robust depth filtering
        good = (z_vals > 0.1) & (z_vals < 20.0)  # Tighter range for better quality
        
        if good.sum() < self.min_mask_pixels:
            return None
            
        # Get valid depth points
        uu_good, vv_good = uu_valid[good], vv_valid[good]
        z_good = z_vals[good]
        
        # Convert to 3D point cloud
        x_3d = (uu_good - cx) * z_good / fx
        y_3d = (vv_good - cy) * z_good / fy
        z_3d = z_good
        
        # Create point cloud array
        points_3d_original = np.column_stack([x_3d, y_3d, z_3d])
        
        # Step 4: Sample points for clustering  
        points_3d_sampled = self._sample_points(points_3d_original, max_points=300)
        
        # Step 5: DBSCAN clustering to get main object cluster
        points_3d_filtered = self._dbscan_clustering(points_3d_sampled)
            
        if len(points_3d_filtered) < self.min_mask_pixels:
            return None
        
        # Store point cloud for debugging
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
        
        # Calculate improved 3D bounding box
        bbox_result = self._compute_oriented_bbox(points_3d_filtered)
        
        # Add debug info
        if bbox_result:
            bbox_result['debug_pointcloud'] = debug_pc
            
        return bbox_result
        
    def _basic_outlier_removal(self, points_3d):
        """Basic outlier removal using distance from centroid"""
        if len(points_3d) < 10:
            return points_3d
            
        centroid = np.mean(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)
        q75 = np.percentile(distances, 75)
        threshold = q75 * 2.0  # Simple threshold
        return points_3d[distances <= threshold]
        
    def _dbscan_clustering(self, points_3d, eps=0.15, min_samples=10):
        """Use DBSCAN to find main object cluster and remove flying pixels"""
        if len(points_3d) < min_samples * 2:
            return points_3d
            
        try:
            from sklearn.cluster import DBSCAN
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(points_3d)
            
            # Find the largest cluster (exclude noise labeled as -1)
            unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
            
            if len(unique_labels) == 0:
                # No valid clusters found, fallback to basic outlier removal
                return self._basic_outlier_removal(points_3d)
            
            # Select the largest cluster
            largest_cluster_label = unique_labels[np.argmax(counts)]
            main_cluster_mask = cluster_labels == largest_cluster_label
            main_cluster_points = points_3d[main_cluster_mask]
            
            # Log cluster statistics
            noise_points = np.sum(cluster_labels == -1)
            total_clusters = len(unique_labels)
            
            self.get_logger().debug(
                f"DBSCAN: {total_clusters} clusters, {len(main_cluster_points)} main points, "
                f"{noise_points} noise points removed", 
                throttle_duration_sec=2.0
            )
            
            # If main cluster is too small, try with relaxed parameters
            if len(main_cluster_points) < self.min_mask_pixels:
                return self._dbscan_clustering(points_3d, eps=eps*1.5, min_samples=max(5, min_samples//2))
            
            return main_cluster_points
            
        except ImportError:
            # sklearn not available, fallback to basic outlier removal
            self.get_logger().warn("sklearn not available, using fallback outlier removal")
            return self._basic_outlier_removal(points_3d)
        except Exception as e:
            self.get_logger().warn(f"DBSCAN clustering failed: {e}, using fallback")
            return self._basic_outlier_removal(points_3d)
        
        
    def _compute_oriented_bbox(self, points_3d):
        """Compute bounding box with camera coordinate system (X=width, Y=height, Z=depth)"""
        try:
            # Calculate centroid
            centroid = np.mean(points_3d, axis=0)
            
            # Calculate axis-aligned bbox in camera coordinates
            min_coords = np.min(points_3d, axis=0)
            max_coords = np.max(points_3d, axis=0)
            
            # Size in each axis: X(width), Y(height), Z(depth)
            sizes = max_coords - min_coords
            
            # For robustness, also calculate using standard deviation
            std_devs = np.std(points_3d, axis=0)
            std_sizes = 4 * std_devs  # 4*std covers ~95% of points
            
            # Use combination of both methods
            alpha = 0.3
            final_sizes = (1 - alpha) * sizes + alpha * std_sizes
            
            # Ensure minimum size
            final_sizes = np.maximum(final_sizes, 0.1)
            
            # Debug log to check camera coordinate sizes
            self.get_logger().info(
                f"Camera coords - Width(X): {final_sizes[0]:.2f}, Height(Y): {final_sizes[1]:.2f}, Depth(Z): {final_sizes[2]:.2f}",
                throttle_duration_sec=2.0
            )
            
            return {
                'center': centroid,
                'size': final_sizes,
                'orientation': np.eye(3),  # Identity matrix for axis-aligned
                'point_count': len(points_3d),
                'bounds': [centroid[0] - final_sizes[0]/2, centroid[0] + final_sizes[0]/2,
                          centroid[1] - final_sizes[1]/2, centroid[1] + final_sizes[1]/2,
                          centroid[2] - final_sizes[2]/2, centroid[2] + final_sizes[2]/2]
            }
            
        except Exception as e:
            # Fallback to simple min-max if PCA fails
            x_min, x_max = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
            y_min, y_max = np.min(points_3d[:, 1]), np.max(points_3d[:, 1])
            z_min, z_max = np.min(points_3d[:, 2]), np.max(points_3d[:, 2])
            
            center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
            size = np.maximum(size, 0.1)  # Minimum size
            
            return {
                'center': center,
                'size': size,
                'point_count': len(points_3d),
                'bounds': [x_min, x_max, y_min, y_max, z_min, z_max]
            }
        
    def _estimate_3d_bbox_from_2d_bbox(self, bbox_2d, depth, fx, fy, cx, cy, class_name):
        """Estimate 3D bounding box from 2D bounding box and depth using class-specific priors"""
        try:
            x1, y1, x2, y2 = int(bbox_2d[0].x), int(bbox_2d[0].y), int(bbox_2d[2].x), int(bbox_2d[2].y)
            
            # Sample depth values in the 2D bbox
            u_samples = np.linspace(x1, x2, 10, dtype=int)
            v_samples = np.linspace(y1, y2, 10, dtype=int)
            
            depth_samples = []
            for u in u_samples:
                for v in v_samples:
                    if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                        d = depth[v, u]
                        if 0 < d < 50.0:
                            depth_samples.append(d)
            
            if len(depth_samples) < 10:
                return None
                
            # Use median depth for stability
            center_depth = np.median(depth_samples)
            
            # Convert 2D bbox to 3D using depth
            u_center = (x1 + x2) / 2
            v_center = (y1 + y2) / 2
            
            center_x = (u_center - cx) * center_depth / fx
            center_y = (v_center - cy) * center_depth / fy
            center_z = center_depth
            
            # Estimate 3D size based on class and distance
            bbox_width_2d = x2 - x1
            bbox_height_2d = y2 - y1
            
            # Class-specific size priors (rough estimates)
            size_priors = {
                'person': [0.6, 1.7, 0.3],
                'car': [1.8, 1.5, 4.5],
                'truck': [2.5, 3.0, 8.0],
                'bus': [2.8, 3.2, 12.0],
                'bicycle': [0.6, 1.2, 1.8],
                'motorcycle': [0.8, 1.3, 2.2]
            }
            
            default_size = [1.0, 1.0, 1.0]
            estimated_size = size_priors.get(class_name.lower(), default_size)
            
            # Scale based on apparent size and distance
            scale_factor = max(bbox_width_2d / 100.0, bbox_height_2d / 100.0) * (center_depth / 5.0)
            scaled_size = np.array(estimated_size) * max(0.5, min(2.0, scale_factor))
            
            return {
                'center': np.array([center_x, center_y, center_z]),
                'size': scaled_size,
                'bounds': [center_x - scaled_size[0]/2, center_x + scaled_size[0]/2,
                          center_y - scaled_size[1]/2, center_y + scaled_size[1]/2,
                          center_z - scaled_size[2]/2, center_z + scaled_size[2]/2]
            }
        except Exception as e:
            self.get_logger().debug(f"3D bbox estimation from 2D failed: {e}")
            return None
        
    def _init_class_mapping(self):
        """Initialize comprehensive class name to COCO ID mapping"""
        self.class_map = {
            # COCO dataset classes (80 classes)
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
        frame = depth_msg.header.frame_id or "zed_camera_link"

        tracked_objects = []

        # Process detections with Memory-SORT track IDs
        for det in detection_msg.detections:
            # Skip if no valid track ID from Memory-SORT
            if det.track_id < 0:
                continue
                
            # Check class filter
            class_name = det.class_name.lower()
            cls_id = self.class_map.get(class_name, -1)
            
            if cls_id == -1:
                self.get_logger().debug(f"Unknown class '{det.class_name}', skipping", throttle_duration_sec=5.0)
                continue
                
            if cls_id not in self.track_ids:
                continue

            bbox_3d_info = None
            
            # Get mask if available
            if det.mask.width > 0 and det.mask.height > 0:
                try:
                    mask = self.bridge.imgmsg_to_cv2(det.mask, "mono8") > 0
                    
                    # Calculate 3D bounding box from mask
                    bbox_3d_info = self._estimate_3d_bbox_from_mask(mask, depth, fx, fy, cx, cy)
                    
                    if bbox_3d_info is None:
                        continue
                        
                    # Use center as tracking point
                    center_3d = bbox_3d_info['center']
                    X, Y, Z = center_3d[0], center_3d[1], center_3d[2]
                    
                except Exception as e:
                    self.get_logger().warn(f"Mask processing failed: {e}")
                    continue
            else:
                # Fall back to bounding box centroid
                try:
                    if len(det.bounding_box.points) >= 4:
                        bbox = det.bounding_box.points
                        
                        # Calculate 3D bounding box from 2D bbox
                        bbox_3d_info = self._estimate_3d_bbox_from_2d_bbox(bbox, depth, fx, fy, cx, cy, det.class_name)
                        
                        if bbox_3d_info is None:
                            # Fallback to simple centroid
                            u_bar = int((bbox[0].x + bbox[2].x) / 2)
                            v_bar = int((bbox[0].y + bbox[2].y) / 2)
                            
                            if (0 <= u_bar < depth.shape[1] and 0 <= v_bar < depth.shape[0]):
                                Z = float(depth[v_bar, u_bar])
                                if Z <= 0 or Z > 50.0:  # Filter unrealistic depths
                                    continue
                                    
                                # Convert to 3D coordinates
                                X = (u_bar - cx) * Z / fx
                                Y = (v_bar - cy) * Z / fy
                            else:
                                continue
                        else:
                            center_3d = bbox_3d_info['center']
                            X, Y, Z = center_3d[0], center_3d[1], center_3d[2]
                    else:
                        continue
                except (IndexError, ValueError) as e:
                    self.get_logger().warn(f"Bounding box processing failed: {e}")
                    continue

            # Sanity check for 3D coordinates
            if abs(X) > 100 or abs(Y) > 100 or abs(Z) > 100:
                self.get_logger().debug(f"Unrealistic 3D coordinates: ({X:.2f}, {Y:.2f}, {Z:.2f})", throttle_duration_sec=5.0)
                continue
                
            # Create tracked object with Memory-SORT track ID
            tracked_obj = {
                'id': det.track_id,  # Use Memory-SORT track ID directly
                'xyz': np.array([X, Y, Z]),
                'class_name': det.class_name,
                'cls_id': cls_id,
                'bbox_3d': bbox_3d_info['size'] if bbox_3d_info else None,
                'confidence': det.confidence
            }
            tracked_objects.append(tracked_obj)
            
            # Store debug pointcloud info
            if bbox_3d_info and 'debug_pointcloud' in bbox_3d_info:
                debug_pc = bbox_3d_info['debug_pointcloud']
                debug_pc['class_name'] = det.class_name
                debug_pc['track_id'] = det.track_id
                self.debug_pointclouds.append(debug_pc)
        
        # Debug: Log track IDs
        track_ids = [obj['id'] for obj in tracked_objects]
        self.get_logger().info(f"Publishing tracks with IDs: {track_ids}", throttle_duration_sec=2.0)
        
        # Publish markers
        marker_arr = self.to_markers_simple(tracked_objects, stamp, frame)
        self.marker_pub.publish(marker_arr)
        
        # Publish BEV markers
        bev_arr = self.create_bev_markers_simple(tracked_objects, stamp, frame)
        self.bev_pub.publish(bev_arr)
        
        # Publish debug point clouds
        if self.debug_pointclouds:
            debug_pc_msg = self._create_debug_pointcloud_msg(stamp, frame)
            self.pointcloud_pub.publish(debug_pc_msg)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"[3d_converter] Frame: {elapsed_ms:.1f}ms, Detections: {len(detection_msg.detections)}, "
            f"Valid 3D objects: {len(tracked_objects)}", 
            throttle_duration_sec=1.0
        )

    def to_markers_simple(self, tracked_objects, stamp, frame):
        arr = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # Add current tracked objects
        for obj in tracked_objects:
            # Center sphere marker
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "tracked_pts"
            m.id = obj['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(obj['xyz'][0])
            m.pose.position.y = float(obj['xyz'][1])
            m.pose.position.z = float(obj['xyz'][2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 0.8
            arr.markers.append(m)

            # 3D Bounding Box
            if obj['bbox_3d'] is not None:
                bbox = Marker()
                bbox.header.stamp = stamp
                bbox.header.frame_id = frame
                bbox.ns = "bbox_3d"
                bbox.id = obj['id']
                bbox.type = Marker.CUBE
                bbox.action = Marker.ADD
                bbox.pose.position.x = float(obj['xyz'][0])
                bbox.pose.position.y = float(obj['xyz'][1])
                bbox.pose.position.z = float(obj['xyz'][2])
                bbox.pose.orientation.w = 1.0
                
                # Set size
                bbox.scale.x = max(0.1, float(obj['bbox_3d'][0]))
                bbox.scale.y = max(0.1, float(obj['bbox_3d'][1]))
                bbox.scale.z = max(0.1, float(obj['bbox_3d'][2]))
                
                bbox.color.r, bbox.color.g, bbox.color.b, bbox.color.a = 1.0, 0.0, 0.0, 0.3
                arr.markers.append(bbox)

            # Text label
            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = frame
            text.ns = "labels"
            text.id = obj['id']
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(obj['xyz'][0])
            text.pose.position.y = float(obj['xyz'][1])
            text.pose.position.z = float(obj['xyz'][2]) + 0.3  # Above object
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
            text.text = f"ID:{obj['id']} {obj['class_name']}"
            arr.markers.append(text)
            
        return arr

    def create_bev_markers_simple(self, tracked_objects, stamp, frame):
        arr = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # Add current tracked objects for BEV (Bird's Eye View)
        for obj in tracked_objects:
            # BEV center marker (Z=0 for top-down view)
            bev = Marker()
            bev.header.stamp = stamp
            bev.header.frame_id = frame
            bev.ns = "bev_tracked_pts"
            bev.id = obj['id']
            bev.type = Marker.CYLINDER
            bev.action = Marker.ADD
            bev.pose.position.x = float(obj['xyz'][0])
            bev.pose.position.y = float(obj['xyz'][1])
            bev.pose.position.z = 0.0  # BEV: Z=0
            bev.pose.orientation.w = 1.0
            bev.scale.x = bev.scale.y = 0.3
            bev.scale.z = 0.1
            bev.color.r, bev.color.g, bev.color.b, bev.color.a = 0.0, 1.0, 0.0, 0.8
            arr.markers.append(bev)

            # BEV text label
            bev_text = Marker()
            bev_text.header.stamp = stamp
            bev_text.header.frame_id = frame
            bev_text.ns = "bev_labels"
            bev_text.id = obj['id']
            bev_text.type = Marker.TEXT_VIEW_FACING
            bev_text.action = Marker.ADD
            bev_text.pose.position.x = float(obj['xyz'][0])
            bev_text.pose.position.y = float(obj['xyz'][1])
            bev_text.pose.position.z = 0.2  # Slightly above BEV marker
            bev_text.pose.orientation.w = 1.0
            bev_text.scale.z = 0.15
            bev_text.color.r, bev_text.color.g, bev_text.color.b, bev_text.color.a = 1.0, 1.0, 1.0, 1.0
            bev_text.text = f"ID:{obj['id']}"
            arr.markers.append(bev_text)
            
        return arr

    def to_markers(self, tracks, stamp, frame):
        arr = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # Add current tracks
        for trk in tracks:
            # Center sphere marker
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "tracked_pts"
            m.id = trk.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(trk.xyz[0])
            m.pose.position.y = float(trk.xyz[1])
            m.pose.position.z = float(trk.xyz[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 0.8
            arr.markers.append(m)

            # 3D Bounding Box
            if hasattr(trk, 'bbox_3d') and trk.bbox_3d is not None:
                bbox = Marker()
                bbox.header.stamp = stamp
                bbox.header.frame_id = frame
                bbox.ns = "bbox_3d"
                bbox.id = trk.id
                bbox.type = Marker.CUBE
                bbox.action = Marker.ADD
                bbox.pose.position.x = float(trk.xyz[0])
                bbox.pose.position.y = float(trk.xyz[1])
                bbox.pose.position.z = float(trk.xyz[2])
                bbox.pose.orientation.w = 1.0
                
                # Set size
                if isinstance(trk.bbox_3d, np.ndarray):
                    bbox.scale.x = max(0.1, float(trk.bbox_3d[0]))
                    bbox.scale.y = max(0.1, float(trk.bbox_3d[1]))
                    bbox.scale.z = max(0.1, float(trk.bbox_3d[2]))
                else:
                    bbox.scale.x = max(0.1, float(trk.bbox_3d[0]))
                    bbox.scale.y = max(0.1, float(trk.bbox_3d[1]))
                    bbox.scale.z = max(0.1, float(trk.bbox_3d[2]))
                
                # Color by class
                class_colors = {
                    'person': [1.0, 0.0, 0.0, 0.3],    # Red
                    'car': [0.0, 1.0, 0.0, 0.3],       # Green
                    'truck': [0.0, 0.0, 1.0, 0.3],     # Blue
                    'bus': [1.0, 1.0, 0.0, 0.3],       # Yellow
                    'bicycle': [1.0, 0.0, 1.0, 0.3],   # Magenta
                    'motorcycle': [0.0, 1.0, 1.0, 0.3] # Cyan
                }
                
                class_name = getattr(trk, 'class_name', 'unknown').lower()
                color = class_colors.get(class_name, [0.5, 0.5, 0.5, 0.3])
                bbox.color.r, bbox.color.g, bbox.color.b, bbox.color.a = color
                arr.markers.append(bbox)

            # ID and class text marker
            txt = Marker()
            txt.header.stamp = stamp
            txt.header.frame_id = frame
            txt.ns = "track_info"
            txt.id = trk.id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            x, y, z = trk.xyz
            txt.pose.position.x = float(x)
            txt.pose.position.y = float(y)
            txt.pose.position.z = float(z + 0.5)
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.3
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            
            class_name = getattr(trk, 'class_name', 'unknown')
            
            # Add bbox size and point count information if available
            bbox_info = ""
            if hasattr(trk, 'bbox_3d') and trk.bbox_3d is not None:
                if isinstance(trk.bbox_3d, np.ndarray):
                    size = trk.bbox_3d
                else:
                    size = np.array(trk.bbox_3d)
                bbox_info = f"\nSize: {size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f}m"
                
                # Add point count if track has this info
                if hasattr(trk, 'point_count'):
                    bbox_info += f"\nPoints: {trk.point_count}"
            
            txt.text = f"ID:{trk.id}\n{class_name}{bbox_info}"
            arr.markers.append(txt)

        return arr
        
    def create_bev_markers(self, tracks, stamp, frame):
        """Create BEV markers - same as 3D but with Z=0"""
        arr = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # Add current tracks
        for trk in tracks:
            # Center point marker (Z=0 for BEV)
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = "bev_center"
            m.id = trk.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(trk.xyz[0])
            m.pose.position.y = float(trk.xyz[1])
            m.pose.position.z = 0.0  # BEV: Z=0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 0.8
            arr.markers.append(m)

            # 3D Bounding Box (but positioned at Z=0 for BEV)
            if hasattr(trk, 'bbox_3d') and trk.bbox_3d is not None:
                bbox = Marker()
                bbox.header.stamp = stamp
                bbox.header.frame_id = frame
                bbox.ns = "bev_bbox"
                bbox.id = trk.id
                bbox.type = Marker.CUBE
                bbox.action = Marker.ADD
                bbox.pose.position.x = float(trk.xyz[0])
                bbox.pose.position.y = float(trk.xyz[1])
                bbox.pose.position.z = 0.0  # BEV: Z=0
                bbox.pose.orientation.w = 1.0
                
                # Set size (same as 3D)
                if isinstance(trk.bbox_3d, np.ndarray):
                    bbox.scale.x = max(0.1, float(trk.bbox_3d[0]))
                    bbox.scale.y = max(0.1, float(trk.bbox_3d[1]))
                    bbox.scale.z = max(0.1, float(trk.bbox_3d[2]))
                else:
                    bbox.scale.x = max(0.1, float(trk.bbox_3d[0]))
                    bbox.scale.y = max(0.1, float(trk.bbox_3d[1]))
                    bbox.scale.z = max(0.1, float(trk.bbox_3d[2]))
                
                # Color based on class
                class_colors = {
                    'person': [1.0, 0.0, 0.0, 0.3],
                    'car': [0.0, 1.0, 0.0, 0.3],
                    'truck': [0.0, 0.0, 1.0, 0.3],
                    'bus': [1.0, 1.0, 0.0, 0.3]
                }
                class_name = getattr(trk, 'class_name', 'unknown').lower()
                color = class_colors.get(class_name, [0.5, 0.5, 0.5, 0.3])
                bbox.color.r, bbox.color.g, bbox.color.b, bbox.color.a = color
                arr.markers.append(bbox)

            # Text label (above ground level for BEV)
            txt = Marker()
            txt.header.stamp = stamp
            txt.header.frame_id = frame
            txt.ns = "bev_text"
            txt.id = trk.id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(trk.xyz[0])
            txt.pose.position.y = float(trk.xyz[1])
            txt.pose.position.z = 0.5  # BEV: above ground
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.3
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            
            class_name = getattr(trk, 'class_name', 'unknown')
            
            # Add bbox size information
            bbox_info = ""
            if hasattr(trk, 'bbox_3d') and trk.bbox_3d is not None:
                if isinstance(trk.bbox_3d, np.ndarray):
                    size = trk.bbox_3d
                else:
                    size = np.array(trk.bbox_3d)
                bbox_info = f"\nSize: {size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f}m"
                
                if hasattr(trk, 'point_count'):
                    bbox_info += f"\nPoints: {trk.point_count}"
            
            txt.text = f"ID:{trk.id}\n{class_name}{bbox_info}"
            arr.markers.append(txt)

        return arr
        
    def _create_debug_pointcloud_msg(self, stamp, frame):
        """Create PointCloud2 message from debug point clouds with processing stage visualization"""
        all_points = []
        
        for i, debug_pc in enumerate(self.debug_pointclouds):
            # Show different processing stages with different colors
            
            # Original points (red, faded)
            if 'points_original' in debug_pc:
                points_orig = debug_pc['points_original']
                for point in points_orig[::5]:  # Sample every 5th point to reduce clutter
                    all_points.append([
                        float(point[0]), float(point[1]), float(point[2]),
                        1.0, 0.3, 0.3  # Light red
                    ])
            
            # Sampled points (yellow)
            if 'points_sampled' in debug_pc:
                points_sampled = debug_pc['points_sampled']
                for point in points_sampled[::2]:  # Sample every 2nd point
                    all_points.append([
                        float(point[0]), float(point[1]), float(point[2]),
                        1.0, 1.0, 0.0  # Yellow
                    ])
            
            # Final clustered points (bright green)
            if 'points_clustered' in debug_pc:
                points_clustered = debug_pc['points_clustered']
                for point in points_clustered:  # Show all final points
                    all_points.append([
                        float(point[0]), float(point[1]), float(point[2]),
                        0.0, 1.0, 0.0  # Bright green
                    ])
            
            # Log preprocessing stats
            if 'preprocessing_stats' in debug_pc:
                stats = debug_pc['preprocessing_stats']
                self.get_logger().info(
                    f"Object {i}: {stats['original_points']} -> "
                    f"{stats['sampled_points']} -> {stats['clustered_points']} points",
                    throttle_duration_sec=2.0
                )
        
        if not all_points:
            return self._create_empty_pointcloud(stamp, frame)
        
        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = stamp
        pc2_msg.header.frame_id = frame
        
        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg.fields = fields
        
        # Pack data
        pc2_msg.width = len(all_points)
        pc2_msg.height = 1
        pc2_msg.is_dense = True
        pc2_msg.point_step = 24
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        
        buffer = []
        for point in all_points:
            buffer.extend(struct.pack('<ffffff', *point))
        
        pc2_msg.data = bytes(buffer)
        
        return pc2_msg
        
    def _create_empty_pointcloud(self, stamp, frame):
        """Create empty PointCloud2 message"""
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = stamp
        pc2_msg.header.frame_id = frame
        pc2_msg.width = 0
        pc2_msg.height = 1
        pc2_msg.is_dense = True
        pc2_msg.point_step = 24
        pc2_msg.row_step = 0
        pc2_msg.data = b''
        return pc2_msg

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