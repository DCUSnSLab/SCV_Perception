"""ROS 2 node extending Jay Tracker with observation-adaptive policies."""

from collections import deque

import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from perception_interface.msg import DetectionArray
from pcdet_tracker.jay_tracker import (
    JayTrackerNode, message_stamp_seconds, parse_config)

from .camera_observation import CameraObservationConfig, CameraObservationAnalyzer
from .cascaded_association import AssociationConfig, CascadedTracker
from .csv_logging import ObservationCsvLogger
from .depth_consistency import DepthConsistencyAnalyzer, DepthConsistencyConfig
from .lidar_observation import LidarObservationAnalyzer, LidarObservationConfig
from .observation_analyzer import ObservationAnalyzer
from .track_lifecycle import LifecycleConfig
from .visualization import ObservationVisualizer


class ObservationTrackerNode(JayTrackerNode):
    def __init__(self, args):
        # Empty adaptive_config keeps the original package untouched. This node
        # installs its own analyzer and cascade immediately after construction.
        args.adaptive_config = ""
        super().__init__(args)
        self._declare_observation_parameters()
        self._configure_observation_tracker()

    def _declare(self, name, value):
        self.declare_parameter(name, value)
        return self.get_parameter(name).value

    def _declare_observation_parameters(self):
        self.parameters = {
            "detector_3d_threshold": self._declare("detector_3d_threshold", 0.65),
            "detector_2d_threshold": self._declare("detector_2d_threshold", 0.25),
            "observation_box_margin_xy": self._declare("observation_box_margin_xy", 0.15),
            "observation_box_margin_z": self._declare("observation_box_margin_z", 0.10),
            "near_range": self._declare("near_range", 15.0),
            "middle_range": self._declare("middle_range", 30.0),
            "far_range": self._declare("far_range", 50.0),
            "lidar_weak_min_points": self._declare("lidar_weak_min_points", 2),
            "centroid_distance_threshold": self._declare("centroid_distance_threshold", 1.25),
            "camera_iou_threshold": self._declare("camera_iou_threshold", 0.20),
            "camera_center_distance_threshold": self._declare(
                "camera_center_distance_threshold", 100.0),
            "depth_consistency_threshold": self._declare("depth_consistency_threshold", 1.5),
            "occlusion_depth_threshold": self._declare("occlusion_depth_threshold", 1.0),
            "min_valid_depth_pixels": self._declare("min_valid_depth_pixels", 20),
            "depth_roi_inner_scale": self._declare("depth_roi_inner_scale", 0.60),
            "min_depth_support_ratio": self._declare(
                "min_depth_support_ratio", 0.15),
            "center_distance_weight": self._declare("center_distance_weight", 1.25),
            "iou_3d_weight": self._declare("iou_3d_weight", 1.25),
            "motion_weight": self._declare("motion_weight", 0.75),
            "camera_iou_weight": self._declare("camera_iou_weight", 0.5),
            "depth_weight": self._declare("depth_weight", 0.25),
            "class_penalty": self._declare("class_penalty", 10.0),
            "center_distance_gate": self._declare("center_distance_gate", 2.5),
            "iou_3d_gate": self._declare("iou_3d_gate", 0.05),
            "motion_gate": self._declare("motion_gate", 4.0),
            "maximum_association_cost": self._declare("maximum_association_cost", 3.5),
            "camera_recovery_position_gain": self._declare(
                "camera_recovery_position_gain", 0.40),
            "camera_recovery_max_correction": self._declare(
                "camera_recovery_max_correction", 0.50),
            "enable_weak_track_continuity": self._declare(
                "enable_weak_track_continuity", True),
            "weak_center_distance_gate": self._declare(
                "weak_center_distance_gate", 1.0),
            "weak_iou_3d_gate": self._declare("weak_iou_3d_gate", 0.20),
            "weak_continuity_max_age": self._declare(
                "weak_continuity_max_age", 2),
            "tentative_min_hits": self._declare("tentative_min_hits", 3),
            "strong_confirmation_hits": self._declare("strong_confirmation_hits", 2),
            "max_miss_count": self._declare("max_miss_count", 6),
            "recovery_max_age": self._declare("recovery_max_age", 3),
            "weak_observation_max_frames": self._declare("weak_observation_max_frames", 4),
            "history_size": self._declare("history_size", 7),
            "confidence_decay": self._declare("confidence_decay", 0.15),
            "min_track_confidence": self._declare("min_track_confidence", 0.10),
            "smoothing_switch_count": self._declare("smoothing_switch_count", 2),
            "enable_lidar_observation": self._declare("enable_lidar_observation", True),
            "enable_camera_observation": self._declare("enable_camera_observation", True),
            "enable_depth_consistency": self._declare("enable_depth_consistency", True),
            "enable_observation_state": self._declare("enable_observation_state", True),
            "enable_cascaded_association": self._declare("enable_cascaded_association", True),
            "enable_adaptive_lifecycle": self._declare("enable_adaptive_lifecycle", True),
            "enable_temporal_smoothing": self._declare("enable_temporal_smoothing", True),
            "enable_odometry": self._declare("enable_odometry", True),
            "enable_imu": self._declare("enable_imu", True),
            "enable_csv_logging": self._declare("enable_csv_logging", True),
            "enable_fused_detection_visualization": self._declare(
                "enable_fused_detection_visualization", True),
            "lidar_points_topic": self._declare("lidar_points_topic", "/velodyne_points"),
            "camera_rgb_topic": self._declare(
                "camera_rgb_topic", "/camera/camera/color/image_raw"),
            "camera_detection_topic": self._declare("camera_detection_topic", "/yolo/detections"),
            "camera_info_topic": self._declare(
                "camera_info_topic", "/camera/camera/color/camera_info"),
            "aligned_depth_topic": self._declare(
                "aligned_depth_topic",
                "/camera/camera/aligned_depth_to_color/image_raw"),
            "camera_optical_frame": self._declare(
                "camera_optical_frame", "camera_color_optical_frame"),
            "lidar_tolerance_sec": self._declare("lidar_tolerance_sec", 0.12),
            "camera_tolerance_sec": self._declare("camera_tolerance_sec", 0.15),
            "depth_tolerance_sec": self._declare("depth_tolerance_sec", 0.15),
            "sensor_cache_size": self._declare("sensor_cache_size", 8),
            "csv_output_dir": self._declare("csv_output_dir", "results/observation_tracker"),
            "debug_marker_topic": self._declare(
                "debug_marker_topic", "/observation_tracker/debug_markers"),
            "fused_detection_marker_topic": self._declare(
                "fused_detection_marker_topic",
                "/observation_tracker/fused_detection_markers"),
            "marker_alpha": self._declare("observation_marker_alpha", 0.55),
        }
        for object_class, defaults in {
                "vehicle": (12, 8, 4),
                "pedestrian": (6, 4, 2),
                "cyclist": (8, 5, 3)}.items():
            for band, default in zip(("near", "middle", "far"), defaults):
                key = f"lidar.{object_class}.{band}.min_points"
                self.parameters[key] = self._declare(key, default)
        for state, default in {
                "multimodal_strong": [0.10, 0.95, 0.25],
                "lidar_dominant": [0.10, 0.55, 1.00],
                "camera_dominant": [0.95, 0.75, 0.10],
                "weak_observation": [0.95, 0.15, 0.15]}.items():
            key = f"visualization.{state}_color"
            self.parameters[key] = self._declare(key, default)

    def _configure_observation_tracker(self):
        p = self.parameters
        self.args.score_thresh = float(p["detector_3d_threshold"])
        thresholds = {
            name: {
                band: int(p[f"lidar.{name}.{band}.min_points"])
                for band in ("near", "middle", "far")}
            for name in ("vehicle", "pedestrian", "cyclist")}
        thresholds["unknown"] = dict(thresholds["vehicle"])
        lidar = LidarObservationAnalyzer(LidarObservationConfig(
            near_range=float(p["near_range"]),
            middle_range=float(p["middle_range"]),
            far_range=float(p["far_range"]),
            weak_min_points=int(p["lidar_weak_min_points"]),
            centroid_distance_threshold=float(p["centroid_distance_threshold"]),
            margin_xy=float(p["observation_box_margin_xy"]),
            margin_z=float(p["observation_box_margin_z"]),
            thresholds=thresholds))
        camera = CameraObservationAnalyzer(CameraObservationConfig(
            iou_threshold=float(p["camera_iou_threshold"]),
            center_distance_threshold=float(p["camera_center_distance_threshold"]),
            detector_threshold=float(p["detector_2d_threshold"])))
        depth = DepthConsistencyAnalyzer(DepthConsistencyConfig(
            consistency_threshold=float(p["depth_consistency_threshold"]),
            occlusion_depth_threshold=float(p["occlusion_depth_threshold"]),
            min_valid_pixels=int(p["min_valid_depth_pixels"]),
            roi_inner_scale=float(p["depth_roi_inner_scale"]),
            min_support_ratio=float(p["min_depth_support_ratio"])))
        self.observation_analyzer = ObservationAnalyzer(
            lidar, camera, depth,
            enable_lidar=bool(p["enable_lidar_observation"]),
            enable_camera=bool(p["enable_camera_observation"]),
            enable_depth=bool(p["enable_depth_consistency"]),
            enable_state=bool(p["enable_observation_state"]))
        association = AssociationConfig(
            center_distance_weight=float(p["center_distance_weight"]),
            iou_3d_weight=float(p["iou_3d_weight"]),
            motion_weight=float(p["motion_weight"]),
            camera_iou_weight=float(p["camera_iou_weight"]),
            depth_weight=float(p["depth_weight"]),
            class_penalty=float(p["class_penalty"]),
            center_distance_gate=float(p["center_distance_gate"]),
            iou_3d_gate=float(p["iou_3d_gate"]),
            motion_gate=float(p["motion_gate"]),
            maximum_cost=float(p["maximum_association_cost"]),
            camera_recovery_position_gain=float(
                p["camera_recovery_position_gain"]),
            camera_recovery_max_correction=float(
                p["camera_recovery_max_correction"]),
            enable_weak_track_continuity=bool(
                p["enable_weak_track_continuity"]),
            weak_center_distance_gate=float(p["weak_center_distance_gate"]),
            weak_iou_3d_gate=float(p["weak_iou_3d_gate"]),
            weak_continuity_max_age=int(p["weak_continuity_max_age"]),
            enable_cascaded_association=bool(p["enable_cascaded_association"]))
        lifecycle = LifecycleConfig(
            tentative_min_hits=int(p["tentative_min_hits"]),
            strong_confirmation_hits=int(p["strong_confirmation_hits"]),
            max_miss_count=int(p["max_miss_count"]),
            recovery_max_age=int(p["recovery_max_age"]),
            weak_observation_max_frames=int(p["weak_observation_max_frames"]),
            history_size=int(p["history_size"]),
            confidence_decay=float(p["confidence_decay"]),
            min_track_confidence=float(p["min_track_confidence"]),
            enable_adaptive_lifecycle=bool(p["enable_adaptive_lifecycle"]),
            enable_temporal_smoothing=bool(p["enable_temporal_smoothing"]),
            smoothing_switch_count=int(p["smoothing_switch_count"]))
        self.tracker = CascadedTracker(
            expected_dt=self.args.lidar_frame_dt,
            max_age_frames=self.args.track_max_age_frames,
            confirm_frames=self.args.track_confirm_frames,
            match_distance=self.args.track_match_distance,
            min_spawn_score=self.args.track_min_spawn_score,
            static_speed_thresh=self.args.track_static_speed_thresh,
            static_yaw_alpha=self.args.track_static_yaw_alpha,
            dynamic_yaw_alpha=self.args.track_dynamic_yaw_alpha,
            ego_motion_match_gain=self.args.ego_motion_match_gain,
            ego_motion_assoc_bonus=self.args.ego_motion_assoc_bonus,
            observation_history_size=int(p["history_size"]),
            association_config=association,
            lifecycle_config=lifecycle)

        self.enable_observation_analysis = True
        self.enable_lidar_support = bool(p["enable_lidar_observation"])
        self.enable_camera_support = bool(p["enable_camera_observation"])
        self.enable_depth_consistency = (
            self.enable_camera_support and bool(p["enable_depth_consistency"]))
        if not bool(p["enable_odometry"]):
            self.destroy_subscription(self.odom_sub)
            self.odom_sub = None
        if not bool(p["enable_imu"]):
            self.destroy_subscription(self.imu_sub)
            self.imu_sub = None
        size = max(2, int(p["sensor_cache_size"]))
        self.lidar_cache = deque(maxlen=size)
        self.camera_detection_cache = deque(maxlen=size)
        self.camera_info_cache = deque(maxlen=size)
        self.depth_cache = deque(maxlen=size)
        self.rgb_cache = deque(maxlen=size)
        self.lidar_tolerance = float(p["lidar_tolerance_sec"])
        self.camera_tolerance = float(p["camera_tolerance_sec"])
        self.depth_tolerance = float(p["depth_tolerance_sec"])
        self.camera_optical_frame = str(p["camera_optical_frame"])
        self.static_tf_only = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_transform_cache = {}
        self.cv_bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.observation_subscriptions = []
        if self.enable_lidar_support:
            self.observation_subscriptions.append(self.create_subscription(
                PointCloud2, str(p["lidar_points_topic"]),
                self.lidar_observation_callback, qos))
        if self.enable_camera_support:
            self.observation_subscriptions.extend([
                self.create_subscription(
                    Image, str(p["camera_rgb_topic"]), self.rgb_callback,
                    camera_qos),
                self.create_subscription(
                    DetectionArray, str(p["camera_detection_topic"]),
                    self.camera_detection_callback, camera_qos),
                self.create_subscription(
                    CameraInfo, str(p["camera_info_topic"]),
                    self.camera_info_callback, camera_qos),
            ])
        if self.enable_depth_consistency:
            self.observation_subscriptions.append(self.create_subscription(
                Image, str(p["aligned_depth_topic"]), self.depth_callback,
                camera_qos))

        colors = {
            "MULTIMODAL_STRONG": tuple(p["visualization.multimodal_strong_color"]),
            "LIDAR_DOMINANT": tuple(p["visualization.lidar_dominant_color"]),
            "CAMERA_DOMINANT": tuple(p["visualization.camera_dominant_color"]),
            "WEAK_OBSERVATION": tuple(p["visualization.weak_observation_color"]),
        }
        self.observation_visualizer = ObservationVisualizer(
            colors=colors, alpha=float(p["marker_alpha"]),
            lifetime=self.args.marker_lifetime)
        self.observation_marker_pub = self.create_publisher(
            MarkerArray, str(p["debug_marker_topic"]), 10)
        self.fused_detection_marker_pub = None
        if bool(p["enable_fused_detection_visualization"]):
            self.fused_detection_marker_pub = self.create_publisher(
                MarkerArray, str(p["fused_detection_marker_topic"]), 10)
        if bool(p["enable_csv_logging"]):
            self.observation_writer = ObservationCsvLogger(
                str(p["csv_output_dir"]), self.args.sequence_id,
                track_provider=lambda: self.tracker.last_debug_tracks)
        self.get_logger().info(
            "observation_tracker ready: lidar=%s camera=%s depth=%s "
            "odom=%s imu=%s cascaded=%s lifecycle=%s smoothing=%s" % (
                self.enable_lidar_support,
                self.enable_camera_support,
                self.enable_depth_consistency,
                p["enable_odometry"],
                p["enable_imu"],
                p["enable_cascaded_association"],
                p["enable_adaptive_lifecycle"],
                p["enable_temporal_smoothing"]))

    def rgb_callback(self, message):
        self.rgb_cache.append((
            message_stamp_seconds(message),
            (int(message.width), int(message.height))))

    def build_observation_provider(self, header, timestamp):
        # An RGB frame means the camera was available even when the 2-D detector
        # published no candidate. Feed an empty detection snapshot only when no
        # DetectionArray exists within the configured tolerance.
        real_cache = self.camera_detection_cache
        if (self.enable_camera_support and
                self._nearest(real_cache, timestamp, self.camera_tolerance) is None):
            rgb = self._nearest(self.rgb_cache, timestamp, self.camera_tolerance)
            if rgb is not None:
                merged = deque(real_cache, maxlen=real_cache.maxlen)
                merged.append((rgb[0], [], rgb[1]))
                self.camera_detection_cache = merged
                try:
                    return super().build_observation_provider(header, timestamp)
                finally:
                    self.camera_detection_cache = real_cache
        return super().build_observation_provider(header, timestamp)

    def publish_markers(self, detections, tracked_objects, frame_id, stamp):
        super().publish_markers(
            detections, tracked_objects, frame_id, stamp)
        from std_msgs.msg import Header
        marker_header = Header()
        marker_header.frame_id = frame_id
        marker_header.stamp = stamp
        debug_tracks = self.tracker.last_debug_tracks
        self.observation_marker_pub.publish(
            self.observation_visualizer.build(debug_tracks, marker_header))
        if self.fused_detection_marker_pub is not None:
            camera_depth_candidates = self._camera_depth_candidates(
                frame_id, stamp, debug_tracks)
            self.fused_detection_marker_pub.publish(
                self.observation_visualizer.build_fused_detections(
                    detections,
                    self.tracker.last_detection_observations,
                    self.class_names,
                    marker_header,
                    camera_depth_candidates,
                    debug_tracks))

    def _camera_depth_candidates(self, frame_id, stamp, debug_tracks):
        """Back-project 2-D detections to depth points for RViz diagnostics."""
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        camera_item = self._nearest(
            self.camera_detection_cache, timestamp, self.camera_tolerance)
        depth_item = self._nearest(
            self.depth_cache, timestamp, self.depth_tolerance)
        info_item = self._nearest(self.camera_info_cache, timestamp, None)
        if camera_item is None or depth_item is None or info_item is None:
            return []
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(
                depth_item[1], desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(
                f"camera-depth candidate conversion failed: {exc}",
                throttle_duration_sec=2.0)
            return []

        fx, fy, cx, cy = info_item[1]
        camera_frame = info_item[3] or self.camera_optical_frame
        camera_from_source = self._lookup_matrix(camera_frame, frame_id, stamp)
        if camera_from_source is None or fx <= 0.0 or fy <= 0.0:
            return []
        source_from_camera = np.linalg.inv(camera_from_source)
        cfg = self.observation_analyzer.depth.config
        matched_tracks = {}
        for track in debug_tracks:
            observation = track.get("observation")
            if observation is not None and observation.camera_detection_index >= 0:
                matched_tracks[observation.camera_detection_index] = track

        candidates = []
        image_height, image_width = depth.shape[:2]
        for index, detection in enumerate(camera_item[1]):
            if float(detection.score) < self.parameters["detector_2d_threshold"]:
                continue
            x1, y1, x2, y2 = map(float, detection.bbox)
            scale = float(np.clip(cfg.roi_inner_scale, 0.05, 1.0))
            center_u, center_v = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            half_width = (x2 - x1) * scale * 0.5
            half_height = (y2 - y1) * scale * 0.5
            left = max(0, min(image_width - 1, int(center_u - half_width)))
            right = max(0, min(image_width, int(center_u + half_width)))
            top = max(0, min(image_height - 1, int(center_v - half_height)))
            bottom = max(0, min(image_height, int(center_v + half_height)))
            if right <= left or bottom <= top:
                continue
            values = np.asarray(depth[top:bottom, left:right]).reshape(-1)
            values = values.astype(np.float32)
            if depth.dtype == np.uint16:
                values *= cfg.depth_scale
            valid = values[
                np.isfinite(values) &
                (values >= cfg.min_depth_m) &
                (values <= cfg.max_depth_m)]
            if len(valid) < cfg.min_valid_pixels:
                continue
            depth_m = float(np.median(valid))
            point_camera = np.array([[
                (center_u - cx) * depth_m / fx,
                (center_v - cy) * depth_m / fy,
                depth_m]], dtype=np.float32)
            point_source = (
                point_camera @ source_from_camera[:3, :3].T +
                source_from_camera[:3, 3])[0]
            track = matched_tracks.get(index)
            if track is not None:
                continue
            state = "CAMERA_DEPTH_CANDIDATE"
            candidates.append({
                "id": index,
                "position": point_source,
                "class_name": detection.class_name,
                "score": float(detection.score),
                "depth": depth_m,
                "track_id": -1,
                "state": state,
            })
        return candidates


def main():
    import rclpy

    args = parse_config()
    rclpy.init()
    node = ObservationTrackerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            # ros2 launch can forward SIGINT while rclpy is releasing handles.
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
