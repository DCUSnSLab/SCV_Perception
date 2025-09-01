#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import cv_bridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Polygon, Point32
from perception_interface.msg import DetectionResult, DetectionArray
from ultralytics import YOLO
import cv2
from .memory_sort import MemorySORT


class SegNode(Node):
    def __init__(self):
        super().__init__("ultralytics_seg_node")

        # ---- Declare parameters ----
        self.declare_parameter("yolo_model", "yolo11n-seg.pt")
        self.declare_parameter("input_topic", "image_raw")
        self.declare_parameter("result_image_topic", "yolo/seg_image")
        self.declare_parameter("detection_topic", "yolo/detections")

        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", "")     # 문자열로 받기
        self.declare_parameter("device", "")      # 문자열로 받기

        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 0)  # int
        self.declare_parameter("result_font_size", 0)   # int
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)
        
        # Memory-SORT tracker parameters
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("iou_thr_active", 0.3)
        self.declare_parameter("iou_thr_memory", 0.2)
        self.declare_parameter("min_hits", 3)
        self.declare_parameter("memory_ttl_sec", 1.0)
        self.declare_parameter("use_mask_assoc", False)

        # ---- Read parameters ----
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.result_image_topic = self.get_parameter("result_image_topic").get_parameter_value().string_value
        self.detection_topic = self.get_parameter("detection_topic").get_parameter_value().string_value

        self.conf_thres = self.get_parameter("conf_thres").value
        self.iou_thres = self.get_parameter("iou_thres").value
        self.max_det = int(self.get_parameter("max_det").value)
        classes_param = self.get_parameter("classes").value
        self.classes = None if classes_param == "" else [int(x.strip()) for x in classes_param.split(",") if x.strip()]
        device_param = self.get_parameter("device").value
        self.device = None if device_param == "" else device_param

        self.result_conf = self.get_parameter("result_conf").value
        self.result_line_width = self.get_parameter("result_line_width").value
        self.result_font_size = self.get_parameter("result_font_size").value
        self.result_font = self.get_parameter("result_font").get_parameter_value().string_value
        self.result_labels = self.get_parameter("result_labels").value
        self.result_boxes = self.get_parameter("result_boxes").value

        # ---- Resolve model path ----
        # 절대/상대 경로가 들어오면 그대로 사용, 파일명만 들어오면 패키지 share/model/ 에서 탐색
        model_path = yolo_model
        if "/" not in yolo_model:
            try:
                pkg_share = get_package_share_directory("ultralytics_ros2")
                model_path = f"{pkg_share}/model/{yolo_model}"
            except Exception:
                pass

        self.get_logger().info(f"[ultralytics_ros2] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass

        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")
        
        # Tracker parameters  
        self.enable_tracking = self.get_parameter("enable_tracking").value
        self.iou_thr_active = self.get_parameter("iou_thr_active").value
        self.iou_thr_memory = self.get_parameter("iou_thr_memory").value
        self.min_hits = self.get_parameter("min_hits").value
        self.memory_ttl_sec = self.get_parameter("memory_ttl_sec").value
        self.use_mask_assoc = self.get_parameter("use_mask_assoc").value

        # Initialize tracker
        self.tracker = None
        self.fps_estimate = 30.0  # Initial estimate
        self.prev_timestamp = None

        # ---- IO ----
        self.sub = self.create_subscription(
            Image, self.input_topic, self.image_cb, qos_profile_sensor_data
        )
        self.pub_img = self.create_publisher(Image, self.result_image_topic, QoSProfile(depth=1))
        self.pub_detections = self.create_publisher(DetectionArray, self.detection_topic, QoSProfile(depth=1))

        self._last_log_t = 0.0
        self.get_logger().info("ultralytics_seg_node ready.")

    def _init_tracker(self, fps):
        """Initialize tracker with estimated FPS"""
        if self.tracker is None and self.enable_tracking:
            memory_ttl_frames = int(self.memory_ttl_sec * fps)
            self.tracker = MemorySORT(
                iou_thr_active=self.iou_thr_active,
                iou_thr_memory=self.iou_thr_memory,
                min_hits=self.min_hits,
                memory_ttl_frames=memory_ttl_frames,
                use_mask_assoc=self.use_mask_assoc and self.use_segmentation,
                output_masks=self.use_segmentation
            )

    def image_cb(self, msg: Image):
        t0 = time.perf_counter()
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Calculate FPS for tracker initialization
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_timestamp is not None:
            dt = current_time - self.prev_timestamp
            if 0.01 < dt < 1.0:  # Valid delta time
                self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * (1.0 / dt)
        self.prev_timestamp = current_time
        
        # Initialize tracker if needed
        self._init_tracker(self.fps_estimate)

        # YOLO Detection
        results = self.model.predict(
            source=cv_img,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )
        if not results:
            return

        res = results[0]
        
        # Prepare detections for tracker
        dets_for_tracker = []
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = res.boxes.conf.cpu().numpy()
            class_ids = res.boxes.cls.cpu().numpy().astype(int)
            
            # Get masks if available (segmentation)
            masks_np = None
            if res.masks is not None:
                masks_np = res.masks.data.cpu().numpy()
            
            # Convert to tracker format: [x1, y1, x2, y2, conf, cls, mask]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                cls_id = class_ids[i]
                mask = None
                if masks_np is not None:
                    mask = masks_np[i]
                    # Resize mask to image size
                    H, W = cv_img.shape[:2]
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                dets_for_tracker.append([x1, y1, x2, y2, conf, cls_id, mask])
        
        # Run tracker
        tracks = []
        if self.enable_tracking and self.tracker is not None:
            dt = 1.0 / max(self.fps_estimate, 1.0)
            tracks = self.tracker.update(dets_for_tracker, dt)
        
        # Create detection array message
        detection_array = DetectionArray()
        detection_array.header = msg.header
        detection_array.source_image = msg
        detection_array.model_name = "YOLO11"
        detection_array.model_version = "v11"
        detection_array.detections = []
        
        # Convert tracks back to DetectionResult format
        if self.enable_tracking and tracks:
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2, track_id, cls_id, pred = track[:7]
                    mask = track[7] if len(track) >= 8 else None
                else:
                    continue
                    
                detection = DetectionResult()
                detection.header = msg.header
                
                # Class name and confidence (use 1.0 for tracked objects)
                detection.class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                detection.confidence = 1.0  # Tracked objects have high confidence
                
                # Track ID from Memory-SORT
                detection.track_id = int(track_id)
                
                # Bounding box as polygon (use track coordinates)
                bbox_polygon = Polygon()
                bbox_polygon.points = [
                    Point32(x=float(x1), y=float(y1), z=0.0),
                    Point32(x=float(x2), y=float(y1), z=0.0),
                    Point32(x=float(x2), y=float(y2), z=0.0),
                    Point32(x=float(x1), y=float(y2), z=0.0)
                ]
                detection.bounding_box = bbox_polygon
                
                # Centroid
                detection.centroid = Point(
                    x=float((x1 + x2) / 2),
                    y=float((y1 + y2) / 2),
                    z=0.0
                )
                
                # Segmentation mask if available
                if mask is not None and isinstance(mask, np.ndarray):
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    detection.mask = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                else:
                    # Create empty mask
                    empty_mask = np.zeros((cv_img.shape[0], cv_img.shape[1]), dtype=np.uint8)
                    detection.mask = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
                
                detection_array.detections.append(detection)
        else:
            # Fallback: use raw detections without tracking
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = res.boxes.conf.cpu().numpy()
                class_ids = res.boxes.cls.cpu().numpy().astype(int)
                
                # Get masks if available (segmentation)
                masks = None
                if res.masks is not None:
                    masks = res.masks.data.cpu().numpy()
                
                for i in range(len(boxes)):
                    detection = DetectionResult()
                    detection.header = msg.header
                    
                    # Class name and confidence
                    detection.class_name = self.model.names[class_ids[i]]
                    detection.confidence = float(confidences[i])
                    
                    # Track ID (not available in detection mode, set to -1)
                    detection.track_id = -1
                    
                    # Bounding box as polygon
                    x1, y1, x2, y2 = boxes[i]
                    bbox_polygon = Polygon()
                    bbox_polygon.points = [
                        Point32(x=float(x1), y=float(y1), z=0.0),
                        Point32(x=float(x2), y=float(y1), z=0.0),
                        Point32(x=float(x2), y=float(y2), z=0.0),
                        Point32(x=float(x1), y=float(y2), z=0.0)
                    ]
                    detection.bounding_box = bbox_polygon
                    
                    # Centroid
                    detection.centroid = Point(
                        x=float((x1 + x2) / 2),
                        y=float((y1 + y2) / 2),
                        z=0.0
                    )
                    
                    # Segmentation mask if available
                    if masks is not None and i < len(masks):
                        mask = masks[i]
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        detection.mask = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                    else:
                        # Create empty mask
                        empty_mask = np.zeros((cv_img.shape[0], cv_img.shape[1]), dtype=np.uint8)
                        detection.mask = self.bridge.cv2_to_imgmsg(empty_mask, encoding="mono8")
                    
                    detection_array.detections.append(detection)
        
        # Publish detection array
        self.pub_detections.publish(detection_array)
        
        # Create visualization
        vis_img = res.plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        
        # Add track IDs to visualization
        if self.enable_tracking and tracks:
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2, track_id, cls_id, pred = track[:7]
                    
                    # Draw track ID text
                    text = f"ID:{int(track_id)}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    color = (0, 255, 255)  # Yellow
                    thickness = 2
                    
                    # Position: right side of bbox, slightly below top
                    text_x = int(x2 - 60)  # Right side of bbox
                    text_y = int(y1 + 20)  # Slightly below top
                    
                    # Draw text with outline for better visibility
                    cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)  # Black outline
                    cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, color, thickness)  # Yellow text
        
        out_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

        ms = (time.perf_counter() - t0) * 1000.0
        now = time.time()
        if now - self._last_log_t >= 1.0:
            self.get_logger().info(f"[ultralytics] 1 frame = {ms:.1f} ms")
            self._last_log_t = now


def main():
    rclpy.init()
    node = SegNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
