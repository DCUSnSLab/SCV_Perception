#!/usr/bin/env python3
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, Polygon
from perception_interface.msg import DetectionArray, DetectionResult
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None

try:
    import torch
except ImportError:
    torch = None


TARGET_CLASSES = {'person', 'car', 'bus', 'truck', 'motorcycle'}
CLASS_COLORS = {
    'person': (0, 255, 0),
    'car': (0, 165, 255),
    'bus': (255, 140, 0),
    'truck': (255, 0, 0),
    'motorcycle': (255, 255, 0),
}
ANNOTATED_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class YoloDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()
        self.last_inference_time = None
        self.latest_depth_frame: Optional[np.ndarray] = None
        self.latest_depth_encoding = ''
        self.latest_depth_stamp_ns: Optional[int] = None

        self.declare_parameter(
            'model_path',
            '/home/scv/SCV/src/perception/yolov8n.pt',
            ParameterDescriptor(description='Absolute path to the YOLOv8 model file.'),
        )
        self.declare_parameter(
            'image_topic',
            '/realsense_1/color/image_raw/compressed',
            ParameterDescriptor(description='Input color image topic.'),
        )
        self.declare_parameter(
            'use_compressed_image',
            True,
            ParameterDescriptor(description='Subscribe to sensor_msgs/CompressedImage instead of sensor_msgs/Image.'),
        )
        self.declare_parameter(
            'depth_topic',
            '/realsense_1/aligned_depth_to_color/image_raw',
            ParameterDescriptor(description='Aligned depth image topic in the color frame.'),
        )
        self.declare_parameter(
            'annotated_image_topic',
            '~/annotated_image',
            ParameterDescriptor(description='Annotated output image topic.'),
        )
        self.declare_parameter(
            'detections_topic',
            '~/detections',
            ParameterDescriptor(description='DetectionArray output topic.'),
        )
        self.declare_parameter(
            'confidence_threshold',
            0.35,
            ParameterDescriptor(description='Minimum confidence for accepted detections.'),
        )
        self.declare_parameter(
            'iou_threshold',
            0.45,
            ParameterDescriptor(description='IoU threshold used by YOLO NMS.'),
        )
        self.declare_parameter(
            'device',
            'auto',
            ParameterDescriptor(description='Inference device string such as auto, cpu, cuda, cuda:0.'),
        )
        self.declare_parameter(
            'publish_annotated_image',
            True,
            ParameterDescriptor(description='Whether to publish the annotated image.'),
        )
        self.declare_parameter(
            'target_classes',
            list(TARGET_CLASSES),
            ParameterDescriptor(description='YOLO class names to keep.'),
        )
        self.declare_parameter(
            'depth_roi_scale',
            0.35,
            ParameterDescriptor(description='Scale factor for the inner bbox ROI used for depth sampling.'),
        )
        self.declare_parameter(
            'max_depth_age_sec',
            0.2,
            ParameterDescriptor(description='Maximum age difference allowed between color and depth frames.'),
        )
        self.declare_parameter(
            'imgsz',
            640,
            ParameterDescriptor(description='YOLO inference image size. Lower values improve throughput.'),
        )
        self.declare_parameter(
            'frame_skip',
            0,
            ParameterDescriptor(description='Skip N incoming frames between inference runs. 0 means infer every frame.'),
        )
        self.declare_parameter(
            'use_half',
            True,
            ParameterDescriptor(description='Use FP16 inference on CUDA when available.'),
        )
        self.declare_parameter(
            'publish_every_n_frames',
            1,
            ParameterDescriptor(description='Publish the annotated image every N processed frames.'),
        )

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.use_compressed_image = self.get_parameter('use_compressed_image').value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.annotated_image_topic = self.get_parameter('annotated_image_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        requested_device = self.get_parameter('device').get_parameter_value().string_value
        self.publish_annotated_image = self.get_parameter('publish_annotated_image').value
        self.target_classes = set(self.get_parameter('target_classes').value)
        self.depth_roi_scale = float(self.get_parameter('depth_roi_scale').value)
        self.max_depth_age_sec = float(self.get_parameter('max_depth_age_sec').value)
        self.imgsz = max(32, int(self.get_parameter('imgsz').value))
        self.frame_skip = max(0, int(self.get_parameter('frame_skip').value))
        self.use_half = bool(self.get_parameter('use_half').value)
        self.publish_every_n_frames = max(1, int(self.get_parameter('publish_every_n_frames').value))
        self.input_frame_count = 0
        self.processed_frame_count = 0

        if YOLO is None:
            raise RuntimeError(
                'ultralytics is not installed. Install ultralytics and torch in the runtime environment.'
            ) from YOLO_IMPORT_ERROR
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'YOLO model file not found: {self.model_path}')

        self.model = YOLO(self.model_path)
        self.class_name_to_id = self._build_class_lookup(self.model.names)
        self.device = self._resolve_device(requested_device)

        if self.use_compressed_image:
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.compressed_image_callback,
                qos_profile_sensor_data,
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.detections_pub = self.create_publisher(
            DetectionArray,
            self.detections_topic,
            10,
        )
        self.annotated_pub = None
        if self.publish_annotated_image:
            self.annotated_pub = self.create_publisher(
                Image,
                self.annotated_image_topic,
                ANNOTATED_IMAGE_QOS,
            )

        active_classes = sorted(self.target_classes.intersection(self.class_name_to_id.keys()))
        self.get_logger().info(
            f'Loaded YOLO model from {self.model_path}. '
            f'Input topic: {self.image_topic}. '
            f'Compressed input: {self.use_compressed_image}. '
            f'Depth topic: {self.depth_topic}. '
            f'Device: {self.device}. '
            f'Image size: {self.imgsz}. '
            f'Frame skip: {self.frame_skip}. '
            f'FP16: {self._use_half_precision()}. '
            f'Annotated publish interval: {self.publish_every_n_frames}. '
            f'Keeping classes: {active_classes}'
        )

    @staticmethod
    def _build_class_lookup(names: Dict[int, str]) -> Dict[str, int]:
        return {class_name: class_id for class_id, class_name in names.items()}

    def _target_class_ids(self) -> List[int]:
        return [
            self.class_name_to_id[name]
            for name in sorted(self.target_classes)
            if name in self.class_name_to_id
        ]

    def _resolve_device(self, requested_device: str) -> str:
        normalized = (requested_device or 'auto').strip().lower()
        if normalized != 'auto':
            if normalized.startswith('cuda') and not self._cuda_available():
                self.get_logger().warn(
                    f"Requested device '{requested_device}' but CUDA is unavailable. Falling back to cpu."
                )
                return 'cpu'
            return requested_device

        if self._cuda_available():
            return 'cuda:0'
        return 'cpu'

    @staticmethod
    def _cuda_available() -> bool:
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _use_half_precision(self) -> bool:
        return self.use_half and self.device.startswith('cuda')

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth_encoding = msg.encoding
        self.latest_depth_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        )

    def compressed_image_callback(self, msg: CompressedImage) -> None:
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        source_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        source_image.header = msg.header
        self._run_inference(frame, source_image)

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._run_inference(frame, msg)

    def _run_inference(self, frame: np.ndarray, source_msg: Image) -> None:
        self.input_frame_count += 1
        if self.frame_skip > 0 and (self.input_frame_count - 1) % (self.frame_skip + 1) != 0:
            return

        start_time = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=float(self.confidence_threshold),
            iou=float(self.iou_threshold),
            classes=self._target_class_ids(),
            device=self.device,
            imgsz=self.imgsz,
            half=self._use_half_precision(),
            verbose=False,
        )
        inference_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_inference_time = inference_ms
        self.processed_frame_count += 1

        detection_array = DetectionArray()
        detection_array.header = source_msg.header
        detection_array.source_image = source_msg
        detection_array.model_name = 'yolov8n'
        detection_array.model_version = 'local_pt'

        annotated = frame.copy()
        if results:
            depth_frame = self._get_synced_depth_frame(source_msg)
            self._fill_detections(results[0], source_msg, detection_array, annotated, depth_frame)

        self.detections_pub.publish(detection_array)
        should_publish_annotated = (
            self.annotated_pub is not None and
            (self.processed_frame_count - 1) % self.publish_every_n_frames == 0
        )
        if should_publish_annotated:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = source_msg.header
            self.annotated_pub.publish(annotated_msg)

        self.get_logger().debug(
            f'Published {len(detection_array.detections)} detections in {inference_ms:.1f} ms'
        )

    def _fill_detections(
        self,
        result,
        source_msg: Image,
        detection_array: DetectionArray,
        annotated: np.ndarray,
        depth_frame: Optional[np.ndarray],
    ) -> None:
        boxes = result.boxes
        if boxes is None:
            return

        names = result.names
        for box in boxes:
            cls_idx = int(box.cls.item())
            class_name = names.get(cls_idx, str(cls_idx))
            if class_name not in self.target_classes:
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            depth_m = self._estimate_depth_meters(depth_frame, x1, y1, x2, y2)

            detection = DetectionResult()
            detection.header = source_msg.header
            detection.class_name = class_name
            detection.confidence = confidence
            detection.bounding_box = self._bbox_to_polygon(x1, y1, x2, y2)
            detection.centroid.x = (x1 + x2) / 2.0
            detection.centroid.y = (y1 + y2) / 2.0
            detection.centroid.z = depth_m if depth_m is not None else float('nan')
            detection.track_id = -1
            detection.mask = Image()
            detection_array.detections.append(detection)

            self._draw_detection(annotated, class_name, confidence, depth_m, x1, y1, x2, y2)

    def _get_synced_depth_frame(self, color_msg: Image) -> Optional[np.ndarray]:
        if self.latest_depth_frame is None or self.latest_depth_stamp_ns is None:
            return None
        color_stamp_ns = int(color_msg.header.stamp.sec) * 1_000_000_000 + int(color_msg.header.stamp.nanosec)
        age_sec = abs(color_stamp_ns - self.latest_depth_stamp_ns) / 1_000_000_000.0
        if age_sec > self.max_depth_age_sec:
            return None
        return self.latest_depth_frame

    def _estimate_depth_meters(
        self,
        depth_frame: Optional[np.ndarray],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Optional[float]:
        if depth_frame is None:
            return None

        height, width = depth_frame.shape[:2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        roi_w = max(3, int(box_w * self.depth_roi_scale))
        roi_h = max(3, int(box_h * self.depth_roi_scale))

        rx1 = max(0, int(cx - roi_w / 2))
        ry1 = max(0, int(cy - roi_h / 2))
        rx2 = min(width, int(cx + roi_w / 2))
        ry2 = min(height, int(cy + roi_h / 2))
        if rx2 <= rx1 or ry2 <= ry1:
            return None

        roi = depth_frame[ry1:ry2, rx1:rx2]
        roi = roi[np.isfinite(roi)]
        roi = roi[roi > 0]
        if roi.size == 0:
            return None

        if self.latest_depth_encoding == '16UC1':
            roi = roi.astype(np.float32) / 1000.0
        else:
            roi = roi.astype(np.float32)

        roi = roi[(roi > 0.1) & (roi < 50.0)]
        if roi.size == 0:
            return None
        return float(np.median(roi))

    @staticmethod
    def _bbox_to_polygon(x1: float, y1: float, x2: float, y2: float) -> Polygon:
        polygon = Polygon()
        polygon.points = [
            Point32(x=x1, y=y1, z=0.0),
            Point32(x=x2, y=y1, z=0.0),
            Point32(x=x2, y=y2, z=0.0),
            Point32(x=x1, y=y2, z=0.0),
        ]
        return polygon

    @staticmethod
    def _draw_detection(
        image: np.ndarray,
        class_name: str,
        confidence: float,
        depth_m: Optional[float],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(image, pt1, pt2, color, 2)
        label = f'{class_name} {confidence:.2f}'
        if depth_m is not None:
            label += f' {depth_m:.1f}m'
        cv2.putText(
            image,
            label,
            (pt1[0], max(20, pt1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def main(args: Iterable[str] = None) -> None:
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
