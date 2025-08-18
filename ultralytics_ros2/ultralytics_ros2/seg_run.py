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
from ultralytics import YOLO


class SegNode(Node):
    def __init__(self):
        super().__init__("ultralytics_seg_node")

        # ---- Declare parameters ----
        self.declare_parameter("yolo_model", "yolo11n-seg.pt")
        self.declare_parameter("input_topic", "image_raw")
        self.declare_parameter("result_image_topic", "yolo/seg_image")

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

        # ---- Read parameters ----
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.result_image_topic = self.get_parameter("result_image_topic").get_parameter_value().string_value

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

        # ---- IO ----
        self.sub = self.create_subscription(
            Image, self.input_topic, self.image_cb, qos_profile_sensor_data
        )
        self.pub_img = self.create_publisher(Image, self.result_image_topic, QoSProfile(depth=1))

        self._last_log_t = 0.0
        self.get_logger().info("ultralytics_seg_node ready.")

    def image_cb(self, msg: Image):
        t0 = time.perf_counter()
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

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
        vis_img = res.plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
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
