#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ultralytics_ros – TrackerNode (det+seg mask publisher)

import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult


class TrackerNode:
    def __init__(self):
        # ── parameters ──────────────────────────────────────────────────────────
        yolo_model           = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.input_topic     = rospy.get_param("~input_topic", "image_raw")
        self.result_topic    = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.seg_mask_topic  = rospy.get_param("~seg_mask_topic", "yolo_seg_mask")

        self.conf_thres      = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres       = rospy.get_param("~iou_thres", 0.45)
        self.max_det         = rospy.get_param("~max_det", 300)
        self.classes         = rospy.get_param("~classes", None)
        self.device          = rospy.get_param("~device", None)

        # result-image overlay options
        self.result_conf        = rospy.get_param("~result_conf", True)
        self.result_line_width  = rospy.get_param("~result_line_width", None)
        self.result_font_size   = rospy.get_param("~result_font_size", None)
        self.result_font        = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels      = rospy.get_param("~result_labels", True)
        self.result_boxes       = rospy.get_param("~result_boxes", True)

        # ── model ───────────────────────────────────────────────────────────────
        pkg_path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{pkg_path}/models/{yolo_model}")
        self.model.fuse()

        # ── ROS I/O ─────────────────────────────────────────────────────────────
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

        self.sub = rospy.Subscriber(
            self.input_topic, Image, self.image_callback,
            queue_size=1, buff_size=2**24
        )
        self.results_pub       = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub  = rospy.Publisher(self.result_image_topic, Image, queue_size=1)
        self.seg_mask_pub      = rospy.Publisher(self.seg_mask_topic, Image, queue_size=1)

    # ───────────────────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.predict(
            source=cv_image,
            conf=self.conf_thres, iou=self.iou_thres, max_det=self.max_det,
            classes=self.classes, device=self.device,
            verbose=False, retina_masks=True
        )

        if not results:
            return

        # ----- detection results -----
        yolo_res_msg             = YoloResult()
        yolo_res_msg.header      = msg.header
        yolo_res_msg.detections  = self.create_detections_array(results)
        if self.use_segmentation:
            yolo_res_msg.masks = self.create_segmentation_mask_msgs(results)

        self.results_pub.publish(yolo_res_msg)

        # ----- overlay image -----
        overlay_msg = self.create_result_image(results, msg.header)
        self.result_image_pub.publish(overlay_msg)

        # ----- combined seg mask (mono8) -----
        if self.use_segmentation:
            seg_mask_msg = self.create_combined_mask_image(results, msg.header)
            if seg_mask_msg is not None:
                self.seg_mask_pub.publish(seg_mask_msg)

    # ───────────────────────────────────────────────────────────────────────────
    # helpers
    def create_detections_array(self, results):
        det_array = Detection2DArray()
        bbox_xywh = results[0].boxes.xywh
        cls_ids   = results[0].boxes.cls
        confs     = results[0].boxes.conf
        for bbox, cid, conf in zip(bbox_xywh, cls_ids, confs):
            det                = Detection2D()
            det.bbox.center.x  = float(bbox[0])
            det.bbox.center.y  = float(bbox[1])
            det.bbox.size_x    = float(bbox[2])
            det.bbox.size_y    = float(bbox[3])
            hyp                = ObjectHypothesisWithPose()
            hyp.id, hyp.score  = int(cid), float(conf)
            det.results.append(hyp)
            det_array.detections.append(det)
        return det_array

    def create_result_image(self, results, header):
        vis_img   = results[0].plot(
            conf=self.result_conf, line_width=self.result_line_width,
            font_size=self.result_font_size, font=self.result_font,
            labels=self.result_labels, boxes=self.result_boxes
        )
        return self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8", header=header)

    # ---- individual mask list (for YoloResult.msg compatibility) -------------
    def create_segmentation_mask_msgs(self, results):
        mask_msgs = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                # result.masks.data shape: (N, H, W)
                for m in result.masks.data:
                    mono = (m.to("cpu").numpy() * 255).astype(np.uint8)
                    mask_msgs.append(self.bridge.cv2_to_imgmsg(mono, encoding="mono8"))
        return mask_msgs

    # ---- combined binary mask image -----------------------------------------
    def create_combined_mask_image(self, results, header):
        res = results[0]
        if not (hasattr(res, "masks") and res.masks is not None):
            return None

        masks_tensor = res.masks.data  # (N, H, W)
        combined     = (masks_tensor.sum(dim=0) > 0).to("cpu").numpy().astype(np.uint8) * 255
        return self.bridge.cv2_to_imgmsg(combined, encoding="mono8", header=header)


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    TrackerNode()
    rospy.spin()
