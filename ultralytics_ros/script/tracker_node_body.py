#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv_bridge
import numpy as np
import roslib.packages
import rospy
import math
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult, Keypoint

def is_person_standing(keypoints, confidence_threshold=0.5, angle_threshold=20):
    """
    COCO 17 keypoint 순서 (인덱스):
      0: 코, 1: 왼쪽 눈, 2: 오른쪽 눈, 3: 왼쪽 귀, 4: 오른쪽 귀,
      5: 왼쪽 어깨, 6: 오른쪽 어깨, 7: 왼쪽 팔꿈치, 8: 오른쪽 팔꿈치,
      9: 왼쪽 손목, 10: 오른쪽 손목, 11: 왼쪽 엉덩이, 12: 오른쪽 엉덩이,
      13: 왼쪽 무릎, 14: 오른쪽 무릎, 15: 왼쪽 발목, 16: 오른쪽 발목
      
    어깨(5,6)와 엉덩이(11,12) 좌표가 유효한 경우, 
    어깨와 엉덩이 중간점을 잇는 벡터 각도가 수직(20° 이하)에 가까우면 서 있다고 판단합니다.
    """
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    if left_shoulder[2] < confidence_threshold or right_shoulder[2] < confidence_threshold:
        return False
    if left_hip[2] < confidence_threshold or right_hip[2] < confidence_threshold:
        return False

    if (left_shoulder[0] == 0.0 and left_shoulder[1] == 0.0) or \
       (right_shoulder[0] == 0.0 and right_shoulder[1] == 0.0) or \
       (left_hip[0] == 0.0 and left_hip[1] == 0.0) or \
       (right_hip[0] == 0.0 and right_hip[1] == 0.0):
        return False

    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_x = (left_hip[0] + right_hip[0]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle <= angle_threshold

def is_person_lying_down(keypoints, confidence_threshold=0.5, lying_angle_threshold=70):
    """
    COCO 17 keypoint 순서에 따라 어깨와 엉덩이(인덱스 5,6,11,12) 좌표가 유효한 경우,
    어깨-엉덩이 중간점 사이의 벡터 각도가 높으면(수평에 가까워 70° 이상이면) 사람이 누워 있다고 판단합니다.
    """
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    if left_shoulder[2] < confidence_threshold or right_shoulder[2] < confidence_threshold:
        return False
    if left_hip[2] < confidence_threshold or right_hip[2] < confidence_threshold:
        return False

    if (left_shoulder[0] == 0.0 and left_shoulder[1] == 0.0) or \
       (right_shoulder[0] == 0.0 and right_shoulder[1] == 0.0) or \
       (left_hip[0] == 0.0 and left_hip[1] == 0.0) or \
       (right_hip[0] == 0.0 and right_hip[1] == 0.0):
        return False

    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_x = (left_hip[0] + right_hip[0]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    # 각도가 70° 이상이면 거의 수평, 즉 누워 있다고 판단
    return angle >= lying_angle_threshold

class TrackerNode:
    def __init__(self):
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", None)
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(
            self.result_image_topic, Image, queue_size=1
        )
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )
        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)

            print("dddddddddddddddddddddddddddddddddd")
            # keypoints 데이터를 detection별로 그룹화하여 추가 및 자세 판단
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                # keypoints_array shape: (num_detections, num_keypoints, 3)
                keypoints_array = results[0].keypoints.data.cpu().numpy()
                print("Keypoints array from model:", keypoints_array)
                keypoints_msg_list = []
                keypoint_counts = []
                # 각 detection에 대해 처리
                for det_idx, det_keypoints in enumerate(keypoints_array):
                    count = 0
                    valid_count = 0
                    for kp in det_keypoints:
                        kp_msg = Keypoint()
                        kp_msg.x = float(kp[0])
                        kp_msg.y = float(kp[1])
                        kp_msg.confidence = float(kp[2])
                        keypoints_msg_list.append(kp_msg)
                        count += 1
                        if (kp[0] != 0.0 or kp[1] != 0.0) and kp[2] >= 0.5:
                            valid_count += 1
                    keypoint_counts.append(count)
                    # 어깨(5,6)와 엉덩이(11,12) keypoint가 유효한지 먼저 확인
                    shoulder_hip_valid = (
                        ((det_keypoints[5][0] != 0.0 or det_keypoints[5][1] != 0.0) and det_keypoints[5][2] >= 0.5) and
                        ((det_keypoints[6][0] != 0.0 or det_keypoints[6][1] != 0.0) and det_keypoints[6][2] >= 0.5) and
                        ((det_keypoints[11][0] != 0.0 or det_keypoints[11][1] != 0.0) and det_keypoints[11][2] >= 0.5) and
                        ((det_keypoints[12][0] != 0.0 or det_keypoints[12][1] != 0.0) and det_keypoints[12][2] >= 0.5)
                    )
                    if shoulder_hip_valid:
                        if is_person_standing(det_keypoints, confidence_threshold=0.5, angle_threshold=20):
                            print("Detection {}: Person is standing.".format(det_idx))
                        elif is_person_lying_down(det_keypoints, confidence_threshold=0.5, lying_angle_threshold=70):
                            print("Detection {}: Person is lying down.".format(det_idx))
                        else:
                            print("Detection {}: Person posture is ambiguous.".format(det_idx))
                    else:
                        print("Detection {}: Insufficient valid shoulder/hip keypoints.".format(det_idx))
                yolo_result_msg.keypoint_counts = keypoint_counts
                yolo_result_msg.keypoints = keypoints_msg_list
            print("dddddddddddddddddddddddddddddddddd")
            
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(np.uint8)
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(mask_numpy, encoding="mono8")
                    masks_msg.append(mask_image_msg)
        return masks_msg


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
