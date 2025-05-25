#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class DepthCorrectionNode:
    def __init__(self):
        # Parameters
        self.min_d = rospy.get_param('~min_depth', 0.1)
        self.max_d = rospy.get_param('~max_depth', 50.0)
        input_topic = rospy.get_param('~input_topic', '/depth_anything/depth_registered/image_rect')
        output_topic = rospy.get_param('~output_topic', '/depth_anything/depth_correction/image')

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(output_topic, Image, queue_size=1)
        self.sub = rospy.Subscriber(input_topic, Image, self.callback, queue_size=1)

        rospy.loginfo('DepthCorrectionNode initialized, subscribing to %s, publishing to %s', input_topic, output_topic)

    def callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            rospy.logerr('CvBridge Error: %s', e)
            return

        # 1) Clamping
        depth = np.clip(depth, self.min_d, self.max_d)
        # 2) NaN / Inf patch -> replace with 0
        depth = np.nan_to_num(depth, nan=0.0, posinf=self.max_d, neginf=self.min_d)
        # 3) Median Blur (kernel 3)
        # OpenCV expects 8U/C1 or CV_8U for medianBlur, so convert
        # Scale to millimeters and convert to uint16
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_mm = cv2.medianBlur(depth_mm, 3)
        depth = (depth_mm.astype(np.float32)) / 1000.0
        # (4) Bilateral Filter까지 마친 depth -> depth_bf
        depth_bf = cv2.bilateralFilter(depth, d=5, sigmaColor=1.0, sigmaSpace=1.0)

        # 5) Unsharp Mask
        gauss = cv2.GaussianBlur(depth_bf, (5, 5), 0)
        # alpha > 1.0 이면 샤프닝 강도 up, beta < 0 (negative) 이면 블러 보정
        alpha, beta = 1.3, -0.3
        depth = cv2.addWeighted(depth_bf, alpha, gauss, beta, 0)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
            out_msg.header = msg.header
            self.pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logerr('CvBridge publish error: %s', e)

if __name__ == '__main__':
    rospy.init_node('depth_correction_node')
    node = DepthCorrectionNode()
    rospy.spin()
