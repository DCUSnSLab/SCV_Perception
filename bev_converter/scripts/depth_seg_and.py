#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
# 필요한 import만 유지
from ultralytics_ros.msg import YoloResult

class BevNode:
    def __init__(self):
        rospy.init_node('bev_node', anonymous=True)

        # 카메라 이미지 토픽 이름 (launch 파라미터로 지정)
        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        rospy.loginfo("Using camera topic: %s", self.camera_topic)
 
        camera_params = rospy.get_param("camera", {})
        if camera_params:
            self.intrinsics = camera_params.get("intrinsics", {})
            self.extrinsics = camera_params.get("extrinsics", {})
            rospy.loginfo("Camera Intrinsics: %s", self.intrinsics)
            rospy.loginfo("Camera Extrinsics: %s", self.extrinsics)
        else:
            rospy.logwarn("Camera parameters not found on parameter server.")
            rospy.signal_shutdown("No camera parameters found")
            return

        self.h = self.extrinsics.get("translation", [0.0, 0.0, 0.45])[2]

        self.marker_frame = rospy.get_param("~marker_frame", "zed2i_base_link")

        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.loginfo("Subscribed to camera topic: %s", self.camera_topic)

        self.yolo_topic = rospy.get_param("~yolo_topic", "/yolo_result")
        self.yolo_sub = rospy.Subscriber(self.yolo_topic, YoloResult, self.yolo_callback)
        rospy.loginfo("Subscribed to YOLO topic: %s", self.yolo_topic)

        self.marker_pub = rospy.Publisher("~bev_marker", Marker, queue_size=10)

        self.H_inv = self.compute_homography_inv()

    def compute_homography_inv(self):
        
    def image_callback(self, img_msg):

    def yolo_callback(self, detections_msg):

    def image_to_ground(self, u, v, H_inv):


if __name__ == '__main__':
    try:
        bev_node = BevNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
