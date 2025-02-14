import argparse
import time
import cv2
import torch
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils.utils import time_synchronized, select_device, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, show_seg_result, AverageMeter, letterbox
import signal
import sys

class LoadImagesFromBag:
    def __init__(self, depth_topic_name, camera_info_topic_name, mask_topic_name, img_size=640, stride=32):
        self.depth_topic_name = depth_topic_name
        self.camera_info_topic_name = camera_info_topic_name
        self.mask_topic_name = mask_topic_name
        self.img_size = img_size
        self.stride = stride
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.mask = None
        rospy.init_node('depth_mask_republisher_node', anonymous=True)
        self.depth_subscriber = rospy.Subscriber(self.depth_topic_name, Image, self.depth_callback)
        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic_name, CameraInfo, self.camera_info_callback)
        self.mask_subscriber = rospy.Subscriber(self.mask_topic_name, Image, self.mask_callback)
        self.rate = rospy.Rate(10)

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            rospy.logerr(f"Error converting ROS Depth Image message to OpenCV depth image: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def mask_callback(self, msg):
        try:
            self.mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr(f"Error converting ROS Mask Image message to OpenCV image: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        while self.depth_image is None or self.camera_info is None or self.mask is None:
            self.rate.sleep()
        try:
            depth_resized = cv2.resize(self.depth_image, (1280, 720))
            mask_resized = cv2.resize(self.mask, (1280, 720))
            camera_info = self.camera_info
            # Reset the images and camera info for the next iteration
            self.depth_image = None
            self.mask = None
            self.camera_info = None
            return depth_resized, mask_resized, camera_info
        except Exception as e:
            rospy.logerr(f"An error occurred: {e}")
            raise StopIteration

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-topic-name', type=str, default='/zed2i/zed_node/depth/depth_registered', help='name of the depth topic to read images from')
    parser.add_argument('--camera-info-topic-name', type=str, default='/zed2i/zed_node/depth/camera_info', help='name of the camera info topic to read from')
    parser.add_argument('--mask-topic-name', type=str, default='/yolo/mask_output', help='name of the mask topic to read from')
    parser.add_argument('--output-topic', type=str, default='/masked_depth_output', help='name of the output topic to publish masked depth images')
    parser.add_argument('--output-camera-info-topic', type=str, default='/masked_depth_output/camera_info', help='name of the output camera info topic to publish')
    return parser

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)

def apply_mask_to_depth(depth_image, mask):
    mask_normalized = mask / 255.0
    masked_depth = depth_image * mask_normalized
    masked_depth = masked_depth.astype(np.float32)
    return masked_depth

def detect():
    opt = make_parser().parse_args()

    dataset = LoadImagesFromBag(opt.depth_topic_name, opt.camera_info_topic_name, opt.mask_topic_name)

    depth_pub = rospy.Publisher(opt.output_topic, Image, queue_size=10)
    camera_info_pub = rospy.Publisher(opt.output_camera_info_topic, CameraInfo, queue_size=10)

    for depth_img, mask, camera_info in dataset:
        masked_depth = apply_mask_to_depth(depth_img, mask)

        try:
            masked_depth_msg = dataset.bridge.cv2_to_imgmsg(masked_depth, encoding="32FC1")
            depth_pub.publish(masked_depth_msg)
            camera_info_pub.publish(camera_info)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image to ROS Image message: {e}")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    detect()
