#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import cv2
import torch
import rosbag
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils.utils import time_synchronized, select_device, increment_path, scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, AverageMeter, letterbox
import signal
import sys

class LoadImagesFromBag:
    def __init__(self, topic_name, depth_topic_name, camera_info_topic_name, img_size=640, stride=32):
        self.topic_name = topic_name
        self.depth_topic_name = depth_topic_name
        self.camera_info_topic_name = camera_info_topic_name
        self.img_size = img_size
        self.bridge = CvBridge()
        self.image = None
        self.depth_image = None
        self.camera_info = None
        self.stride = stride
        rospy.init_node('zed_camera_lane_detection_using_yolo_node')
        self.subscriber = rospy.Subscriber(self.topic_name, Image, self.callback)
        self.depth_subscriber = rospy.Subscriber(self.depth_topic_name, Image, self.depth_callback)
        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic_name, CameraInfo, self.camera_info_callback)
        self.rate = rospy.Rate(10)

    def callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV image: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV depth image: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def __iter__(self):
        return self

    def __next__(self):
        while self.image is None or self.depth_image is None or self.camera_info is None:
            pass
        try:
            img = cv2.resize(self.image, (1280, 720))
            img_resized = letterbox(img, self.img_size, stride=self.stride)[0]
            depth_resized = self.depth_image
            camera_info = self.camera_info
            self.image = None
            self.depth_image = None
            self.camera_info = None
            return img_resized, img, depth_resized, camera_info
        except Exception as e:
            print(f"An error occurred: {e}")
            raise StopIteration

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--topic-name', type=str, default='/zed_node/left/image_rect_color', help='name of the topic to read images from')
    parser.add_argument('--depth-topic-name', type=str, default='/zed_node/depth/depth_registered', help='name of the depth topic to read images from')
    parser.add_argument('--camera-info-topic-name', type=str, default='/zed_node/depth/camera_info', help='name of the camera info topic to read from')
    parser.add_argument('--output-topic', type=str, default='/masked_depth_output', help='name of the output topic to publish masked depth images')
    parser.add_argument('--output-camera-info-topic', type=str, default='/masked_depth_output/camera_info', help='name of the output camera info topic')
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
    weights, imgsz = opt.weights, opt.img_size
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)

    if half:
        model.half()  # FP16으로 변환
    model.eval()

    dataset = LoadImagesFromBag(opt.topic_name, opt.depth_topic_name, opt.camera_info_topic_name, img_size=imgsz, stride=stride)

    depth_pub = rospy.Publisher(opt.output_topic, Image, queue_size=10)
    camera_info_pub = rospy.Publisher(opt.output_camera_info_topic, CameraInfo, queue_size=10)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    for img, im0s, depth_img, camera_info in dataset: # resize, 원본, depth 원본, info
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        ll_only_mask = np.where(da_seg_mask == 1, 0, ll_seg_mask)
        
        ll_only_mask_resized = cv2.resize(ll_only_mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask = ll_only_mask_resized * 255
        
        masked_depth = apply_mask_to_depth(depth_img, mask)

        da_seg_mask_colored = np.zeros_like(im0s)
        ll_seg_mask_colored = np.zeros_like(im0s)
        da_seg_mask_colored[da_seg_mask == 1] = [0, 255, 0]  # for driving area
        ll_seg_mask_colored[ll_seg_mask == 1] = [255, 255, 255] # lane mask
        combined_mask = cv2.addWeighted(ll_seg_mask_colored, 0.5, da_seg_mask_colored, 0.5, 0)
        black_mask = cv2.addWeighted(im0s, 1, combined_mask, 0.5, 0)
        ll_only_mask_colored = np.zeros_like(im0s)
        ll_only_mask_colored[ll_only_mask == 1] = [255, 255, 255]
        
        testimg_ll_only = cv2.addWeighted(im0s, 1, ll_only_mask_colored, 0.5, 0)
        
        cv2.imshow('LL Only Mask on Depth', masked_depth)
        cv2.imshow('LL Only Mask Visualization', testimg_ll_only)
        cv2.imshow('test', black_mask)
        cv2.waitKey(1)
        try:
            masked_depth_msg = dataset.bridge.cv2_to_imgmsg(masked_depth, encoding="32FC1")
            timestamp = rospy.Time.now()
            masked_depth_msg.header.stamp = timestamp
            masked_depth_msg.header.frame_id = camera_info.header.frame_id
            camera_info.header.stamp = timestamp

            depth_pub.publish(masked_depth_msg)
            camera_info_pub.publish(camera_info)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image to ROS Image message: {e}")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    opt = make_parser().parse_args()
    print(opt)
    with torch.no_grad():
        detect()
