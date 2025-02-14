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
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import tf
from tf import TransformBroadcaster  # TF 브로드캐스터 추가

class LoadImagesFromBag:
    def __init__(self, topic_name, img_size=640, stride=32):
        self.topic_name = topic_name
        self.img_size = img_size
        self.bridge = CvBridge()
        self.image = None
        self.stride = stride
        rospy.init_node('lane_birdseyesview_using_yolo_node')
        self.subscriber = rospy.Subscriber(self.topic_name, Image, self.callback)
        self.rate = rospy.Rate(10)

    def callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV image: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        while self.image is None:
            pass
        try:
            img = cv2.resize(self.image, (1280, 720))
            img_resized = letterbox(img, self.img_size, stride=self.stride)[0]
            return img_resized, img
        except Exception as e:
            print(f"An error occurred: {e}")
            raise StopIteration

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--topic-name', type=str, default='/zed_node/left/image_rect_color', help='name of the topic to read images from')
    return parser

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)

def transform_to_bev(image, src_points, dst_points):
    width = int(max(dst_points[:, 0]) - min(dst_points[:, 0]))
    height = int(max(dst_points[:, 1]) - min(dst_points[:, 1]))
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, M, (width, height))
    return bev_image

def draw_points(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # 파란색 점
    return image

def image_to_pointcloud(bev_image, frame_id="map"):
    # 빈 포인트 클라우드 배열
    points = []
    
    # 흰색 픽셀을 찾아 3D 좌표로 변환 (임의의 스케일링 적용)
    indices = np.where(bev_image[:, :, 0] > 0)
    for x, y in zip(indices[1], indices[0]):
        # Y 값을 이미지의 y 좌표로 설정 (깊이 Z가 아닌 평면 좌표로 전환)
        X_world = float(x - bev_image.shape[1] / 2) / 100.0  # 이미지 중심을 기준으로 X 좌표 설정
        Y_world = float(bev_image.shape[0] - y) / 100.0  # 상하 반전된 y를 Z로 변환 (깊이)
        Z_world = 0.0  # 평면 좌표이므로 Z를 0으로 설정
        points.append([X_world, Y_world, Z_world])

    # 포인트 클라우드 메시지 생성
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    cloud = pc2.create_cloud_xyz32(header, points)
    return cloud

def publish_tf(br):
    # 'zed2i_base_link'을 부모 프레임으로, 'map'을 자식 프레임으로 설정
    br.sendTransform(
        (0.0, 0.0, 0.0),  # 위치: 변환이 필요없으므로 0으로 설정
        tf.transformations.quaternion_from_euler(0, 0, 0),  # 회전: 필요 없으므로 0으로 설정
        rospy.Time.now(),
        "map",  # 자식 프레임
        "zed2i_base_link"  # 부모 프레임
    )

def detect():
    weights, imgsz = opt.weights, opt.img_size
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)

    if half:
        model.half()
    model.eval()

    dataset = LoadImagesFromBag(opt.topic_name, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    src_points = np.float32([[-2200, 666], [3480, 666], [1280, 440], [0, 440]])
    dst_points = np.float32([[0, 729], [1280, 720], [1280, 0], [0, 0]])

    pcl_publisher = rospy.Publisher('/lane_points', PointCloud2, queue_size=1)
    br = TransformBroadcaster()  # TF 브로드캐스터 초기화

    for img, im0s in dataset:
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

        ll_binary_image = (ll_seg_mask * 255).astype(np.uint8)
        ll_binary_image_color = cv2.cvtColor(ll_binary_image, cv2.COLOR_GRAY2BGR)
        bev_image = transform_to_bev(ll_binary_image_color, src_points, dst_points)
        cv2.imshow('Lane Line Binary', ll_binary_image_color)
        cv2.imshow('bev', bev_image)
        cv2.waitKey(1)

        pcl_msg = image_to_pointcloud(bev_image, frame_id="map")
        pcl_publisher.publish(pcl_msg)

        publish_tf(br)  # TF 변환을 주기적으로 퍼블리시

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    opt = make_parser().parse_args()
    print(opt)
    with torch.no_grad():
        detect()
