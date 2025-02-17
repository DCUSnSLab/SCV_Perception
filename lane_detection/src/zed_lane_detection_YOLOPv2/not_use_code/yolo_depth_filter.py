import argparse
import time
from pathlib import Path
import cv2
import torch
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils.utils import time_synchronized, select_device, increment_path, scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, AverageMeter,letterbox
import signal
import sys

class LoadImagesFromBag:
    def __init__(self, topic_name, depth_topic_name, img_size=640, stride=32):
        self.topic_name = topic_name
        self.depth_topic_name = depth_topic_name
        self.img_size = img_size
        self.stride = stride
        self.bridge = CvBridge()
        self.image = None
        self.depth_image = None
        rospy.init_node('zed_camera_lane_detection_using_yolo_node', anonymous=True)
        self.subscriber = rospy.Subscriber(self.topic_name, Image, self.callback)
        self.depth_subscriber = rospy.Subscriber(self.depth_topic_name, Image, self.depth_callback)
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

    def __iter__(self):
        return self

    def __next__(self):
        while self.image is None or self.depth_image is None:
            self.rate.sleep()
        try:
            img = cv2.resize(self.image, (1280, 720))
            img_resized = letterbox(img, self.img_size, stride=self.stride)[0]
            depth_resized = self.depth_image
            print(self.depth_image.shape)
            self.image = None
            self.depth_image = None
            return None, img_resized, img, depth_resized
        except Exception as e:
            print(f"An error occurred: {e}")
            raise StopIteration

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--topic-name', type=str, default='/zed2i/zed_node/left/image_rect_color', help='name of the topic to read images from')
    parser.add_argument('--depth-topic-name', type=str, default='/zed2i/zed_node/depth/depth_registered', help='name of the depth topic to read images from')
    parser.add_argument('--output-topic', type=str, default='/masked_depth_output', help='name of the output topic to publish masked depth images')
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
    # 설정 및 디렉토리 생성
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)

    if half:
        model.half()  # FP16으로 변환
    model.eval()

    dataset = LoadImagesFromBag(opt.topic_name, opt.depth_topic_name, img_size=imgsz, stride=stride)

    # ROS 퍼블리셔 설정
    depth_pub = rospy.Publisher(opt.output_topic, Image, queue_size=10)

    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 한 번 실행
    t0 = time.time()

    for path, img, im0s, depth_img in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()
        # 추가 시간 소모
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # NMS 적용
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()
        
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # 마스크를 원본 깊이 이미지 크기로 리사이즈
        ll_seg_mask_resized = cv2.resize(ll_seg_mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = ll_seg_mask_resized[:, :] * 255  # 마스크를 0과 255로 스케일링
        masked_depth = apply_mask_to_depth(depth_img, mask)

        try:
            masked_depth_msg = dataset.bridge.cv2_to_imgmsg(masked_depth, encoding="32FC1")
            depth_pub.publish(masked_depth_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image to ROS Image message: {e}")

        for i, det in enumerate(pred):  # 이미지당 디텍션
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            cv2.waitKey(1)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
