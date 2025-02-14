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

def sliding_window_lane_tracking(binary_warped, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

class LoadImagesFromBag:
    def __init__(self, topic_name, img_size=640, stride=32):
        self.topic_name = topic_name
        self.img_size = img_size
        self.stride = stride
        self.bridge = CvBridge()
        self.image = None
        rospy.init_node('lane_detection_using_yolo_node', anonymous=True)
        self.subscriber = rospy.Subscriber(self.topic_name, Image, self.callback)
        self.rate = rospy.Rate(10)
        with np.load('calibration_result.npz') as data: # 왜곡 보정
            self.mtx = data['mtx']
            self.dist = data['dist']

    def callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV image: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        while self.image is None:
            # rospy.loginfo("Waiting for image...")
            self.rate.sleep()
        try:
            # img_resized = cv2.resize(self.image, (640, 384))
            # current_image = self.image
        
            # h, w = self.image.shape[:2] # 왜곡보정
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
            # x, y, w, h = roi
            # img_undistort = cv2.undistort(self.image, self.mtx, self.dist, self.dist, newcameramtx)
            # img_undistort = img_undistort[y:y+h, x:x+w]
            # img_undistort = cv2.resize(img_undistort, (1280, 720))

            # img_resized = cv2.resize(img_undistort, (640, 360))
            # img_resized = letterbox(img_undistort, self.img_size,stride=self.stride)[0]
            # img_resized = cv2.resize(img_undistort, (640, 360))
            img = cv2.resize(self.image, (1280, 720))
            img = cv2.flip(img, 0)
            img_resized = letterbox(img, self.img_size,stride=self.stride)[0]
            self.image = None
            return None, img_resized, img, None
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV image: {e}")
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
    parser.add_argument('--topic-name', type=str, default='', help='name of the topic to read images from')
    return parser

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)

# Birds Eye View
def transform_to_bev(image, src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return bev_image

# 화면에 점찍기
def draw_points(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # 파란색 점
    return image

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
    half = device.type != 'cpu'  # half precision은 CUDA에서만 지원
    model = model.to(device)

    if half:
        model.half()  # FP16으로 변환
    model.eval()

    dataset = LoadImagesFromBag(opt.topic_name, img_size=imgsz, stride=stride)

    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 한 번 실행
    t0 = time.time()
    # fps 계산
    frame_count = 0
    start_time = time.time()
    # 버드아이뷰
    src_points = np.float32([[-200, 550], [1480, 550], [920, 340], [370,340]])
    dst_points = np.float32([[0, 720], [1280, 720], [1280, 0], [0, 0]])

    for path, img, im0s, vid_cap in dataset:
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
        # da, ll 겹치는 부분 삭제
        # ll_seg_mask_cleaned = np.where(da_seg_mask == 1, 0, ll_seg_mask)
        # ll_seg_mask_cleaned_colored = np.zeros_like(im0s)
        # ll_seg_mask_cleaned_colored [ll_seg_mask_cleaned == 1] = [255,0,0]
        da_seg_mask_colored = np.zeros_like(im0s)
        ll_seg_mask_colored = np.zeros_like(im0s)
        da_seg_mask_colored[da_seg_mask == 1] = [0, 255, 0]  # for driving area
        ll_seg_mask_colored[ll_seg_mask == 1] = [255, 255, 255] # lane mask

        black_lane = np.zeros_like(im0s) # 검은색 배경 생성
        combined_mask = cv2.addWeighted(da_seg_mask_colored, 0.5, ll_seg_mask_colored, 0.5, 0) # 차선, 주행가능지역 둘다
        black_lane = cv2.addWeighted(black_lane, 1, ll_seg_mask_colored, 1, 0)
        for i, det in enumerate(pred):  # 이미지당 디텍션
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # testimg = cv2.addWeighted(im0, 1, ll_seg_mask_cleaned_colored, 1, 0)
            testimg = cv2.addWeighted(im0, 1, ll_seg_mask_colored, 0.5, 0)
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            black_lane_point = draw_points(black_lane.copy(), src_points)
            bev_image = transform_to_bev(testimg, src_points, dst_points)
            img1 = cv2.resize(im0, (640, 360))
            img2 = cv2.resize(testimg, (640, 360))
            img3 = cv2.resize(bev_image, (640, 360))
            img4 = cv2.resize(black_lane_point, (640, 360))
            top_row = np.hstack((img1, img2))
            bottom_row = np.hstack((img3, img4))
            combined_image = np.vstack((top_row, bottom_row))
            cv2.imshow('Combined Image', combined_image)
            cv2.imshow('pointview', black_lane_point)
            cv2.waitKey(1)
            # cv2.imshow('img_undistort', im0)
            # cv2.imshow('onlyLane', testimg)
            # cv2.imshow('bev', bev_image)
            # cv2.imshow('Lane and point',black_lane_point)
            # cv2.waitKey(1)
            # cv2.imshow('1', black_background)

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))

        # 프레임 카운트 증가
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time>1:
            fps = frame_count
            frame_count = 0
            elapsed_time = 0
            start_time = time.time()
            print(f"FPS: {fps:.2f}")
        
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
