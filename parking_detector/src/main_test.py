#!/usr/bin/env python3
from collections import deque, Counter
import cv2
import time
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from command_center_interfaces.msg import MultipleWaypoints

T_CONFIDENCE_THRESHOLD = 18 # blue percentage threshold for T area
P_CONFIDENCE_THRESHOLD = 10 # blue percentage threshold for P area

class ParkingAreaDetect(Node):
    def __init__(self):
        super().__init__('parking_area_detect')
        
        self.bridge = CvBridge()
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        try:
            self.model = YOLO('/home/ssc/SCV/src/perception/parking_detector/model/yellow_box/last.pt')
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            self.model = None
        
        self.zed_img_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self._img_callback, 10)
        #self.img_sub = self.create_subscription(Image, '/image_raw', self._img_callback, 10)
        self.waypoints_sub = self.create_subscription(MultipleWaypoints, '/multiple_waypoints', self._waypoints_callback, self.reliable_qos)

        self.tp_area_pub = self.create_publisher(Bool, '/path_availability', 10)
        self.img_pub = self.create_publisher(Image, '/parking/img', 10)

        self.section = 'T'
        self.decisions = []  # 1초 동안 수집할 리스트
        self.start_time = None
                
    def _img_callback(self, msg):
        if self.model is None or self.section is None:
            return
        
        # 1초 지났으면 발행하고 리셋
        if self.start_time and time.time() - self.start_time > 1.0:
            if self.decisions:
                # 가장 많이 나온 결정 발행
                counter = Counter(self.decisions)
                final_decision = counter.most_common(1)[0][0]

                is_a_parking_area_available = final_decision in ['T_A', 'P_A']

                pub_msg = Bool()
                pub_msg.data = is_a_parking_area_available
                self.tp_area_pub.publish(pub_msg)
                
                self.get_logger().info(f'Published: {final_decision} from {dict(counter)}')
            
            # 리셋
            self.section = None
            self.start_time = None
            self.decisions = []
            return
        
        # 이미지 처리
        img = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        results = self.model(img, verbose=False)
        img_origin = img.copy()
        
        all_detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[class_id]
                    area = (x2 - x1) * (y2 - y1)
                    
                    all_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'confidence': conf,
                        'label': label,
                        'area': area
                    })

        if all_detections:
            largest_detection = max(all_detections, key=lambda x: x['area'])
            
            x1, y1, x2, y2 = largest_detection['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if self.section == 'T':
                t_roi_xyxy = self._get_t_roi(largest_detection)
                color_percentages, _ = self._analyze_hsv_colors(img_origin, t_roi_xyxy, "T")
                cv2.rectangle(img, (t_roi_xyxy[0], t_roi_xyxy[1]), (t_roi_xyxy[2], t_roi_xyxy[3]), (255, 0, 0), 2)

                if color_percentages:
                    b = color_percentages[0]
                    self.decisions.append('T_B' if b > T_CONFIDENCE_THRESHOLD else 'T_A')

            elif self.section == 'P':
                p_roi_xyxy = self._get_p_roi(largest_detection)
                color_percentages, _ = self._analyze_hsv_colors(img_origin, p_roi_xyxy, "P")
                cv2.rectangle(img, (p_roi_xyxy[0], p_roi_xyxy[1]), (p_roi_xyxy[2], p_roi_xyxy[3]), (0, 0, 255), 2)

                if color_percentages:
                    b = color_percentages[0]
                    self.decisions.append('P_B' if b > P_CONFIDENCE_THRESHOLD else 'P_A')
                        
        img_msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        self.img_pub.publish(img_msg)

    def _analyze_hsv_colors(self, img, roi_xyxy, section_name):
        x1, y1, x2, y2 = roi_xyxy
        
        # 이미지 경계 체크
        img_height, img_width = img.shape[:2]
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        if x2 <= x1 or y2 <= y1:
            print(f"{section_name} ROI: Invalid ROI coordinates")
            return (0, 0, 0, 0, 0), 'Other'
        
        roi = img[y1:y2, x1:x2]
        
        # HSV로 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 파란색 (100-130)
        blue_mask = cv2.inRange(hsv_roi, np.array([100, 50, 50]), np.array([130, 255, 255]))
        
        # 빨간색 (0-10, 170-180)
        red_mask1 = cv2.inRange(hsv_roi, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv_roi, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 주황색 (10-25)
        orange_mask = cv2.inRange(hsv_roi, np.array([10, 50, 50]), np.array([24, 255, 255]))
        
        # 노란색 (25-35) - 추가
        yellow_mask = cv2.inRange(hsv_roi, np.array([25, 50, 50]), np.array([35, 255, 255]))
        
        # 녹색 (35-85) - 추가
        green_mask = cv2.inRange(hsv_roi, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])

        white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
        
        # 픽셀 수 계산
        blue_count = np.sum(blue_mask > 0)
        red_count = np.sum(red_mask > 0)
        orange_count = np.sum(orange_mask > 0)
        yellow_count = np.sum(yellow_mask > 0)
        green_count = np.sum(green_mask > 0)
        white_count = np.sum(white_mask > 0)
        
        # 전체 픽셀 수
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        
        # if total_pixels == 0:
        #     print(f"{section_name} ROI: No pixels to analyze")
        #     return
        
        # 백분율 계산
        blue_percent = (blue_count / total_pixels) * 100
        red_percent = (red_count / total_pixels) * 100
        orange_percent = (orange_count / total_pixels) * 100
        yellow_percent = (yellow_count / total_pixels) * 100
        green_percent = (green_count / total_pixels) * 100
        white_percent = (white_count / total_pixels) * 100
        
        # 기타 색상 (위 색상들에 해당하지 않는 픽셀)
        colored_pixels = blue_count + red_count + orange_count + yellow_count + green_count + white_count
        other_percent = ((total_pixels - colored_pixels) / total_pixels) * 100
        
        # 지배적인 색상 찾기
        color_percentages = {
            'Blue': blue_percent,
            'Red': red_percent,
            'Orange': orange_percent,
            'Yellow': yellow_percent,
            'Green': green_percent,
            'White': white_percent,
            'Other': other_percent
        }
        dominant_color = max(color_percentages, key=color_percentages.get)
        
        print(f"=== {section_name} ROI HSV Color Analysis ===")
        print(f"ROI Size: {x2-x1} x {y2-y1} ({total_pixels} pixels)")
        print(f"Blue:   {blue_percent:.1f}% ({blue_count} pixels)")
        print(f"Red:    {red_percent:.1f}% ({red_count} pixels)")
        print(f"Orange: {orange_percent:.1f}% ({orange_count} pixels)")
        print(f"Yellow: {yellow_percent:.1f}% ({yellow_count} pixels)")
        print(f"Green:  {green_percent:.1f}% ({green_count} pixels)")
        print(f"White:  {white_percent:.1f}% ({white_count} pixels)")
        print(f"Other:  {other_percent:.1f}% ({total_pixels - colored_pixels} pixels)")
        print(f"Dominant Color: {dominant_color} ({color_percentages[dominant_color]:.1f}%)")
        print("-" * 40)

        return (blue_percent, red_percent, orange_percent, yellow_percent, green_percent, white_percent), dominant_color
        
    def _get_t_roi(self, detection):
        x1, y1, x2, y2 = detection['bbox']

        alpha = x2 - x1
        beta = y2 - y1
        
        roi_x1 = x1 - (alpha // 2)
        roi_y1 = y1 + (beta // 3)
        roi_x2 = x2
        roi_y2 = y2
        
        return roi_x1, roi_y1, roi_x2, roi_y2
            
    def _get_p_roi(self, detection):
        x1, y1, x2, y2 = detection['bbox']

        alpha = x2 - x1
        beta = y2 - y1

        roi_x1 = x2
        roi_y1 = y1 + (beta // 2)
        roi_x2 = x2 + (alpha * 5)
        roi_y2 = y2

        return roi_x1, roi_y1, roi_x2, roi_y2

    def _waypoints_callback(self, msg):
        print(msg.current_goal_node_type)
        next_node_type = msg.current_goal_node_type
        
        # 새로운 섹션 감지시 시작
        if next_node_type == 12 and self.section != 'T':
            self.section = 'T'
            self.start_time = time.time()
            self.decisions = []
            self.get_logger().info('T section detected')
            
        elif next_node_type == 13 and self.section != 'P':
            self.section = 'P'
            self.start_time = time.time()
            self.decisions = []
            self.get_logger().info('P section detected')
            
        
            
def main(args=None):
    rclpy.init(args=args)
    parking_detect = ParkingAreaDetect()

    try:
        rclpy.spin(parking_detect)
    except KeyboardInterrupt:
        pass
    parking_detect.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()