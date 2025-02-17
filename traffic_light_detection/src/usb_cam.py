#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('/home/ssc/SSC/src/perception/src/traffic_light_detection/models/traffic_light.pt')  # YOLOv8 모델 경로
bridge = CvBridge()  # CV Bridge for ROS
print("Start")

# 퍼블리셔 생성: RViz에서 이미지를 시각화하기 위해 사용
image_pub = rospy.Publisher('/traffic_light/detected_image', Image, queue_size=10)
traffic_light_pub = rospy.Publisher('/traffic_light_detect', Bool, queue_size=10)  # /traffic_light_detect 토픽으로 Bool 값 퍼블리시

# 신호등 색상 감지 함수
def detect_traffic_light_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_lower1, red_upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
    green_lower, green_upper = np.array([40, 50, 50]), np.array([90, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_area = cv2.countNonZero(red_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    green_area = cv2.countNonZero(green_mask)

    if red_area > yellow_area and red_area > green_area:
        return "Red"
    elif yellow_area > red_area and yellow_area > green_area:
        return "Yellow"
    elif green_area > red_area and green_area > yellow_area:
        return "Green"
    else:
        return "Unknown"

# 신호등 색상에 따른 True/False 발행 함수
def publish_traffic_light_bool(color):
    # (기존 코드와 동일)
    ...

def image_callback(msg):
    try:
        # Convert ROS CompressedImage message to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ROI 설정 및 확대
        roi_x, roi_y, roi_w, roi_h = 200, 0, 1080, 500  # Adjust ROI coordinates as needed
        scale = 1  # 확대 배율
        roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        roi_resized = cv2.resize(roi_frame, (roi_w * scale, roi_h * scale), interpolation=cv2.INTER_LINEAR)

        # 언샤프 마스킹을 이용한 선명도 조정
        gaussian_blurred = cv2.GaussianBlur(roi_resized, (0, 0), sigmaX=2, sigmaY=2)
        sharpened = cv2.addWeighted(roi_resized, 1.5, gaussian_blurred, -0.5, 0)

        # HSV로 변환 후 채도와 밝기 낮추기
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # 채도 조정
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.0, 0, 255)  # 밝기 조정
        sharpened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # YOLOv8 모델로 예측 수행 (확대된 ROI)
        results = model.predict(sharpened)
        detections = results[0].boxes

        # 바운딩 박스, 신뢰도, 클래스 정보 추출
        xyxy = detections.xyxy.cpu().numpy()  # 바운딩 박스 좌표
        confs = detections.conf.cpu().numpy()  # 신뢰도 점수
        classes = detections.cls.cpu().numpy()  # 클래스 번호

        detected_traffic_light = False  # 신호등 탐지 여부 플래그

        for i, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
            if conf > 0.2:
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]

                # "None", "etc", "bus" 클래스를 무시
                if class_name in ["None", "etc", "bus"]:
                    continue

                # 바운딩 박스 그리기
                cv2.rectangle(sharpened, (x1, y1), (x2, y2), (23, 230, 210), 2)
                cv2.putText(sharpened, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                print(class_name)
                if class_name in ["red", "yellow", "green", "red and yellow", "green arrow", "green and green arrow"]:
                    print("Detected traffic light signal:", class_name)
                    detected_traffic_light = True  # 신호등 탐지됨

                    # 색상에 따라 True 또는 False 값 퍼블리시
                    if class_name in ["red", "red and yellow", "yellow"]:
                        traffic_light_bool = False
                        rospy.loginfo("Publishing False to /traffic_light_detect")
                        traffic_light_pub.publish(traffic_light_bool)
                    elif class_name in ["green", "green arrow", "green and green arrow"]:
                        traffic_light_bool = True
                        rospy.loginfo("Publishing True to /traffic_light_detect")
                        traffic_light_pub.publish(traffic_light_bool)

        # YOLO가 탐지하지 못한 경우 True 발행
        if not detected_traffic_light:
            rospy.loginfo("No traffic light detected, publishing True to /traffic_light_detect")
            traffic_light_pub.publish(True)

        # Convert processed image to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(sharpened, "bgr8")

        # Publish the image to the designated topic for RViz
        image_pub.publish(ros_image)

    except CvBridgeError as e:
        rospy.logerr(f"Could not convert image: {e}")

def main():
    rospy.init_node('traffic_light_detection', anonymous=True)
    image_topic = "/usb_cam/image_raw/compressed"  # USB 카메라에서 받아오는 이미지 토픽으로 변경
    rospy.Subscriber(image_topic, CompressedImage, image_callback)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass