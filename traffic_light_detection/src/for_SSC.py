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
zoomed_image_pub = rospy.Publisher('/traffic_light/detected_image_zoom', Image, queue_size=10)  # 확대된 ROI 이미지를 퍼블리시할 토픽
traffic_light_pub = rospy.Publisher('/traffic_light_detect', Bool, queue_size=10)  # /traffic_light_detect 토픽으로 Bool 값 퍼블리시

# 탐지 상태를 저장하기 위한 변수들
last_detected_frame = None
detected_traffic_light = False
detection_count = 0  # 탐지가 유지된 프레임 수 카운터
frame_threshold = 10  # 미탐지를 방지할 프레임 수 임계값

# ROS Image Callback
def image_callback(msg):
    global last_detected_frame, detected_traffic_light, detection_count

    try:
        # Convert ROS CompressedImage message to OpenCV image
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        # ROI 설정 및 확대
        roi_x, roi_y, roi_w, roi_h = 140, 70, 360, 200  # Adjust ROI coordinates as needed
        scale = 3  # 확대 배율
        roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        roi_resized = cv2.resize(roi_frame, (roi_w * scale, roi_h * scale), interpolation=cv2.INTER_LINEAR)

        # hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        #
        # # 밝기 값(V)을 조정하고, uint8로 변환
        # v = np.clip(v * 1.2, 0, 255).astype(np.uint8)
        #
        # # 다시 HSV를 BGR로 변환
        # final_hsv = cv2.merge((h, s, v))
        # roi_resized = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # 1. CLAHE 적용
        # BGR 이미지를 YUV로 변환
        img_yuv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2YUV)

        # Y 채널에 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

        # 다시 YUV를 BGR로 변환
        roi_resized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        #########################################################
        # 화면 중앙 좌표 계산
        frame_center_x = roi_resized.shape[1] // 2
        frame_center_y = roi_resized.shape[0] // 2

        # Draw a red bounding box around the ROI (region of interest)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)

        # 언샤프 마스킹을 이용한 선명도 조정
        gaussian_blurred = cv2.GaussianBlur(roi_resized, (0, 0), sigmaX=2, sigmaY=2)
        sharpened = cv2.addWeighted(roi_resized, 1.5, gaussian_blurred, -0.5, 0)

        # HSV로 변환 후 채도와 밝기 낮추기
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.0, 0, 255)  # 채도 조정
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.1, 0, 255)  # 밝기 조정
        sharpened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # YOLOv8 모델로 예측 수행 (확대된 ROI)
        results = model.predict(sharpened)
        detections = results[0].boxes

        largest_box = None
        largest_area = 0
        min_center_distance = float('inf')
        detected_traffic_light_in_frame = False  # 현재 프레임에서 신호등 탐지 여부 플래그

        # 모든 바운딩 박스에 대해 가장 크고 중앙에 가까운 박스를 찾는 과정
        for i, (box, conf, cls) in enumerate(zip(detections.xyxy.cpu().numpy(), detections.conf.cpu().numpy(), detections.cls.cpu().numpy())):
            if conf > 0.09:
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]

                # "None", "etc", "bus" 클래스를 무시
                if class_name in ["None", "etc", "bus"]:
                    continue

                # 바운딩 박스의 크기 계산
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # 바운딩 박스 중심 좌표 계산
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 화면 중앙과 바운딩 박스 중심 사이의 거리 계산
                center_distance = np.sqrt((center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2)

                # 가장 큰 바운딩 박스 중에서 중앙에 가장 가까운 박스 선택
                if area > largest_area or (area == largest_area and center_distance < min_center_distance):
                    largest_area = area
                    min_center_distance = center_distance
                    largest_box = (x1, y1, x2, y2, class_name, conf)

        # 가장 큰 바운딩 박스가 존재하면 탐지 결과 적용
        if largest_box is not None:
            x1, y1, x2, y2, class_name, conf = largest_box

            # 바운딩 박스 그리기 (ROI에 그려줌)
            cv2.rectangle(roi_resized, (x1, y1), (x2, y2), (23, 230, 210), 2)
            cv2.putText(roi_resized, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Largest and central traffic light detected: {class_name} with area: {largest_area}, confidence: {conf}")

            detected_traffic_light_in_frame = True

            # 탐지 카운트 증가
            if detected_traffic_light_in_frame:
                detection_count += 1
                if detection_count >= frame_threshold:  # 탐지가 일정 시간 유지된 경우
                    detected_traffic_light = True
                    last_detected_frame = class_name  # 탐지된 신호등 상태 저장
            else:
                detection_count = 0  # 탐지되지 않으면 카운트 초기화

            # 색상에 따라 True 또는 False 값 퍼블리시
            if detected_traffic_light:
                if class_name in ["red", "red and yellow", "yellow"]:
                    traffic_light_bool = False
                    rospy.loginfo("Publishing False to /traffic_light_detect")
                    traffic_light_pub.publish(traffic_light_bool)
                elif class_name in ["green", "green arrow", "green and green arrow", "green and yellow"]:
                    traffic_light_bool = True
                    rospy.loginfo("Publishing True to /traffic_light_detect")
                    traffic_light_pub.publish(traffic_light_bool)

        # 탐지가 유지되지 않았을 경우 이전 탐지 상태를 유지
        elif detected_traffic_light and detection_count < frame_threshold:
            rospy.loginfo(f"Maintaining previous detection result: {last_detected_frame}")
            # 이전 탐지 상태에 맞춰 값을 퍼블리시
            if last_detected_frame in ["red", "red and yellow", "yellow"]:
                traffic_light_bool = False
                rospy.loginfo("Publishing False to /traffic_light_detect (maintained)")
                traffic_light_pub.publish(traffic_light_bool)
            elif last_detected_frame in ["green", "green arrow", "green and green arrow", "green and yellow"]:
                traffic_light_bool = True
                rospy.loginfo("Publishing True to /traffic_light_detect (maintained)")
                traffic_light_pub.publish(traffic_light_bool)

        # YOLO가 탐지하지 못한 경우 True 발행
        if not detected_traffic_light_in_frame:
            rospy.loginfo("No traffic light detected, publishing True to /traffic_light_detect")
            traffic_light_pub.publish(True)

        # Convert original image with ROI rectangle (frame) to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
        image_pub.publish(ros_image)

        # Convert zoomed-in ROI image to ROS Image message and publish
        ros_zoomed_image = bridge.cv2_to_imgmsg(roi_resized, "bgr8")
        zoomed_image_pub.publish(ros_zoomed_image)

    except CvBridgeError as e:
        rospy.logerr(f"Could not convert image: {e}")

def main():
    rospy.init_node('traffic_light_detection', anonymous=True)
    #image_topic = "/usb_cam/image_raw/compressed"  # USB 카메라에서 받아오는 이미지 토픽으로 변경
    image_topic = "/zed_node/left/image_rect_color"  # USB 카메라에서 받아오는 이미지 토픽으로 변경
    #rospy.Subscriber(image_topic, CompressedImage, image_callback)
    rospy.Subscriber(image_topic, Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
