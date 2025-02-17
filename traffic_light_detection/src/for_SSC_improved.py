#!/usr/bin/env python3
from enum import Enum
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import os

# YOLOv8 모델 로드
model = YOLO('/home/ssc/SSC/src/perception/src/traffic_light_detection/models/traffic_light.pt')  # YOLOv8 모델 경로
bridge = CvBridge()  # CV Bridge for ROS
print("Start")

# 퍼블리셔 생성: RViz에서 이미지를 시각화하기 위해 사용
image_pub = rospy.Publisher('/traffic_light/detected_image', Image, queue_size=10)
zoomed_image_pub = rospy.Publisher('/traffic_light/detected_image_zoom', Image, queue_size=10)  # 확대된 ROI 이미지를 퍼블리시할 토픽
traffic_light_pub = rospy.Publisher('/traffic_light_detect', Int32MultiArray, queue_size=10)  # /traffic_light_detect 토픽으로 Bool 값 퍼블리시

# 탐지 상태를 저장하기 위한 변수들
hztosec = 14
invisible_cnt = 0
prev_signal = None

last_image_time = None

# 탐지 기록을 위한 리스트 및 카운터 딕셔너리
detected_classes = []  # 탐지된 클래스 이름을 순서대로 저장
class_counts = {"red": 0, "yellow": 0, "green": 0, "red and green arrow": 0, "green arrow": 0, "none": 0}

T_SIGNAL = {'red': 0, 'yellow': 0, 'green': 1, 'red and green arrow': 1, 'green arrow': 1, 'none': -1}
class TRAFFIC_SIG(Enum):
    RED = (0, 0, 'red')
    YELLOW = (0, 1, 'yellow')
    RED_AND_YELLOW = (0, 3, 'red and yellow')
    GREEN = (1, 2, 'green')
    RED_AND_GREEN_ARROW = (1, 3, 'red and green arrow')
    GREEN_ARROW = (1, 4, 'green arrow')
    GREEN_AND_GREEN_ARROW = (1, 5, 'green and green arrow')
    YELLOW_AND_GREEN_ARROW = (1, 6, 'yellow and green arrow')
    GREEN_AND_YELLOW = (1, 7, 'green and yellow')
    GREEN_ARROW_AND_GREEN_ARROW = (1, 8, 'green arrow and green arrow')
    GREEN_ARROW_DOWN = (1, 9, 'green_arrow(down)')
    NONE = (-1, 6, 'none')

    @classmethod
    def from_sig_class(cls, value):
        for signal in cls:
            if signal.value[2] == value:
                return signal
        return cls.NONE  # 해당 값이 없으면 None 반환

    @classmethod
    def from_sig_num(cls, value):
        for signal in cls:
            if signal.value[1] == value:
                return signal
        return cls.NONE  # 해당 값이 없으면 None 반환

# 탐지 기록을 txt 파일에 저장하는 함수
def save_detection_log(filename="detection_log.txt"):
    with open(filename, "w") as f:
        # 탐지된 순서대로 클래스 기록
        f.write("Detected classes in order:\n")
        for cls in detected_classes:
            f.write(f"{cls}\n")

        # 각 클래스당 탐지 횟수 기록
        f.write("\nClass detection counts:\n")
        for cls, count in class_counts.items():
            f.write(f"{cls}: {count}\n")

    print(f"Detection log saved to {filename}")

# ROS Image Callback
def image_callback(msg):
    global invisible_cnt, prev_signal, hztosec, last_image_time

    last_image_time = rospy.Time.now()  # 이미지 콜백이 호출될 때마다 시간 기록

    try:
        # Convert ROS CompressedImage message to OpenCV image
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        # ROI 설정 및 확대
        roi_x, roi_y, roi_w, roi_h = 200, 0, 880, 650  # Adjust ROI coordinates as needed 720p
        #roi_x, roi_y, roi_w, roi_h = 100, 0, 440, 350  # Adjust ROI coordinates as needed
        #roi_x, roi_y, roi_w, roi_h = 200, 0, 1420, 800
        scale = 1  # 확대 배율
        roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        roi_resized = cv2.resize(roi_frame, (roi_w * scale, roi_h * scale), interpolation=cv2.INTER_LINEAR)

        # CLAHE 적용
        img_yuv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        roi_resized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # 화면 중앙 좌표 계산
        frame_center_x = roi_resized.shape[1] // 2
        frame_center_y = roi_resized.shape[0] // 2

        # Draw a red bounding box around the ROI (region of interest)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)

        # YOLOv8 모델로 예측 수행 (확대된 ROI)
        results = model.predict(roi_resized)
        detections = results[0].boxes

        largest_box = None
        largest_area = 0
        min_center_distance = float('inf')

        # 탐지 결과 중 가장 큰 신호등 바운딩 박스 선택
        for i, (box, conf, cls) in enumerate(zip(detections.xyxy.cpu().numpy(), detections.conf.cpu().numpy(), detections.cls.cpu().numpy())):
            if conf > 0.35:
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]

                if class_name in ["None", "etc", "bus"]:
                    continue

                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_distance = np.sqrt((center_x - frame.shape[1] // 2) ** 2 + (center_y - frame.shape[0] // 2) ** 2)

                if area > largest_area or (area == largest_area and center_distance < min_center_distance):
                    largest_area = area
                    min_center_distance = center_distance
                    largest_box = (x1, y1, x2, y2, class_name, conf)

        if largest_box is not None:
            x1, y1, x2, y2, class_name, conf = largest_box
            prev_signal = largest_box
            invisible_cnt = 0

            x1, y1, x2, y2, class_name, conf = prev_signal
            cls_text = class_name + ':' + str(round(conf, 2))
            # 바운딩 박스 그리기 (ROI에 그려줌)
            cv2.rectangle(roi_resized, (x1, y1), (x2, y2), (23, 230, 210), 2)
            cv2.putText(roi_resized, cls_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(
                f"Largest and central traffic light detected: {class_name} with area: {largest_area}, confidence: {conf}")
        else:
            invisible_cnt += 1

        #convert hz to sec
        inv_sec = invisible_cnt / hztosec
        if inv_sec > 10:
            prev_signal = None
        if prev_signal is not None:
            print(f'signal : {prev_signal[4]}')
            print(f'signal : {TRAFFIC_SIG.from_sig_class(prev_signal[4])}')
        direction = TRAFFIC_SIG.NONE if prev_signal is None else TRAFFIC_SIG.from_sig_class(prev_signal[4])
        dirarray = Int32MultiArray()
        print('direction : ', direction)
        dirarray.data = [direction.value[0], direction.value[1]]
        traffic_light_pub.publish(dirarray)
        print(f'current sig data : {TRAFFIC_SIG.from_sig_num(dirarray.data[1]).name}')

        # 차량 상태 및 신호등 상태 텍스트 표시
        vehicle_status = "None" if direction.value[0] == -1 else "Stop" if direction.value[0] == 0 else "Go"
        traffic_light_signal = TRAFFIC_SIG.from_sig_num(dirarray.data[1]).name

        # 화면에 텍스트로 표시
        cv2.putText(roi_resized, f"Vehicle Status: {vehicle_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(roi_resized, f"Traffic Light Signal: {traffic_light_signal}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # 확대된 ROI 이미지를 ROS 이미지로 변환 및 퍼블리시
        zoomed_img_msg = bridge.cv2_to_imgmsg(roi_resized, encoding="bgr8")
        zoomed_image_pub.publish(zoomed_img_msg)

        # 결과 이미지를 ROS 이미지로 변환하여 퍼블리시
        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_pub.publish(img_msg)

    except CvBridgeError as e:
        print(e)

# Add a timer callback to check if the image callback has not been called for a while

def check_camera_timeout(event):
    global last_image_time

    current_time = rospy.Time.now()

    if last_image_time is None:
        return 

    time_diff = current_time - last_image_time

    if time_diff.to_sec() > 1.0:
        dirarray = Int32MultiArray()
        dirarray.data = [1, 6] 
        traffic_light_pub.publish(dirarray)

def main():
    global last_image_time
    
    # rospy.init_node('image_listener', anonymous=True)
    rospy.init_node('traffic_light_detection', anonymous=True)

    image_topic = "/zed_node/left/image_rect_color"  # USB 카메라에서 받아오는 이미지 토픽으로 변경
    
    # Subscribe to the image topic (adjust topic name as necessary)
    # rospy.Subscriber("/camera_topic", Image, image_callback)
    rospy.Subscriber(image_topic, Image, image_callback)

    rospy.Timer(rospy.Duration(1.0 / 15), check_camera_timeout)

    # Create a timer that checks for camera timeout every 5 seconds
    #rospy.Timer(rospy.Duration(0.05), check_camera_timeout)

    # Start the ROS event loop
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
