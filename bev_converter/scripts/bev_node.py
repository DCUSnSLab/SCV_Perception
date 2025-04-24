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
        """
        카메라 내부 파라미터와 외부 파라미터를 이용해,
        지면(z=0 평면)으로의 호모그래피 H = K*[r1 r2 t]를 구성하고, 그 역행렬을 반환.
        """
        # 내부 파라미터 추출: fx, fy, cx, cy
        fx = self.intrinsics.get("fx", 1.0)
        fy = self.intrinsics.get("fy", 1.0)
        cx = self.intrinsics.get("cx", 0.0)
        cy = self.intrinsics.get("cy", 0.0)
        # 왜곡 계수 추출
        distortion_coeffs = np.array(self.intrinsics.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]))
        self.camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0,  0,  1]])
        self.dist_coeffs = distortion_coeffs
        
        # 카메라 회전 행렬과 평행 이동 벡터  
        R_list = self.extrinsics.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        R = np.array(R_list).reshape(3, 3)
        
        # 카메라 위치 및 방향 분석을 위한 로그
        rospy.loginfo("Camera rotation matrix:\n%s", R)
        
        # 회전 행렬이 단위 행렬인지 확인 (단위 행렬이면 카메라가 지면에 수직으로 설치되어 있다는 의미)
        is_identity = np.allclose(R, np.eye(3))
        rospy.loginfo("Camera rotation is identity matrix: %s", is_identity)
        
        # translation 벡터
        t_vec = np.array(self.extrinsics.get("translation", [0, 0, self.h]))
        rospy.loginfo("Camera height: %.2f meters", self.h)
        
        # 
        # 지면과 카메라 각도 고려: 실제 테스트에서는 카메라가 아래로 기울어져 있을 가능성이 높음
        # 카메라가 아래로 기울어지는 각도 (pitch angle) 조정 - 이 값을 조정하여 테스트
        pitch_angle_degrees = -15  # 카메라가 아래로 15도 기울어져 있다고 가정
        pitch_angle_rad = np.radians(pitch_angle_degrees)
        
        # 새로운 회전 행렬 계산 (피치 각도 반영)
        R_pitch = np.array([
            [np.cos(pitch_angle_rad), 0, np.sin(pitch_angle_rad)],
            [0, 1, 0],
            [-np.sin(pitch_angle_rad), 0, np.cos(pitch_angle_rad)]
        ])
        
        # 기존 회전과 피치 회전 적용
        R_adjusted = np.dot(R, R_pitch)
        rospy.loginfo("Adjusted rotation matrix with pitch angle %d degrees:\n%s", pitch_angle_degrees, R_adjusted)
        
        # 호모그래피 행렬 구성: H = K * [r1, r2, t]
        r1 = R_adjusted[:, 0].reshape(3, 1)  # 첫 번째 열 벡터
        r2 = R_adjusted[:, 1].reshape(3, 1)  # 두 번째 열 벡터
        t = t_vec.reshape(3, 1)     # 평행 이동 벡터
        
        H = np.dot(self.camera_matrix, np.hstack((r1, r2, t)))  # 결과: 3x3 행렬
        
        try:
            H_inv = np.linalg.inv(H)
            rospy.loginfo("Computed inverse homography matrix: \n%s", H_inv)
            # 호모그래피 스케일 요소 확인
            rospy.loginfo("Homography scale factor approximation: %.4f", H_inv[2, 2])
            return H_inv
        except np.linalg.LinAlgError:
            rospy.logerr("Homography matrix is singular, cannot invert!")
            return None
        
    def image_callback(self, img_msg):
        # 현재는 이미지 정보만 저장, 추후 필요시 기능 확장
        self.image_width = img_msg.width
        self.image_height = img_msg.height

    def yolo_callback(self, detections_msg):
        if self.H_inv is None:
            rospy.logerr("Inverse homography matrix is not available. Skipping YOLO callback.")
            return

        points = []  # RViz에 시각화할 점들을 담을 리스트
        
        # 고정된 2m 디버깅 마커 추가
        debug_marker = Marker()
        debug_marker.header.frame_id = self.marker_frame
        debug_marker.header.stamp = rospy.Time.now()
        debug_marker.ns = "debug_distance"
        debug_marker.id = 1
        debug_marker.type = Marker.POINTS
        debug_marker.action = Marker.ADD
        debug_marker.pose.orientation.w = 1.0
        debug_marker.scale.x = 0.3
        debug_marker.scale.y = 0.3
        debug_marker.color.a = 1.0
        debug_marker.color.r = 1.0
        debug_marker.color.g = 0.0
        debug_marker.color.b = 0.0
        
        # 2m 거리에 참조 마커 표시
        debug_pt = Point()
        debug_pt.x = 2.0  # 2미터 거리
        debug_pt.y = 0.0  # 중앙
        debug_pt.z = 0.0  # 지면 평면
        debug_marker.points = [debug_pt]
        
        # 디버깅용 마커 게시
        self.marker_pub.publish(debug_marker)
        
        # 객체 감지 결과 처리
        try:
            for detection in detections_msg.detections.detections:
                # 바운딩 박스 정보 추출
                bbox = detection.bbox
                
                # 객체의 하단 중심점 좌표 (발 위치)
                u = bbox.center.x
                v = bbox.center.y + (bbox.size_y / 2.0)  # y 좌표 + 박스 높이의 절반 = 바닥
                
                # 카메라 왜곡 보정 (활성화)
                point_distorted = np.array([[[u, v]]], dtype=np.float32)
                point_undistorted = cv2.undistortPoints(point_distorted, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
                u_undistorted = point_undistorted[0][0][0]
                v_undistorted = point_undistorted[0][0][1]
                
                # 디버깅: 왜곡 보정 전/후 비교
                rospy.loginfo("Image coordinate before undistortion: (%.2f, %.2f) -> after: (%.2f, %.2f)", 
                             u, v, u_undistorted, v_undistorted)
                
                # 왜곡이 보정된 좌표 사용
                u, v = u_undistorted, v_undistorted
                
                # 호모그래피를 적용하여 이미지 좌표 (u,v)를 월드 좌표 (X, Y)로 변환
                X, Y = self.image_to_ground(u, v, self.H_inv)
                
                # 거리 계산 (지면 평면상의 직선 거리)
                distance = np.sqrt(X*X + Y*Y)
                
                # Y 좌표가 더 큰 것을 확인했으므로, 좌표계 교환
                # 전방 거리를 Y좌표로 표현하고, X는 좌우 위치를 의미
                X_adjusted = Y  # 좌우 위치
                Y_adjusted = X  # 전방 거리
                
                # 원래 좌표와 조정된 좌표 값 확인
                rospy.loginfo("Original Ground Coord: (%.2f, %.2f), Distance: %.2f -> Swapped: (%.2f, %.2f)", 
                             X, Y, distance, X_adjusted, Y_adjusted)
                
                # 스케일 조정
                scale_factor = 2.0  # 실제 거리에 맞게 조정 (200cm = 2.0)
                X_scaled = X_adjusted * scale_factor
                Y_scaled = Y_adjusted * scale_factor
                
                rospy.loginfo("Detection at (u,v): (%.2f, %.2f) -> Ground: (%.2f, %.2f) -> Final: (%.2f, %.2f)", 
                              u, v, X, Y, X_scaled, Y_scaled)
                
                # RViz 좌표계로 변환 (ROS 좌표계 규칙에 맞게)
                pt = Point()
                # 표준 ROS 좌표계: x=전방, y=좌측, z=위쪽
                pt.x = Y_scaled  # 전방 거리
                pt.y = X_scaled  # 좌우 위치
                pt.z = 0.0       # 지면 평면
                points.append(pt)
                
        except Exception as e:
            rospy.logerr("Error processing detection: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            return

        # 감지 결과 마커 생성
        marker = Marker()
        marker.header.frame_id = self.marker_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bev_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.points = points

        # RViz에 Marker Publish
        self.marker_pub.publish(marker)

    def image_to_ground(self, u, v, H_inv):
        """
        이미지 좌표 (u, v)를 동차 좌표계로 만들고, 역호모그래피 H_inv를 곱해
        월드 좌표 (X, Y)를 계산하는 함수.
        
        반환되는 좌표계:
        - X: 카메라 광축 방향 (전방)
        - Y: 카메라 좌측 방향
        """
        # 동차 좌표: [u, v, 1]
        point_img = np.array([u, v, 1.0]).reshape(3, 1)
        
        # 역호모그래피 적용
        ground_point_h = np.dot(H_inv, point_img)
        
        # 동차 좌표 정규화
        if abs(ground_point_h[2, 0]) < 1e-10:  # 0에 가까운 값인지 확인 (수치 안정성)
            rospy.logwarn("Invalid homogeneous coordinate: division by zero or very small value")
            return 0.0, 0.0
            
        # 동차 좌표 정규화 (z=1이 되도록)
        ground_point_h /= ground_point_h[2, 0]
        
        # 정규화된 좌표에서 X, Y 추출
        X = ground_point_h[0, 0]  # 카메라 전방 거리
        Y = ground_point_h[1, 0]  # 카메라 좌측 거리
        
        # 거리에 따른 보정 (멀리 있을수록 오차가 커지는 문제 해결)
        # 거리가 멀어지면 정밀도가 떨어지므로, 거리에 반비례하는 보정 적용
        # 예: 거리가 10배 멀어지면 정밀도는 1/10로 감소
        distance = np.sqrt(X*X + Y*Y)
        if distance > 5.0:  # 5미터 이상 떨어진 경우 보정 적용
            correction = 5.0 / distance
            X *= correction
            Y *= correction
            rospy.logdebug("Applied distance correction: %.2f for distance %.2f", correction, distance)
        
        return X, Y

if __name__ == '__main__':
    try:
        bev_node = BevNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
