#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
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

        self.marker_frame = rospy.get_param("~marker_frame", "front_box")

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
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        
        # 외부 파라미터: 회전 행렬과 평행 이동 벡터  
        # extrinsics["rotation"]는 9개 요소의 리스트라고 가정 (3x3 행렬, row-major order)
        R_list = self.extrinsics.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        R = np.array(R_list).reshape(3, 3)
        # translation 벡터
        t_vec = np.array(self.extrinsics.get("translation", [0, 0, self.h]))
        
        # 호모그래피 행렬 구성: H = K * [r1, r2, t]
        r1 = R[:, 0].reshape(3, 1)
        r2 = R[:, 1].reshape(3, 1)
        t = t_vec.reshape(3, 1)
        H = np.dot(K, np.hstack((r1, r2, t)))  # 결과: 3x3 행렬
        
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            rospy.logerr("Homography matrix is singular, cannot invert!")
            H_inv = None
        rospy.loginfo("Computed inverse homography matrix: \n%s", H_inv)
        return H_inv
        
    def image_callback(self, img_msg):
        image_width = img_msg.width
        image_height = img_msg.height

    def yolo_callback(self, detections_msg):
        if self.H_inv is None:
            rospy.logerr("Inverse homography matrix is not available. Skipping YOLO callback.")
            return

        points = []  # RViz에 시각화할 점들을 담을 리스트
        
        for detection in detections_msg.detections.detections:
            # 임시로 person 클래스만 처리: detection.results에 저장된 id가 0인 경우만 사용
            if not detection.results:
                continue
            if detection.results[0].id != 0:
                continue

            bbox = detection.bbox
            u = bbox.center.x
            v = bbox.center.y + (bbox.size_y / 2.0)
            # 호모그래피를 적용하여 이미지 좌표 (u,v)를 월드 좌표 (X, Y)로 변환
            X, Y = self.image_to_ground(u, v, self.H_inv)
            rospy.loginfo("Person detection bottom (u,v): (%.2f, %.2f) -> Ground (X,Y): (%.2f, %.2f)", u, v, X, Y)
            # RViz에 추가할 점
            pt = Point()
            pt.x = X
            pt.y = Y
            pt.z = 0.0  # 지면 평면 상으로 가정
            points.append(pt)

        # Marker 메시지 생성 (여러 점을 POINTS 형태로 시각화)
        marker = Marker()
        marker.header.frame_id = self.marker_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bev_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        # 점의 크기 설정
        marker.scale.x = 0.2
        marker.scale.y = 0.2

        # 색상 설정 (예: 초록색)
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
        """
        # 동차 좌표: [u, v, 1]
        point_img = np.array([u, v, 1.0]).reshape(3, 1)
        # 역호모그래피 적용
        ground_point_h = np.dot(H_inv, point_img)
        # 동차 좌표 정규화
        if ground_point_h[2, 0] == 0:
            rospy.logwarn("Invalid homogeneous coordinate: division by zero")
            return 0.0, 0.0
        ground_point_h /= ground_point_h[2, 0]
        X = ground_point_h[0, 0]
        Y = ground_point_h[1, 0]
        return X, Y

if __name__ == '__main__':
    try:
        bev_node = BevNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
