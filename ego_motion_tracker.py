#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import math

# =============================================================================
# 1. KalmanBoxTracker: 개별 객체 추적 클래스
# =============================================================================
class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D):
        """
        bbox3D: [x, y, z, theta, l, w, h]
        State: [x, y, z, theta, l, w, h, vx, vy, vz] (10차원)
        """
        # Kalman Filter 초기화
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.eye(10) # 상태 전이 행렬
        self.kf.H = np.eye(7, 10) # 관측 행렬

        # 초기 상태 설정
        self.kf.x[:7] = bbox3D.reshape((7, 1))
        
        # 불확실성(P) 및 노이즈(Q, R) 튜닝
        self.kf.P[7:, 7:] *= 1000.0 # 초기 속도 불확실성 높게
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.R[0:3, 0:3] *= 0.1 # 위치 측정 노이즈
        self.kf.R[3, 3] *= 0.1     # 각도 측정 노이즈

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update_ego_motion(self, rot_mat, trans_vec, dt_theta):
        """
        [핵심] 연구원님의 아이디어: 로봇 이동량만큼 상태 벡터 보정
        """
        # 1. 위치 보정 (x, y, z)
        # Pos_new = R * Pos_old + T
        self.kf.x[:3] = np.dot(rot_mat, self.kf.x[:3]) + trans_vec.reshape((3, 1))
        
        # 2. 속도 보정 (vx, vy, vz)
        # Vel_new = R * Vel_old (회전만 영향 받음)
        self.kf.x[7:10] = np.dot(rot_mat, self.kf.x[7:10])
        
        # 3. 헤딩 보정 (theta)
        self.kf.x[3] += dt_theta

    def predict(self):
        """
        칼만 필터 예측 단계 (등속 운동 모델)
        """
        # 각도 정규화 (-pi ~ pi)
        self.kf.x[3] = (self.kf.x[3] + np.pi) % (2 * np.pi) - np.pi
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.kf.x

    def update(self, bbox3D):
        """
        관측 업데이트 단계
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # 각도 차이 보정 (350도와 10도 차이가 340도가 되지 않게)
        res = bbox3D[3] - self.kf.x[3]
        if abs(res) > np.pi:
            bbox3D[3] -= np.sign(res) * 2 * np.pi
            
        self.kf.update(bbox3D)

    def get_state(self):
        """현재 추정된 상태 반환 [x, y, z, theta, l, w, h, vx, vy, vz]"""
        return self.kf.x.flatten()

# =============================================================================
# 2. EgoMotionTracker: 전체 트래킹 매니저
# =============================================================================
class EgoMotionTracker(object):
    def __init__(self, max_age=5, min_hits=1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        
    def track(self, detections, odom_twist, dt):
        """
        detections: N x 7 numpy array
        odom_twist: {'vx': float, 'wz': float}
        dt: float (시간 경과)
        """
        self.frame_count += 1
        
        # --- Step 1: Ego-Motion Compensation ---
        # 로봇이 dt 동안 움직인 양 계산
        vx = odom_twist['vx']
        wz = odom_twist['wz']
        
        # 로봇 회전의 반대 방향 행렬 (Active -> Passive Transform)
        cos_a = np.cos(-wz * dt)
        sin_a = np.sin(-wz * dt)
        rot_mat = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        
        # 로봇 이동의 반대 방향 벡터
        trans_vec = np.array([-vx * dt, 0, 0])
        
        # 모든 트랙 보정
        for trk in self.trackers:
            trk.update_ego_motion(rot_mat, trans_vec, -wz * dt)

        # --- Step 2: Prediction ---
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[:7].flatten() # 예측된 [x,y,z,a,l,w,h]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        for t in reversed(to_del):
            self.trackers.pop(t)

        # --- Step 3: Association (Hungarian) ---
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)

        # --- Step 4: Update ---
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(detections[d, :][0])

        # 신규 트랙 생성
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :])
            self.trackers.append(trk)

        # --- Step 5: Life Cycle ---
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i - 1)
            i -= 1

        # 결과 반환 (State + ID)
        ret = []
        for trk in self.trackers:
            # 최소 히트 수 이상이고, 최근에 업데이트 된 것만 출력
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):
                d = np.concatenate((trk.get_state(), [trk.id])) 
                ret.append(d)
        
        if len(ret) > 0: return np.array(ret)
        return np.empty((0, 11))

    def associate(self, detections, trackers):
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # 거리 행렬 (Euclidean Distance)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                dist = np.linalg.norm(det[:3] - trk[:3]) # 중심점 거리
                iou_matrix[d, t] = dist

        # 헝가리안 매칭
        row_ind, col_ind = linear_sum_assignment(iou_matrix)
        
        dist_thresh = 2.0 # 2m 이상이면 다른 물체
        matched_indices = []
        unmatched_dets = []
        unmatched_trks = []
        
        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] < dist_thresh:
                matched_indices.append([d, t])
            else:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
        
        for d in range(len(detections)):
            if d not in row_ind: unmatched_dets.append(d)
        for t in range(len(trackers)):
            if t not in col_ind: unmatched_trks.append(t)

        return np.array(matched_indices), np.array(unmatched_dets), np.array(unmatched_trks)

# =============================================================================
# 3. ROS 2 Node
# =============================================================================
class LidarTrackerNode(Node):
    def __init__(self):
        super().__init__('lidar_tracker_node')
        
        # 구독
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10) # odom 이름 확인
        
        self.sub_det = self.create_subscription(
            MarkerArray, '/pcdet/detections', self.det_callback, 10)

        # 발행
        self.pub_track = self.create_publisher(MarkerArray, '/tracker/markers', 10)
        
        # 트래커 인스턴스
        self.tracker = EgoMotionTracker()
        
        # 상태 변수
        self.current_twist = {'vx': 0.0, 'wz': 0.0}
        self.last_time = self.get_clock().now()
        
        self.get_logger().info("Ego-Motion Compensated Tracker Started!")

    def odom_callback(self, msg):
        self.current_twist['vx'] = msg.twist.twist.linear.x
        self.current_twist['wz'] = msg.twist.twist.angular.z

    def det_callback(self, msg):
        # 1. 시간 차이(dt) 계산
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # dt가 너무 크거나 작으면(첫 실행 등) 스킵
        if dt > 1.0 or dt < 0.001: 
            dt = 0.1

        # 2. MarkerArray -> Numpy 변환
        dets_list = []
        for marker in msg.markers:
            if marker.action == Marker.DELETE: continue
            
            # 중심점 (Pose)
            cx, cy, cz = marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
            # 크기 (Scale) -> l, w, h
            l, w, h = marker.scale.x, marker.scale.y, marker.scale.z
            # 방향 (Quaternion -> Yaw) - 여기선 간단히 0으로 가정하거나 쿼터니언 변환 필요
            theta = 0.0 
            
            # 좌표가 0인 경우 포인트 평균으로 대체 (OpenPCDet 포맷 대응)
            if cx == 0 and cy == 0 and len(marker.points) > 0:
                pts = np.array([(p.x, p.y, p.z) for p in marker.points])
                cx, cy, cz = np.mean(pts, axis=0)

            dets_list.append([cx, cy, cz, theta, l, w, h])

        dets_np = np.array(dets_list)
        if len(dets_np) == 0: 
            return

        # 3. 트래킹 수행 (핵심!)
        track_results = self.tracker.track(dets_np, self.current_twist, dt)
        
        # 4. 결과 시각화 (MarkerArray 생성)
        self.publish_tracks(track_results)

    def publish_tracks(self, track_results):
        marker_array = MarkerArray()
        
        # 잔상 제거용 DeleteAll
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        marker_array.markers.append(del_marker)
        
        for trk in track_results:
            # trk: [x, y, z, theta, l, w, h, vx, vy, vz, id]
            x, y, z = trk[0], trk[1], trk[2]
            l, w, h = trk[4], trk[5], trk[6]
            vx, vy = trk[7], trk[8]
            trk_id = int(trk[10])
            
            # 1. 박스 마커
            marker = Marker()
            marker.header.frame_id = "base_link" # 혹은 사용중인 frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "objects"
            marker.id = trk_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.scale.x = l
            marker.scale.y = w
            marker.scale.z = h
            marker.color.a = 0.6
            # ID별로 색상 다르게 (Hash)
            np.random.seed(trk_id)
            marker.color.r = np.random.rand()
            marker.color.g = np.random.rand()
            marker.color.b = np.random.rand()
            marker.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(marker)
            
            # 2. 정보 텍스트 (ID + 속도)
            text_marker = Marker()
            text_marker.header.frame_id = "base_link"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "info"
            text_marker.id = trk_id + 10000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + h + 0.5 # 박스 위에 표시
            text_marker.scale.z = 0.5 # 글자 크기
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            
            speed = math.sqrt(vx**2 + vy**2)
            text_marker.text = f"ID:{trk_id}\n{speed:.1f} m/s"
            text_marker.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(text_marker)

        self.pub_track.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = LidarTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()