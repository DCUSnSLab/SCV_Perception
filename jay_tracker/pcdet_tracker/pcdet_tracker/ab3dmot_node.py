#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
from tracking_msgs.msg import DetectedObject, DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray
from scipy.optimize import linear_sum_assignment

# ====================================================================
# 1. 3D Kalman Filter Class (개별 객체 추적기)
# ====================================================================
class KalmanBoxTracker3D:
    count = 0
    def __init__(self, bbox3d, class_name):
        # bbox3d: [x, y, z, l, w, h, theta]
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.id = KalmanBoxTracker3D.count
        KalmanBoxTracker3D.count += 1
        self.class_name = class_name
        
        # State: [x, y, z, theta, l, w, h, vx, vy, vz] (10 dim)
        self.x = np.zeros((10, 1))
        self.x[:7, 0] = bbox3d
        
        # F: State Transition Matrix (등속 모델)
        self.F = np.eye(10)
        self.F[0, 7] = 0.1 # dt = 0.1s
        self.F[1, 8] = 0.1
        self.F[2, 9] = 0.1
        
        # H: Measurement Matrix
        self.H = np.eye(7, 10)
        
        # P, Q, R: 공분산 행렬들
        self.P = np.eye(10) * 1000.0
        self.P[7:, 7:] *= 1000.0 
        self.Q = np.eye(10) * 0.01
        self.Q[7:, 7:] *= 0.01
        self.R = np.eye(7) * 0.1
        self.R[3, 3] *= 10.0 # 각도 노이즈

        self.R = np.eye(7) * 0.01
        self.R[3, 3] *= 1.0

    def update(self, bbox3d):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # 각도 보정 (-pi ~ pi)
        pred_theta = self.x[3, 0]
        meas_theta = bbox3d[3]
        diff = meas_theta - pred_theta
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        
        # 90도 이상 차이나면 방향 뒤집힘 방지
        if abs(diff) > np.pi / 2:
            bbox3d[3] += np.pi
            if bbox3d[3] > np.pi: bbox3d[3] -= 2*np.pi
            
        z = np.array(bbox3d).reshape((7, 1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.x - np.dot(K, np.dot(self.H, self.P))

        # ==================================================================
        # [핵심 수정] 크기와 방향은 필터링하지 말고 Detector 값을 그대로 사용!
        # ==================================================================
        # bbox3d: [x, y, z, l, w, h, theta]
        # self.x: [x, y, z, theta, l, w, h, vx, vy, vz]
        
        # 크기 (Length, Width, Height) 강제 동기화
        self.x[4, 0] = bbox3d[3]  # l
        self.x[5, 0] = bbox3d[4]  # w
        self.x[6, 0] = bbox3d[5]  # h
        
        # 방향 (Theta) 강제 동기화 (필요하면 주석 해제)
        # 트래커가 방향을 너무 굼뜨게 따라가면 이것도 강제로 맞추세요.
        self.x[3, 0] = bbox3d[6]  # theta

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.x[:7].flatten()

    def get_state(self):
        return self.x[:7].flatten()

# ====================================================================
# 2. Tracker Manager Class (클래스별 관리자)
# ====================================================================
class AB3DMOT_Manager:
    def __init__(self, class_name):
        self.class_name = class_name
        self.trackers = []
        
        # [수정 1] 메모리 유지 시간 대폭 증가 (ID 스위칭 방지 핵심)
        # LiDAR 10Hz 가정 시, 50프레임 = 5초
        self.max_age = 10       
        
        self.min_hits = 1       # 1번만 잡혀도 바로 트래킹 시작 (빠른 반응)
        self.iou_threshold = 0.01 # 매칭 임계값 (느슨하게 해서 재매칭 확률 높임)

    def update(self, dets):
        # 1. Predict
        trks = []
        to_del = []
        for t in self.trackers:
            pred = t.predict()
            trks.append(pred)
            if np.any(np.isnan(pred)):
                to_del.append(t)
        for t in to_del: self.trackers.remove(t)

        # 2. Associate
        matched, unmatched_dets, unmatched_trks = self.associate(dets, trks)

        # 3. Update Matched
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(dets[d_idx])

        # 4. Create New
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker3D(dets[d_idx], self.class_name)
            self.trackers.append(trk)

        # 5. Delete Dead
        final_trackers = []
        ret = []
        for t in self.trackers:
            # max_age(50프레임) 동안 업데이트 없으면 삭제
            if t.time_since_update < self.max_age:
                final_trackers.append(t)
                
                # 시각화용 리스트에 추가 (조건: 최소 히트 수 만족 or 방금 생성됨)
                if t.hits >= self.min_hits or t.age <= self.min_hits:
                    ret.append(t)
                    
        self.trackers = final_trackers
        return ret

    def associate(self, dets, trks):
        if len(trks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5), dtype=int)
            
        iou_matrix = np.zeros((len(trks), len(dets)), dtype=float)
        for t, trk in enumerate(trks):
            for d, det in enumerate(dets):
                dist = np.linalg.norm(trk[:2] - det[:2]) 
                iou_matrix[t, d] = 1.0 / (1.0 + dist)

        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_dets = []
        for d in range(len(dets)):
            if d not in matched_indices[:, 1]: unmatched_dets.append(d)
        
        unmatched_trks = []
        for t in range(len(trks)):
            if t not in matched_indices[:, 0]: unmatched_trks.append(t)

        matches = []
        for t, d in matched_indices:
            if iou_matrix[t, d] < self.iou_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matches.append([t, d])
        
        if len(matches) == 0: matches = np.empty((0, 2), dtype=int)
        else: matches = np.array(matches)

        return matches, unmatched_dets, unmatched_trks

# ====================================================================
# 3. ROS 2 Node
# ====================================================================
class AB3DMOTNode(Node):
    def __init__(self):
        super().__init__('ab3dmot_node')
        
        # 1. 클래스별 트래커 생성
        self.target_classes = ['Car', 'Pedestrian', 'Cyclist']
        self.managers = {cls: AB3DMOT_Manager(cls) for cls in self.target_classes}
        
        # 2. 통신 설정
        self.sub = self.create_subscription(
            DetectedObjectArray, '/detected_objects_3d', self.callback, 10
        )
        self.pub_tracks = self.create_publisher(DetectedObjectArray, '/tracked_objects_3d', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/tracking/visual_markers', 10)
        
        self.get_logger().info("AB3DMOT Tracker Initialized (Long Memory Mode)")

    def callback(self, msg):
        dets_by_class = {cls: [] for cls in self.target_classes}
        label_map = {'1': 'Car', '2': 'Pedestrian', '3': 'Cyclist'}
        
        for obj in msg.objects:
            cls_name = label_map.get(obj.label, obj.label)
            if cls_name in self.target_classes:
                dets_by_class[cls_name].append([
                    obj.pose[0], obj.pose[1], obj.pose[2],
                    obj.dimensions[0], obj.dimensions[1], obj.dimensions[2],
                    obj.yaw
                ])

        all_tracked_objects = []
        for cls_name in self.target_classes:
            dets = np.array(dets_by_class[cls_name])
            if len(dets) == 0: dets = np.empty((0, 7))
                
            active_trackers = self.managers[cls_name].update(dets)
            all_tracked_objects.extend(active_trackers)

        self.publish_results(all_tracked_objects, msg.header)

    def publish_results(self, trackers, header):
        out_msg = DetectedObjectArray()
        out_msg.header = header
        marker_array = MarkerArray()
        
        for trk in trackers:
            state = trk.get_state() # [x, y, z, theta, l, w, h]
            
            # [수정 2] Ghost 여부 확인 (이번 프레임에 업데이트 안 됐으면 Ghost)
            is_ghost = trk.time_since_update > 0

            # --- 1. System Msg (Ghost도 위치 추정값으로 계속 발행) ---
            obj = DetectedObject()
            obj.header = header
            obj.id = trk.id
            obj.label = trk.class_name
            obj.pose = [float(state[0]), float(state[1]), float(state[2])]
            obj.yaw = float(state[3])
            obj.dimensions = [float(state[4]), float(state[5]), float(state[6])]
            out_msg.objects.append(obj)
            
            # --- 2. Visual Marker ---
            marker = Marker()
            marker.header = header
            marker.ns = "tracking_box"
            marker.id = trk.id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000 
            
            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = float(state[2])
            
            yaw = float(state[3])
            marker.pose.orientation.z = math.sin(yaw / 2.0)
            marker.pose.orientation.w = math.cos(yaw / 2.0)
            
            marker.scale.x = float(state[4])
            marker.scale.y = float(state[5])
            marker.scale.z = float(state[6])
            
            # [수정 3] 색상 분기 처리 (Ghost vs Active)
            if is_ghost:
                # Ghost: 하늘색 (Cyan), 약간 더 투명하게
                marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 0.3
            else:
                # Active: 진한 파란색 (Blue)
                marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 0.6
                
            marker_array.markers.append(marker)
            
            # --- 3. Text Marker ---
            text = Marker()
            text.header = header
            text.ns = "tracking_text"
            text.id = trk.id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.lifetime.sec = 0
            text.lifetime.nanosec = 200000000
            
            text.pose.position.x = marker.pose.position.x
            text.pose.position.y = marker.pose.position.y
            text.pose.position.z = marker.pose.position.z + marker.scale.z/2.0 + 0.5
            
            text.scale.z = 0.5
            text.color.a = 1.0; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0
            
            # 텍스트 내용에 상태 표시 (Ghost일 경우 표시)
            status_str = "(Ghost)" if is_ghost else ""
            text.text = f"ID:{trk.id} {status_str}\n{trk.class_name}"
            marker_array.markers.append(text)

        self.pub_tracks.publish(out_msg)
        self.pub_markers.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = AB3DMOTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()