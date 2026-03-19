#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import time


class KalmanTracker:
    """
    각 물체(트랙)를 나타냄.
    상태: [x, y, vx, vy]
    크기(scale_x,y,z)와 color는 마지막 관측 기반으로 유지
    """
    _next_id = 0

    def __init__(self, cx, cy, sx, sy, sz, dt=0.1):
        # 고유 ID
        self.id = KalmanTracker._next_id
        KalmanTracker._next_id += 1

        # 상태 (4x1)
        self.dt = dt
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)

        # 공분산
        self.P = np.eye(4) * 10.0

        # 상태 전이 행렬 F
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,      0],
            [0, 0, 0,      1],
        ], dtype=float)

        # 관측 행렬 H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # 프로세스 잡음 Q, 측정 잡음 R
        self.Q = np.eye(4) * 0.05
        self.R = np.eye(2) * 0.5

        # 시각화용 정보
        self.scale_x = sx
        self.scale_y = sy
        self.scale_z = sz

        # ID마다 고정 색상
        self.color = ColorRGBA()
        self.color.r = random.random()
        self.color.g = random.random()
        self.color.b = random.random()
        self.color.a = 0.7

        # bookkeeping
        self.time_since_update = 0  # 최근 업데이트로부터 지난 step 수
        self.hits = 1               # 총 몇 번 관측과 매칭되었는지
        self.miss = 0               # 연속으로 관측 안 된 횟수

    def predict(self):
        # x_k+1 = F x_k
        self.x = self.F @ self.x
        # P_k+1 = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:2].ravel()  # predicted [cx, cy]

    def update(self, meas_cx, meas_cy, sx, sy, sz):
        """
        관측 (measured center (cx, cy) + 박스 사이즈)
        """
        z = np.array([[meas_cx], [meas_cy]], dtype=float)

        # y = z - Hx
        y = z - (self.H @ self.x)

        # S = H P H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # K = P H^T S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # x = x + K y
        self.x = self.x + K @ y

        # P = (I-KH)P
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        # 사이즈는 최신 관측으로 갱신
        self.scale_x = sx
        self.scale_y = sy
        self.scale_z = sz

        self.time_since_update = 0
        self.hits += 1
        self.miss = 0

    def mark_missed(self):
        # 관측이 안 붙었을 때 부르는 함수
        self.time_since_update += 1
        self.miss += 1

    def get_center(self):
        return self.x[0,0], self.x[1,0]

    def get_scale(self):
        return self.scale_x, self.scale_y, self.scale_z


class MultiObjectTracker:
    """
    여러 KalmanTracker를 관리하고 Hungarian 알고리즘으로 데이터 연관(matching)
    """
    def __init__(self,
                 dist_thresh=2.0,
                 max_miss=5):
        self.trackers = []
        self.dist_thresh = dist_thresh  # 매칭 허용 거리 (m)
        self.max_miss = max_miss        # 이 이상 놓치면 삭제

    def update(self, detections):
        """
        detections: list of (cx, cy, sx, sy, sz)
        출력: self.trackers 리스트 (업데이트된 상태)
        """

        # 1) 모든 tracker 예측 단계
        predictions = []
        for trk in self.trackers:
            pred_xy = trk.predict()
            predictions.append(pred_xy)

        # 2) 매칭할 detection이 없다면 -> 트래커들 miss 증가만
        if len(detections) == 0:
            for trk in self.trackers:
                trk.mark_missed()
            # 오래된거 제거
            self.trackers = [t for t in self.trackers if t.miss <= self.max_miss]
            return self.trackers

        # 3) 트래커가 하나도 없으면 -> 전부 새로 생성
        if len(self.trackers) == 0:
            for det in detections:
                cx, cy, sx, sy, sz = det
                self.trackers.append(KalmanTracker(cx, cy, sx, sy, sz))
            return self.trackers

        # 4) 비용 행렬 (pred vs det)
        cost = np.zeros((len(self.trackers), len(detections)), dtype=float)
        for i, trk in enumerate(self.trackers):
            px, py = trk.get_center()  # NOTE: could also use predictions[i]
            for j, det in enumerate(detections):
                cx, cy, sx, sy, sz = det
                cost[i, j] = np.sqrt((px - cx)**2 + (py - cy)**2)

        # 5) Hungarian (최소 비용 매칭)
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_trk = set()
        matched_det = set()

        # 6) 매칭된 것들 업데이트
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.dist_thresh:
                cx, cy, sx, sy, sz = detections[c]
                self.trackers[r].update(cx, cy, sx, sy, sz)
                matched_trk.add(r)
                matched_det.add(c)

        # 7) 매칭 안 된 트래커 -> miss 증가
        for i, trk in enumerate(self.trackers):
            if i not in matched_trk:
                trk.mark_missed()

        # 8) 매칭 안 된 detection -> 새 트래커 생성
        for j, det in enumerate(detections):
            if j not in matched_det:
                cx, cy, sx, sy, sz = det
                self.trackers.append(KalmanTracker(cx, cy, sx, sy, sz))

        # 9) miss 오래된 트래커 제거
        self.trackers = [t for t in self.trackers if t.miss <= self.max_miss]

        return self.trackers


class TrackerNode(Node):
    """
    구독: /obstacle_boxes (cluster_node에서 퍼블리시한 MarkerArray: 각 감지된 바운딩박스)
    퍼블리시: /tracked_boxes (MarkerArray: 칼만필터로 안정화된 바운딩박스)
    - 같은 ID는 같은 색 유지
    """

    def __init__(self):
        super().__init__('obstacle_tracker_node')

        # 파라미터 (원하면 바꿀 수 있게 선언)
        self.declare_parameter('input_topic', '/obstacle_boxes')
        self.declare_parameter('output_topic', '/tracked_boxes')
        self.declare_parameter('dist_thresh', 2.0)
        self.declare_parameter('max_miss', 5)

        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        dist_thresh = float(self.get_parameter('dist_thresh').get_parameter_value().double_value)
        max_miss = int(self.get_parameter('max_miss').get_parameter_value().integer_value)

        self.tracker = MultiObjectTracker(
            dist_thresh=dist_thresh,
            max_miss=max_miss
        )

        self.sub_ = self.create_subscription(MarkerArray, in_topic, self.cb, 10)
        self.pub_ = self.create_publisher(MarkerArray, out_topic, 10)

        self.get_logger().info("✅ obstacle_tracker_node ready (Kalman + box tracking w/ stable colors)")

    def cb(self, msg: MarkerArray):
        t0 = time.time()

        # 1) 감지 결과 추출: (cx, cy, sx, sy, sz)
        detections = []
        for m in msg.markers:
            cx = m.pose.position.x
            cy = m.pose.position.y
            sx = m.scale.x
            sy = m.scale.y
            sz = m.scale.z
            detections.append((cx, cy, sx, sy, sz))

        # 2) 트래커 업데이트
        tracks = self.tracker.update(detections)

        # 3) 결과를 MarkerArray로 만들어서 퍼블리시
        out_arr = MarkerArray()
        for trk in tracks:
            cx, cy = trk.get_center()
            sx, sy, sz = trk.get_scale()

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()

            # frame은 입력의 frame과 동일하게 맞춰야 RViz에서 어긋나지 않아
            # 입력 msg가 비어있지 않다면 첫 마커의 frame_id를 그대로 사용
            marker.header.frame_id = (
                msg.markers[0].header.frame_id
                if len(msg.markers) > 0
                else "map"
            )

            marker.ns = "tracked_boxes"
            marker.id = trk.id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(cx)
            marker.pose.position.y = float(cy)
            # Z는 추적기 상태에는 없으니까, 관측된 scale 기준으로 적당히 올려줌
            marker.pose.position.z = float(sz * 0.5)

            marker.scale.x = float(sx)
            marker.scale.y = float(sy)
            marker.scale.z = float(sz)

            # 고정된 색 (트래커 생성 시 결정된 색)
            marker.color = trk.color

            marker.lifetime.sec = 0  # 0이면 계속 보임; 필요하면 짧게 줄 수 있음
            out_arr.markers.append(marker)

        self.pub_.publish(out_arr)

        self.get_logger().info(
            f"tracked={len(tracks)} elapsed={(time.time()-t0)*1000:.1f}ms"
        )


def main():
    rclpy.init()
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
