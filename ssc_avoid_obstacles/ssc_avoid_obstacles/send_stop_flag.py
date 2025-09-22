#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2       # ← 구독 타입은 PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Bool, Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
# from zed_msgs.msg import ObjectsStamped

#             (높이) Z
#                   △   X (정면)
#                   |  ◁
#                   | / 
#                   |/
#       Y ◁------- 차

TOPIC_PNT_SUB = '/zed/zed_node/point_cloud/cloud_registered'
TOPIC_ROI_PUB = '/roi_vis'
TOPIC_FLAG_PUB = '/stop_flag'
STANDARD_FRAME = 'zed_camera_center'    ## ROI 영역이 시각화될 기준 프레임
PUB_HZ = 10
MAX_NUM = 0                             ## 가장 많이 검출된 점의 개수(계속 갱신됨)

MIN_X, MAX_X = 4.0, 8.0                 ## 3차원 ROI가 시작/끝나는 지점과 기준 프레임과의 거리(X)
MIN_Y, MAX_Y = -1.0, 1.0                ## 2차원 ROI의 우측/좌측 끝(Y)
MIN_Z, MAX_Z = 0.0, 1.5                 ## 2차원 ROI의 제일 아래/위(Z)
NUM_OF_POINTS = 250                     ## 장애물 판단의 기준이 될 점의 개수

DEBUG = False


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('send_stop_flag')

        ## 매개변수 선언
        global MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_Z, MAX_Z, NUM_OF_POINTS, DEBUG

        self.declare_parameter('min_x', MIN_X)
        self.declare_parameter('max_x', MAX_X)
        self.declare_parameter('min_y', MIN_Y)
        self.declare_parameter('max_y', MAX_Y)
        self.declare_parameter('min_z', MIN_Z)
        self.declare_parameter('max_z', MAX_Z)
        self.declare_parameter('num_of_points', NUM_OF_POINTS)
        self.declare_parameter('debug', DEBUG)

        ## 파라미터 반영
        MIN_X = float(self.get_parameter('min_x').value)
        MAX_X = float(self.get_parameter('max_x').value)
        MIN_Y = float(self.get_parameter('min_y').value)
        MAX_Y = float(self.get_parameter('max_y').value)
        MIN_Z = float(self.get_parameter('min_z').value)
        MAX_Z = float(self.get_parameter('max_z').value)
        NUM_OF_POINTS = int(self.get_parameter('num_of_points').value)
        DEBUG = bool(self.get_parameter('debug').value)

        ## Pub/Sub
        self.flag_pub = self.create_publisher(Bool, TOPIC_FLAG_PUB, PUB_HZ)
        self.box_pub  = self.create_publisher(Marker, TOPIC_ROI_PUB, PUB_HZ)
        self.pnt_sub  = self.create_subscription(
            PointCloud2,              # ← 올바른 메시지 타입
            TOPIC_PNT_SUB,
            self.pnt_callback,
            10
        )

        ## ROI 마커 1회 생성 & 상태 캐시
        self._roi_marker = self._init_roi_marker()
        self._last_color_is_red = None  # 마지막 색 상태 (변경시에만 publish)

    # ---------- 여기 추가: 마커 초기화 ----------
    def _init_roi_marker(self) -> Marker:
        m = Marker()
        m.header.frame_id = STANDARD_FRAME
        m.ns = "roi"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD

        # ROI 중심/크기
        cx = (MIN_X + MAX_X) / 2.0
        cy = (MIN_Y + MAX_Y) / 2.0
        cz = (MIN_Z + MAX_Z) / 2.0
        sx = max((MAX_X - MIN_X), 1e-6)
        sy = max((MAX_Y - MIN_Y), 1e-6)
        sz = max((MAX_Z - MIN_Z), 1e-6)

        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = cz
        m.pose.orientation.w = 1.0

        m.scale.x = sx
        m.scale.y = sy
        m.scale.z = sz

        # 초기색: 초록(안전)
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.25

        # RViz에서 계속 보이도록 영구수명
        m.lifetime = Duration(sec=0, nanosec=0)
        return m

    # ---------- 여기 추가: 색만 바뀔 때 발행 ----------
    def publish_roi(self, is_red: bool) -> None:
        # 변화 없으면 발행 생략 (불필요한 비용 절약)
        if self._last_color_is_red == is_red:
            return
        self._last_color_is_red = is_red

        m = self._roi_marker
        m.header.stamp = self.get_clock().now().to_msg()
        if is_red:
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.25
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.25

        self.box_pub.publish(m)

    def build_roi_marker(self, is_red: bool) -> Marker:
        """(참고용) 기존 방식: 매번 새로 생성 — 이제는 사용 안 함"""
        marker = Marker()
        marker.header.frame_id = STANDARD_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "roi"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        cx = (MIN_X + MAX_X) / 2.0
        cy = (MIN_Y + MAX_Y) / 2.0
        cz = (MIN_Z + MAX_Z) / 2.0
        sx = (MAX_X - MIN_X)
        sy = (MAX_Y - MIN_Y)
        sz = (MAX_Z - MIN_Z)

        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = cz
        marker.pose.orientation.w = 1.0

        marker.scale.x = max(sx, 1e-6)
        marker.scale.y = max(sy, 1e-6)
        marker.scale.z = max(sz, 1e-6)

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.25
        if is_red:
            marker.color.r = 1.0
            marker.color.g = 0.0

        marker.lifetime = Duration(sec=0, nanosec=0)
        return marker

    def pnt_callback(self, msg: PointCloud2) -> None:
        start = self.get_clock().now()

        flag = Bool()
        flag.data = self.check_obstacles(msg)

        self.flag_pub.publish(flag)
        # 변경점: 매 콜백마다 새 Marker를 만들지 않고, 색상 바뀔 때만 발행
        self.publish_roi(flag.data)

        # self.get_logger().info(f"[Publish] flag={flag.data}")

        end = self.get_clock().now()
        if DEBUG:
            elapsed = (end - start).nanoseconds / 1e6
            self.get_logger().info(f"[Time] callback took {elapsed:.2f} ms")
            self.get_logger().info(f"[Info] flag = {flag.data}")

    def check_obstacles(self, data: PointCloud2) -> bool:
        global NUM_OF_POINTS, MAX_NUM
        try:
            arr = pc2.read_points_numpy(data, field_names=("x","y","z"))
            k = 4
            arr = arr[::k]

            valid = np.isfinite(arr).all(axis=1)
            xyz = arr[valid]

            in_roi = (
                (MIN_X <= xyz[:,0]) & (xyz[:,0] <= MAX_X) &
                (MIN_Y <= xyz[:,1]) & (xyz[:,1] <= MAX_Y) &
                (MIN_Z <= xyz[:,2]) & (xyz[:,2] <= MAX_Z)
            )

            cnt = int(in_roi.sum())
            if cnt > MAX_NUM:
                MAX_NUM = cnt
            if DEBUG:
                self.get_logger().info(f"[Info] points = {cnt} / {NUM_OF_POINTS}")
                self.get_logger().info(f"[Info] max_num = {MAX_NUM}")
            return cnt >= NUM_OF_POINTS

        except AttributeError:
            # 구버전 대체 경로
            buf = np.frombuffer(data.data, dtype=np.uint8)
            point_step = data.point_step
            n = len(buf) // point_step
            buf = buf[:n*point_step].reshape(n, point_step)

            ofs = {f.name: f.offset for f in data.fields}
            def f32(col):
                return np.frombuffer(
                    buf[:, ofs[col]:ofs[col]+4].reshape(-1,4).data, dtype='<f4', count=n
                )

            x = f32('x'); y = f32('y'); z = f32('z')

            k = 4
            x = x[::k]; y = y[::k]; z = z[::k]

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x = x[valid]; y = y[valid]; z = z[valid]

            in_roi = (
                (MIN_X <= x) & (x <= MAX_X) &
                (MIN_Y <= y) & (y <= MAX_Y) &
                (MIN_Z <= z) & (z <= MAX_Z)
            )
            cnt = int(in_roi.sum())
            return cnt >= NUM_OF_POINTS


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
