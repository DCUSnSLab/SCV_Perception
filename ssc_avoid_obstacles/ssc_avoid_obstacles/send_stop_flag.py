#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
#from zed_msgs.msg import ObjectsStamped

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

MIN_X, MAX_X = 4.0, 8.0                 ## 3차원 ROI가 시작/끝나는 지점과 기준 프레임과의 거리(X)
MIN_Y, MAX_Y = -1.0, 1.0                ## 2차원 ROI의 우측/좌측 끝(Y)
MIN_Z, MAX_Z = 0.0, 1.5                 ## 2차원 ROI의 제일 아래/위(Z)
NUM_OF_POINTS = 500                       ## 장애물 판단의 기준이 될 점의 개수
#ROI_POINTS = [(), (), (), ()]          ## 2차원 ROI의 네 꼭짓점의 좌표(y,z)의 배열 [length=4] 근데 이거 없어도 될 듯?
#       ---------
#     /.        /|
#    0 ------- 1 |
#    | .       | |
#    | ........| |  <= MAX_DIS
#    |.        |/
#    3 ------- 2    <= MIN_DIS
# GPT 안 썼습니다 위의 멋진 그림들은 StereoLabs 사이트에서 훔쳐오거나 직접 만든 겁니다
# 주.꾸(주석 꾸미기)가 제 취미입니다


class PointCloudSubscriber(Node):
    def __init__(self):

        ## 노드 이름 설정
        super().__init__('send_stop_flag')                                      

        ## 매개변수 선언
        global MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_Z, MAX_Z, NUM_OF_POINTS

        self.declare_parameter('min_x', MIN_X)
        self.declare_parameter('max_x', MAX_X)
        self.declare_parameter('min_y', MIN_Y)
        self.declare_parameter('max_y', MAX_Y)
        self.declare_parameter('min_z', MIN_Z)
        self.declare_parameter('max_z', MAX_Z)
        self.declare_parameter('num_of_points', NUM_OF_POINTS)

        ## 선언된 파라미터로 모듈 상수 덮어쓰기
        MIN_X = float(self.get_parameter('min_x').value)
        MAX_X = float(self.get_parameter('max_x').value)
        MIN_Y = float(self.get_parameter('min_y').value)
        MAX_Y = float(self.get_parameter('max_y').value)
        MIN_Z = float(self.get_parameter('min_z').value)
        MAX_Z = float(self.get_parameter('max_z').value)
        NUM_OF_POINTS = int(self.get_parameter('num_of_points').value)

        self.flag_pub = self.create_publisher(Bool, TOPIC_FLAG_PUB, PUB_HZ)     ## 정지 플래그를 발행하는 Publisher 선언
        self.box_pub = self.create_publisher(Marker, TOPIC_ROI_PUB, PUB_HZ)     ## ROI 영역을 Marker로 발행하는 Publisher 선언
        self.pnt_sub = self.create_subscription(                                ## 포인트클라우드를 구독하는 Subscriber 선언
            PointCloud2,
            TOPIC_PNT_SUB,
            self.pnt_callback,
            PUB_HZ
        )

    def build_roi_marker(self, is_red: bool) -> Marker:
        marker = Marker()
        marker.header.frame_id = STANDARD_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "roi"
        marker.id = 0                   # ★ 같은 ID 유지: 누적 대신 갱신
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
        '''
        카메라 센서가 발행하는 포인트클라우드 토픽을 구독해서
        장애물을 탐지하면 정지 플래그를 날리는 콜백함수
        - 평소: False
        - 정지해야 함: True
        '''

        flag = Bool()
        
        if True:                                               ## 모든 점이 유효하다면
            flag.data = self.check_obstacles(msg)
        else:                                                           ## 유효하지 않은 점이 존재한다면
            self.get_logger().warn(f"[Warning] Invalid PointCloud2 Data ! (msg.is_dense={msg.is_dense})")
            flag.data = msg.is_dense

        self.flag_pub.publish(flag)
        self.box_pub.publish(self.build_roi_marker(flag.data))
        self.get_logger().info(f"[Publish] flag={flag.data}")


    def check_obstacles(self, data: PointCloud2) -> bool:
        '''
        3차원 ROI 영역 내 점의 개수를 세고
        기준이 되는 개수 n보다 많은지 적은지를
        True/False로 반환하는 함수
        '''

        cnt = 0     # ROI 안에 있는 점의 개수

        for x, y, z, rgb in point_cloud2.read_points(data, field_names=("x", "y", "z", "rgb"), skip_nans=True):    ## 점 데이터를 읽어와서 
            #self.get_logger().info(f"[Point] >>> p({x:.2f}, {y:.2f}, {z:.2f})")
            if (MIN_X <= x <= MAX_X) and (MIN_Y <= y <= MAX_Y) and (MIN_Z <= z <= MAX_Z):         ## 해당 점이 ROI 안에 있다면 개수 합산
                cnt+=1
                if cnt >= NUM_OF_POINTS:
                    break
        
        result = bool(cnt >= NUM_OF_POINTS)
        self.get_logger().info(f"[Result] cnt={cnt}")
        self.get_logger().info(f"[Result] obstacle? : {result}")
        return result
    

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