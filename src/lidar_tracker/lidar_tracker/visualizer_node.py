import rclpy
from rclpy.node import Node
from tracking_msgs.msg import DetectedObjectArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')
        
        # 입력: 추적 결과
        self.sub = self.create_subscription(
            DetectedObjectArray, '/tracked_objects_3d', self.callback, 10
        )
        
        # 출력: Rviz용 마커
        self.pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)

    def callback(self, msg):
        marker_array = MarkerArray()
        
        # 기존 마커 삭제 (Clean up)
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, obj in enumerate(msg.objects):
            # 1. Bounding Box Marker (Cube)
            box_marker = Marker()
            box_marker.header = msg.header
            box_marker.ns = "boxes"
            box_marker.id = i
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            
            # 위치
            box_marker.pose.position.x = float(obj.pose[0])
            box_marker.pose.position.y = float(obj.pose[1])
            box_marker.pose.position.z = float(obj.pose[2]) + (float(obj.dimensions[2]) / 2.0) # 바닥 기준 보정
            
            # 회전 (Yaw -> Quaternion)
            q = self.yaw_to_quaternion(obj.yaw)
            box_marker.pose.orientation.x = q[0]
            box_marker.pose.orientation.y = q[1]
            box_marker.pose.orientation.z = q[2]
            box_marker.pose.orientation.w = q[3]
            
            # 크기
            box_marker.scale.x = float(obj.dimensions[0]) # L
            box_marker.scale.y = float(obj.dimensions[1]) # W
            box_marker.scale.z = float(obj.dimensions[2]) # H
            
            # 색상 (초록색, 투명도 약간)
            box_marker.color.r = 0.0
            box_marker.color.g = 1.0
            box_marker.color.b = 0.0
            box_marker.color.a = 0.5
            
            marker_array.markers.append(box_marker)
            
            # 2. Text Marker (ID & Score)
            text_marker = Marker()
            text_marker.header = msg.header
            text_marker.ns = "ids"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(obj.pose[0])
            text_marker.pose.position.y = float(obj.pose[1])
            text_marker.pose.position.z = float(obj.pose[2]) + float(obj.dimensions[2]) + 0.5 # 박스 위
            
            text_marker.scale.z = 0.5 # 글자 크기
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = f"{obj.label} {obj.id}\n({obj.score:.2f})"
            
            marker_array.markers.append(text_marker)
            
        self.pub.publish(marker_array)

    def yaw_to_quaternion(self, yaw):
        # 간단한 Euler to Quaternion 변환 (Z축 회전만 고려)
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()