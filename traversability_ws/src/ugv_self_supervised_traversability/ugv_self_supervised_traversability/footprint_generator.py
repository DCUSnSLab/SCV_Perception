"""Publish current and accumulated rectangular traversed footprints."""

from typing import List

import rclpy
from geometry_msgs.msg import Point, Point32, PolygonStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from .utils.geometry import footprint_corners
from .utils.transforms import euler_from_quaternion


class FootprintGenerator(Node):
    """Convert each accepted odometry pose into the UGV's occupied ground area."""

    def __init__(self) -> None:
        super().__init__('footprint_generator')
        for name, default in (
            ('odom_topic', '/odom'), ('footprint_topic', '/traversability/footprint'),
            ('traversed_area_topic', '/traversability/traversed_area'),
                ('world_frame', 'odom')):
            self.declare_parameter(name, default)
        self.declare_parameter('robot_width', 0.8)
        self.declare_parameter('robot_length', 1.2)
        self.declare_parameter('footprint_margin', 0.05)
        self.declare_parameter('min_footprint_distance', 0.05)
        self.declare_parameter('max_visualized_footprints', 2000)
        self.width = float(self.get_parameter('robot_width').value)
        self.length = float(self.get_parameter('robot_length').value)
        self.margin = float(self.get_parameter('footprint_margin').value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.min_distance = float(self.get_parameter('min_footprint_distance').value)
        self.max_footprints = int(self.get_parameter('max_visualized_footprints').value)
        self.last_xy = None
        self.footprints: List[List[Point]] = []
        self.polygon_pub = self.create_publisher(
            PolygonStamped, str(self.get_parameter('footprint_topic').value), 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter('traversed_area_topic').value), 10)
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self._callback, qos)

    def _callback(self, message: Odometry) -> None:
        if message.header.frame_id and message.header.frame_id != self.world_frame:
            self.get_logger().warning(
                'Ignoring odometry outside world_frame', throttle_duration_sec=5.0)
            return
        position = message.pose.pose.position
        if self.last_xy is not None:
            distance = ((position.x - self.last_xy[0]) ** 2 +
                        (position.y - self.last_xy[1]) ** 2) ** 0.5
            if distance < self.min_distance:
                return
        quaternion = message.pose.pose.orientation
        yaw = euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]
        corners = footprint_corners(position.x, position.y, position.z, yaw,
                                    self.length, self.width, self.margin)
        polygon = PolygonStamped()
        polygon.header = message.header
        polygon.header.frame_id = self.world_frame
        polygon.polygon.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                                  for p in corners]
        self.polygon_pub.publish(polygon)
        points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]) + 0.015)
                  for p in corners]
        self.footprints.append(points)
        self.footprints = self.footprints[-self.max_footprints:]
        self.last_xy = (position.x, position.y)
        self._publish_markers(message)

    def _publish_markers(self, message: Odometry) -> None:
        current = Marker()
        current.header = message.header
        current.header.frame_id = self.world_frame
        current.ns = 'current_footprint'
        current.id, current.type, current.action = 0, Marker.LINE_STRIP, Marker.ADD
        current.scale.x = 0.04
        current.color.r, current.color.g, current.color.b, current.color.a = 0.1, 1.0, 0.1, 1.0
        current.points = self.footprints[-1] + [self.footprints[-1][0]]
        area = Marker()
        area.header = current.header
        area.ns = 'traversed_area'
        area.id, area.type, area.action = 0, Marker.TRIANGLE_LIST, Marker.ADD
        area.color.r, area.color.g, area.color.b, area.color.a = 0.1, 0.8, 0.2, 0.25
        area.scale.x = area.scale.y = area.scale.z = 1.0
        for footprint in self.footprints:
            area.points.extend([footprint[0], footprint[1], footprint[2],
                                footprint[0], footprint[2], footprint[3]])
        self.marker_pub.publish(MarkerArray(markers=[current, area]))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FootprintGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()
