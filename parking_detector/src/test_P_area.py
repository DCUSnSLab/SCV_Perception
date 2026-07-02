#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from parking_detector.msg import MultipleWaypoints
import time

class TestWaypointsPublisher(Node):
    def __init__(self):
        super().__init__('test_waypoints_publisher')
        self.publisher = self.create_publisher(MultipleWaypoints, '/MultipleWaypoints', 10)
        self.timer = self.create_timer(5.0, self.publish_test_waypoints)
        self.counter = 0
        self.get_logger().info('Test waypoints publisher started')

    def publish_test_waypoints(self):
        msg = MultipleWaypoints()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        current_goal = PoseStamped()
        current_goal.header = msg.header
        current_goal.pose.position.x = 1.0
        current_goal.pose.position.y = 2.0
        current_goal.pose.position.z = 0.0
        current_goal.pose.orientation.w = 1.0

        msg.current_goal = current_goal
        msg.current_goal_reverse_heading = False

        msg.current_goal_node_type = 1
        current_section_name = "일반"

        next_waypoint1 = PoseStamped()
        next_waypoint1.header = msg.header
        next_waypoint1.pose.position.x = 2.0
        next_waypoint1.pose.position.y = 3.0
        next_waypoint1.pose.position.z = 0.0
        next_waypoint1.pose.orientation.w = 1.0

        next_waypoint2 = PoseStamped()
        next_waypoint2.header = msg.header
        next_waypoint2.pose.position.x = 3.0
        next_waypoint2.pose.position.y = 4.0
        next_waypoint2.pose.position.z = 0.0
        next_waypoint2.pose.orientation.w = 1.0

        msg.next_waypoints = [next_waypoint1, next_waypoint2]
        msg.next_waypoints_reverse_heading = [False, False]
        msg.next_waypoints_node_types = [13, 1]
        msg.path_id = "test_path_001"
        msg.current_waypoint_index = self.counter
        msg.total_waypoints = 10
        msg.is_final_waypoint = (self.counter >= 9)

        self.publisher.publish(msg)
        self.get_logger().info(f'Published waypoint #{self.counter} - Current: {current_section_name} (node_type: {msg.current_goal_node_type}), Next nodes: {msg.next_waypoints_node_types}')
        self.counter += 1
        msg.next_waypoints = [next_waypoint1, next_waypoint2]
        msg.next_waypoints_reverse_heading = [False, False]
        msg.next_waypoints_node_types = [13, 1]
        msg.path_id = "test_path_001"
        msg.current_waypoint_index = self.counter
        msg.total_waypoints = 10
        msg.is_final_waypoint = (self.counter >= 9)

        self.publisher.publish(msg)
        self.get_logger().info(f'Published waypoint #{self.counter} - Current: {current_section_name} (node_type: {msg.current_goal_node_type}), Next nodes: {msg.next_waypoints_node_types}')
        self.counter += 1
        if self.counter >= 10:
            self.counter = 0

def main(args=None):
    rclpy.init(args=args)
    
    test_publisher = TestWaypointsPublisher()
    
    try:
        rclpy.spin(test_publisher)
    except KeyboardInterrupt:
        print("\nShutting down test publisher...")
    
    test_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()