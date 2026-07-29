from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('left_detections_topic', default_value='/perception/left/detections'),
        DeclareLaunchArgument('right_detections_topic', default_value='/perception/right/detections'),
        Node(
            package='crosswalk_safety_ros2',
            executable='crosswalk_safety_node',
            name='crosswalk_safety_node',
            output='screen',
            parameters=[{
                'left_detections_topic': LaunchConfiguration('left_detections_topic'),
                'right_detections_topic': LaunchConfiguration('right_detections_topic'),
            }],
        ),
    ])
