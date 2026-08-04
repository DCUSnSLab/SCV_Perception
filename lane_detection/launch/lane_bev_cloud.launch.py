from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('lane_detection'), 'config', 'bev_cloud.yaml'
    ])

    return LaunchDescription([
        Node(
            package='lane_detection',
            executable='lane_bev_cloud_node',
            name='lane_bev_cloud_node',
            output='screen',
            parameters=[config],
        ),
    ])
