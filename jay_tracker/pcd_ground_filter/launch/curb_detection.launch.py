import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('pcd_ground_filter')
    cfg = os.path.join(pkg, 'config', 'curb_params.yaml')
    return LaunchDescription([
        Node(
            package='pcd_ground_filter',
            executable='curb_detection_node',
            name='curb_detection_node',
            output='screen',
            respawn=True,
            respawn_delay=1.0,
            parameters=[cfg],
        ),
    ])
