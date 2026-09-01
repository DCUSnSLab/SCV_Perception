import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    curb_pkg = get_package_share_directory('pcd_ground_filter')
    curb_cfg = os.path.join(curb_pkg, 'config', 'curb_params.yaml')

    costmap_pkg = get_package_share_directory('local_costmap')
    costmap_cfg = os.path.join(costmap_pkg, 'config', 'costmap_params.yaml')

    return LaunchDescription([
        # curb_method: 'below_grade'(검증 기본) | 'ring'(링 미분, 2026-08 램프 대응)
        DeclareLaunchArgument('curb_method', default_value='below_grade'),
        # 1) curb detector: /velodyne_points -> /velodyne_points_curb (+curbs)
        Node(
            package='pcd_ground_filter',
            executable='curb_detection_node',
            name='curb_detection_node',
            output='screen',
            respawn=True,
            respawn_delay=1.0,
            parameters=[curb_cfg,
                        {'method': LaunchConfiguration('curb_method')}],
        ),
        # 2) local costmap consuming the curb-augmented cloud
        Node(
            package='local_costmap',
            executable='costmap_node',
            name='local_costmap_node',
            output='screen',
            respawn=True,
            respawn_delay=1.0,
            parameters=[
                costmap_cfg,
                {'point_cloud_topic': '/velodyne_points_curb'},
            ],
        ),
    ])
