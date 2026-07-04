import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    curb_pkg = get_package_share_directory('pcd_ground_filter')
    curb_cfg = os.path.join(curb_pkg, 'config', 'curb_params.yaml')

    costmap_pkg = get_package_share_directory('local_costmap')
    costmap_cfg = os.path.join(costmap_pkg, 'config', 'costmap_params.yaml')

    return LaunchDescription([
        # 1) curb detector: /velodyne_points -> /velodyne_points_curb (+curbs)
        Node(
            package='pcd_ground_filter',
            executable='curb_detection_node',
            name='curb_detection_node',
            output='screen',
            parameters=[curb_cfg],
        ),
        # 2) local costmap consuming the curb-augmented cloud
        Node(
            package='local_costmap',
            executable='costmap_node',
            name='local_costmap_node',
            output='screen',
            parameters=[
                costmap_cfg,
                {'point_cloud_topic': '/velodyne_points_curb'},
            ],
        ),
    ])
