import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('negative_obstacle_detector')
    config_file = os.path.join(pkg_share, 'config', 'detector_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        Node(
            package='negative_obstacle_detector',
            executable='detector_node',
            name='ground_filter',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
        ),
    ])
