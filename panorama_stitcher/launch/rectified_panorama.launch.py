from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('panorama_stitcher'),
        'config',
        'rectified_panorama.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Front-facing rectified panorama parameter file',
        ),
        Node(
            package='panorama_stitcher',
            executable='rgbd_panorama_stitcher_node',
            name='rectified_panorama_stitcher',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
