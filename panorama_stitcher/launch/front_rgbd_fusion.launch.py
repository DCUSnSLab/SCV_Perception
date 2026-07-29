from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('panorama_stitcher'),
        'config',
        'front_rgbd_fusion.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Dual-front-camera RGB-D fusion parameters',
        ),
        Node(
            package='panorama_stitcher',
            executable='front_rgbd_fusion_node',
            name='front_rgbd_fusion',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
