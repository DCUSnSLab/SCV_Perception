from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('panorama_stitcher'),
        'config',
        'rig_calibration.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Dual-camera ChArUco RGB-D calibration parameters',
        ),
        Node(
            package='panorama_stitcher',
            executable='charuco_rig_calibrator',
            name='charuco_rig_calibrator',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
