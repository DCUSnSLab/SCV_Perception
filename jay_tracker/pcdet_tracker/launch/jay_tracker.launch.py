"""Launch Jay_Tracker with the ablatable observability prototype."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    adaptive_default = PathJoinSubstitution([
        FindPackageShare('pcdet_tracker'), 'config', 'adaptive_tracking.yaml'])
    arguments = [
        DeclareLaunchArgument(
            'adaptive_config', default_value=adaptive_default,
            description='Object-wise observability YAML; empty is baseline.'),
        DeclareLaunchArgument(
            'detection_topic', default_value='/detected_objects_3d'),
        DeclareLaunchArgument(
            'tracked_topic', default_value='/tracked_objects_3d'),
        DeclareLaunchArgument(
            'bbox_topic', default_value='/pcdet/jay_tracks'),
        DeclareLaunchArgument('odom_topic', default_value='/odometry/wheel'),
        DeclareLaunchArgument('imu_topic', default_value='/vectornav/imu'),
        DeclareLaunchArgument('sequence_id', default_value='seq01'),
    ]
    tracker = Node(
        package='pcdet_tracker',
        executable='jay_tracker',
        name='jay_tracker',
        output='screen',
        arguments=[
            '--adaptive_config', LaunchConfiguration('adaptive_config'),
            '--detection_topic', LaunchConfiguration('detection_topic'),
            '--tracked_topic', LaunchConfiguration('tracked_topic'),
            '--bbox_topic', LaunchConfiguration('bbox_topic'),
            '--odom_topic', LaunchConfiguration('odom_topic'),
            '--imu_topic', LaunchConfiguration('imu_topic'),
            '--sequence_id', LaunchConfiguration('sequence_id'),
        ],
    )
    return LaunchDescription(arguments + [tracker])
