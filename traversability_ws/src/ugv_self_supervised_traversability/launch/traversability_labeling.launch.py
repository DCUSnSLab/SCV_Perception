"""Launch the complete phase-1 delayed traversability labeling pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution([
        FindPackageShare('ugv_self_supervised_traversability'),
        'config', 'traversability.yaml'])
    config_argument = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Traversability pipeline parameter YAML')
    config = LaunchConfiguration('config_file')
    nodes = [
        Node(package='ugv_self_supervised_traversability',
             executable='trajectory_recorder', name='trajectory_recorder',
             output='screen', parameters=[config]),
        Node(package='ugv_self_supervised_traversability',
             executable='footprint_generator', name='footprint_generator',
             output='screen', parameters=[config]),
        Node(package='ugv_self_supervised_traversability',
             executable='traversability_labeler', name='traversability_labeler',
             output='screen', parameters=[config]),
        Node(package='ugv_self_supervised_traversability',
             executable='label_visualizer', name='label_visualizer',
             output='screen', parameters=[config]),
    ]
    return LaunchDescription([config_argument, *nodes])
