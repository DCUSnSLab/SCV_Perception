"""Launch the D555 SegFormer semantic point-cloud node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('terrain_seg'), 'config', 'terrain_seg.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Absolute path to the terrain_seg parameter YAML.'),
        Node(
            package='terrain_seg',
            executable='terrain_seg_node',
            name='terrain_seg_node',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
