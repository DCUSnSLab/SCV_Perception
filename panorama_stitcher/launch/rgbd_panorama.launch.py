from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('panorama_stitcher'),
        'config',
        'rgbd_panorama.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='panorama_stitcher',
            executable='rgbd_panorama_torch_node',
            name='panorama_stitcher',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'), value_type=bool),
                },
            ],
        ),
    ])
