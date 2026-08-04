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
        'far_hybrid_panorama.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Far-hybrid panorama parameter file',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/panorama_far_hybrid/image_raw',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
        ),
        Node(
            package='panorama_stitcher',
            executable='far_hybrid_panorama_stitcher_node',
            name='panorama_far_hybrid_stitcher',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'output_topic': LaunchConfiguration('output_topic'),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
