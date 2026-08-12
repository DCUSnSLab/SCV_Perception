from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
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
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Experimental RGB-D panorama parameter file',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/panorama/image_raw',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'depth_aware_color',
            default_value='true',
            description='Use calibrated depth-aware overlap ownership',
        ),
        DeclareLaunchArgument(
            'panorama_backend',
            default_value='python',
            choices=['python', 'cpp'],
            description='Python/PyTorch default or previous native C++ rollback',
        ),
        DeclareLaunchArgument(
            'use_cuda',
            default_value='true',
            description='Enable the CUDA projection backend when available',
        ),
        Node(
            package='panorama_stitcher',
            executable='rgbd_panorama_torch_node',
            name='panorama_stitcher',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'output_topic': LaunchConfiguration('output_topic'),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                    'depth_aware_color': ParameterValue(
                        LaunchConfiguration('depth_aware_color'),
                        value_type=bool,
                    ),
                    'use_cuda': ParameterValue(
                        LaunchConfiguration('use_cuda'),
                        value_type=bool,
                    ),
                },
            ],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('panorama_backend'), "' == 'python'",
            ])),
        ),
        Node(
            package='panorama_stitcher',
            executable='rgbd_panorama_stitcher_node',
            name='panorama_stitcher',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'output_topic': LaunchConfiguration('output_topic'),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                    'depth_aware_color': ParameterValue(
                        LaunchConfiguration('depth_aware_color'),
                        value_type=bool,
                    ),
                    'use_cuda': ParameterValue(
                        LaunchConfiguration('use_cuda'),
                        value_type=bool,
                    ),
                },
            ],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('panorama_backend'), "' == 'cpp'",
            ])),
        ),
    ])
