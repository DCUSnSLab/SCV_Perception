#!/usr/bin/env python3
"""Run the calibrated panorama on Gazebo's upright RGB-D streams."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    bounded_math_threads = {
        'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'NUMEXPR_NUM_THREADS': '2',
    }
    parameters = [
        PathJoinSubstitution([
            FindPackageShare('panorama_stitcher'),
            'config',
            'rgbd_panorama.yaml',
        ]),
        {
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'left_input_image_rotated_180': False,
            'enable_exposure_compensation': False,
        },
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='panorama_stitcher',
            executable='rgbd_panorama_torch_node',
            name='panorama_stitcher',
            output='screen',
            parameters=parameters,
            additional_env=bounded_math_threads,
        ),
    ])
