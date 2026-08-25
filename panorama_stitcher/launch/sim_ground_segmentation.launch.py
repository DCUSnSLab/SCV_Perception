#!/usr/bin/env python3
"""Run the real-car front and rear ground segmentation on simulated topics."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_rear = LaunchConfiguration('enable_rear')
    common_parameters = {
        'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
    }
    bounded_math_threads = {
        'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'NUMEXPR_NUM_THREADS': '2',
    }

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('enable_rear', default_value='true'),
        Node(
            package='panorama_stitcher',
            executable='panorama_ground_segmentation_node',
            name='panorama_ground_segmentation',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('panorama_stitcher'),
                    'config',
                    'panorama_ground_segmentation.yaml',
                ]),
                common_parameters,
            ],
            additional_env=bounded_math_threads,
        ),
        Node(
            package='panorama_stitcher',
            executable='panorama_ground_segmentation_node',
            name='rear_ground_segmentation',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('panorama_stitcher'),
                    'config',
                    'rear_ground_segmentation.yaml',
                ]),
                common_parameters,
                {
                    # Gazebo's depth camera publishes the same XYZRGB cloud
                    # without the RealSense wrapper's extra /color namespace.
                    # Keep the real-car output contract unchanged.
                    'input_topic': '/rear/rear/depth/points',
                },
            ],
            additional_env=bounded_math_threads,
            condition=IfCondition(enable_rear),
        ),
    ])
