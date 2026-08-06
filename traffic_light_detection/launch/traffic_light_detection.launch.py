#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_model = os.path.join(
        get_package_share_directory('traffic_light_detection'),
        'weights',
        'best.pt',
    )

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument(
            'image_topic', default_value='/panorama/image_raw'),
        DeclareLaunchArgument('show_windows', default_value='false'),
        DeclareLaunchArgument('max_fps', default_value='15.0'),
        DeclareLaunchArgument('detector_device', default_value='cpu'),
        Node(
            package='traffic_light_detection',
            executable='traffic_light_detector',
            name='traffic_light_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'show_windows': ParameterValue(
                    LaunchConfiguration('show_windows'), value_type=bool),
                'max_fps': ParameterValue(
                    LaunchConfiguration('max_fps'), value_type=float),
                'detector_device': LaunchConfiguration('detector_device'),
            }],
        ),
    ])
