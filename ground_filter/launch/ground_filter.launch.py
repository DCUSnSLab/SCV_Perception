"""Launch the VNR ground filter (SCV bag defaults)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("ground_filter"),
        "config",
        "ground_filter.param.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use /clock (bag playback)",
            ),
            Node(
                package="ground_filter",
                executable="ground_filter_node",
                name="ground_filter_node",
                output="screen",
                parameters=[
                    config_path,
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
