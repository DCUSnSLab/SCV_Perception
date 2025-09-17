#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    min_x_arg = DeclareLaunchArgument(
        'min_x', default_value='4.0', description='ROI min X (m)'
    )
    max_x_arg = DeclareLaunchArgument(
        'max_x', default_value='8.0', description='ROI max X (m)'
    )
    min_y_arg = DeclareLaunchArgument(
        'min_y', default_value='-1.0', description='ROI min Y (m)'
    )
    max_y_arg = DeclareLaunchArgument(
        'max_y', default_value='1.0', description='ROI max Y (m)'
    )
    min_z_arg = DeclareLaunchArgument(
        'min_z', default_value='0.0', description='ROI min Z (m)'
    )
    max_z_arg = DeclareLaunchArgument(
        'max_z', default_value='1.5', description='ROI max Z (m)'
    )
    num_points_arg = DeclareLaunchArgument(
        'num_of_points', default_value='300', description='장애물 판단 기준 점 개수'
    )
    debug = DeclareLaunchArgument(
        'debug', default_value='False', description='로그 출력 여부'
    )

    send_stop_flag_node = Node(
        package='ssc_avoid_obstacles',
        executable='send_stop_flag',
        name='send_stop_flag',
        output='screen',
        parameters=[{
            'min_x': LaunchConfiguration('min_x'),
            'max_x': LaunchConfiguration('max_x'),
            'min_y': LaunchConfiguration('min_y'),
            'max_y': LaunchConfiguration('max_y'),
            'min_z': LaunchConfiguration('min_z'),
            'max_z': LaunchConfiguration('max_z'),
            'num_of_points': LaunchConfiguration('num_of_points'),
            'debug': LaunchConfiguration('debug'),
        }],
    )

    return LaunchDescription([
        min_x_arg, max_x_arg, min_y_arg, max_y_arg, min_z_arg, max_z_arg, num_points_arg, debug,
        send_stop_flag_node,
    ])
