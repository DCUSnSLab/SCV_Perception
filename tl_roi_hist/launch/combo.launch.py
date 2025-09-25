#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ---- 공통(선택): 네임스페이스 ----
    ns_arg = DeclareLaunchArgument(
        'ns', default_value='', description='Optional ROS namespace'
    )

    # ---- tl_roi_hist 기본 경로(패키지 내부 파일 사용) ----
    tl_share = get_package_share_directory('tl_roi_hist')
    tl_default_params = os.path.join(tl_share, 'config', 'tl_crop_only.param.yaml')
    tl_default_model  = os.path.join(tl_share, 'model', 'yolo11s.pt')

    # ---- tl_roi_hist 인자 ----
    tl_model_arg  = DeclareLaunchArgument('model_path',  default_value=tl_default_model)
    tl_topic_arg  = DeclareLaunchArgument('image_topic', default_value='/zed/zed_node/left/image_rect_color')
    tl_params_arg = DeclareLaunchArgument('tl_params_file', default_value=tl_default_params)

    tl_node = Node(
        package='tl_roi_hist',
        executable='tl_crop_only',
        name='tl_crop_only',
        output='screen',
        parameters=[
            LaunchConfiguration('tl_params_file'),
            {'model_path':  LaunchConfiguration('model_path')},
            {'image_topic': LaunchConfiguration('image_topic')},
        ],
        # 필요 시 remappings=[('/tl/state','/my/state')] 등 추가
    )

    # ---- ssc_avoid_obstacles 인자 ----
    ssc_min_x  = DeclareLaunchArgument('min_x',           default_value='2.0')
    ssc_max_x  = DeclareLaunchArgument('max_x',           default_value='3.5')
    ssc_min_y  = DeclareLaunchArgument('min_y',           default_value='-1.5')
    ssc_max_y  = DeclareLaunchArgument('max_y',           default_value='1.5')
    ssc_min_z  = DeclareLaunchArgument('min_z',           default_value='-0.5')
    ssc_max_z  = DeclareLaunchArgument('max_z',           default_value='1.0')
    ssc_minsz  = DeclareLaunchArgument('min_cluster_size', default_value='1300')
    ssc_debug  = DeclareLaunchArgument('debug',           default_value='False')

    ssc_node = Node(
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
            'min_cluster_size': LaunchConfiguration('min_cluster_size'),
            'debug': LaunchConfiguration('debug'),
        }],
        # 필요 시 remappings=[('입력토픽','다른토픽')] 추가
    )

    group = GroupAction([
        PushRosNamespace(LaunchConfiguration('ns')),
        tl_node,
        ssc_node,
    ])

    return LaunchDescription([
        ns_arg,
        tl_model_arg, tl_topic_arg, tl_params_arg,
        ssc_min_x, ssc_max_x, ssc_min_y, ssc_max_y, ssc_min_z, ssc_max_z, ssc_minsz, ssc_debug,
        group
    ])
