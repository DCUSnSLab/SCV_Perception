#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='/yolo/detections',
        description='Topic for detection results (DetectionArray)'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', 
        default_value='/zed/zed_node/depth/depth_registered',
        description='Topic for depth images'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/zed/zed_node/depth/camera_info', 
        description='Topic for camera info'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/object_3d_tracker/markers',
        description='Topic for output markers'
    )
    
    # Processing parameters
    min_mask_pixels_arg = DeclareLaunchArgument(
        'min_mask_pixels',
        default_value='20',
        description='Minimum number of mask pixels'
    )
    
    min_depth_arg = DeclareLaunchArgument(
        'min_depth',
        default_value='2.0',
        description='Minimum depth value in meters'
    )
    
    max_depth_arg = DeclareLaunchArgument(
        'max_depth',
        default_value='10.0',
        description='Maximum depth value in meters'
    )
    
    marker_scale_arg = DeclareLaunchArgument(
        'marker_scale',
        default_value='0.2',
        description='Scale of sphere markers'
    )
    
    use_morphology_arg = DeclareLaunchArgument(
        'use_morphology',
        default_value='true',
        description='Whether to apply morphological operations to masks'
    )
    
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug',
        default_value='true',
        description='Whether to publish debug information'
    )
    
    # Velocity tracking parameters
    velocity_history_size_arg = DeclareLaunchArgument(
        'velocity_history_size',
        default_value='10',
        description='History size for velocity calculation'
    )
    
    velocity_smoothing_window_arg = DeclareLaunchArgument(
        'velocity_smoothing_window',
        default_value='3',
        description='Smoothing window for velocity calculation'
    )
    
    min_velocity_threshold_arg = DeclareLaunchArgument(
        'min_velocity_threshold',
        default_value='0.05',
        description='Minimum velocity threshold (m/s)'
    )
    
    velocity_outlier_threshold_arg = DeclareLaunchArgument(
        'velocity_outlier_threshold',
        default_value='10.0',
        description='Velocity outlier threshold (m/s)'
    )
    
    publish_velocity_markers_arg = DeclareLaunchArgument(
        'publish_velocity_markers',
        default_value='true',
        description='Whether to publish velocity arrows'
    )
    
    velocity_arrow_scale_arg = DeclareLaunchArgument(
        'velocity_arrow_scale',
        default_value='1.0',
        description='Scale factor for velocity arrows'
    )
    
    # Object history parameters
    enable_history_arg = DeclareLaunchArgument(
        'enable_history',
        default_value='true',
        description='Enable object history storage and visualization'
    )
    
    history_max_size_arg = DeclareLaunchArgument(
        'history_max_size',
        default_value='1000',
        description='Maximum number of object records to store'
    )
    
    show_all_points_arg = DeclareLaunchArgument(
        'show_all_points',
        default_value='true',
        description='Show all detected points as white dots'
    )
    
    show_trajectories_arg = DeclareLaunchArgument(
        'show_trajectories',
        default_value='true',
        description='Show object trajectories as colored lines'
    )
    
    show_heatmap_arg = DeclareLaunchArgument(
        'show_heatmap',
        default_value='false',
        description='Show detection density heatmap'
    )
    
    # Node
    object_3d_tracker_node = Node(
        package='object_3d_tracker',
        executable='object_3d_tracker_node',
        name='object_3d_tracker',
        output='screen',
        parameters=[{
            'detection_topic': LaunchConfiguration('detection_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'min_mask_pixels': LaunchConfiguration('min_mask_pixels'),
            'min_depth': LaunchConfiguration('min_depth'),
            'max_depth': LaunchConfiguration('max_depth'), 
            'marker_scale': LaunchConfiguration('marker_scale'),
            'use_morphology': LaunchConfiguration('use_morphology'),
            'publish_debug': LaunchConfiguration('publish_debug'),
            # Velocity parameters
            'velocity_history_size': LaunchConfiguration('velocity_history_size'),
            'velocity_smoothing_window': LaunchConfiguration('velocity_smoothing_window'),
            'min_velocity_threshold': LaunchConfiguration('min_velocity_threshold'),
            'velocity_outlier_threshold': LaunchConfiguration('velocity_outlier_threshold'),
            'publish_velocity_markers': LaunchConfiguration('publish_velocity_markers'),
            'velocity_arrow_scale': LaunchConfiguration('velocity_arrow_scale'),
            # History parameters
            'enable_history': LaunchConfiguration('enable_history'),
            'history_max_size': LaunchConfiguration('history_max_size'),
            'show_all_points': LaunchConfiguration('show_all_points'),
            'show_trajectories': LaunchConfiguration('show_trajectories'),
            'show_heatmap': LaunchConfiguration('show_heatmap'),
            # Other parameters
            'queue_size': 10,
            'slop': 0.1,
            'depth_filter_kernel_size': 3,
            'text_height': 0.3,
            'history_max_age': 300.0,
        }],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        detection_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        output_topic_arg,
        min_mask_pixels_arg,
        min_depth_arg,
        max_depth_arg,
        marker_scale_arg,
        use_morphology_arg,
        publish_debug_arg,
        velocity_history_size_arg,
        velocity_smoothing_window_arg,
        min_velocity_threshold_arg,
        velocity_outlier_threshold_arg,
        publish_velocity_markers_arg,
        velocity_arrow_scale_arg,
        enable_history_arg,
        history_max_size_arg,
        show_all_points_arg,
        show_trajectories_arg,
        show_heatmap_arg,
        object_3d_tracker_node,
    ])