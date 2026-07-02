#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    track_class_ids   = LaunchConfiguration("track_class_ids")
    detection_topic   = LaunchConfiguration("detection_topic")
    depth_topic       = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    # Optional tuning args for the node (feel free to remove if not needed)
    bev_frame            = LaunchConfiguration("bev_frame")
    slop_sec             = LaunchConfiguration("slop_sec")
    min_mask_pixels      = LaunchConfiguration("min_mask_pixels")
    min_cluster_points   = LaunchConfiguration("min_cluster_points")
    max_depth_m          = LaunchConfiguration("max_depth_m")
    sample_points_max    = LaunchConfiguration("sample_points_max")
    dbscan_min_samples   = LaunchConfiguration("dbscan_min_samples")
    dbscan_eps_per_meter = LaunchConfiguration("dbscan_eps_per_meter")
    use_morphology       = LaunchConfiguration("use_morphology")
    use_median_blur      = LaunchConfiguration("use_median_blur_depth")
    median_blur_ksize    = LaunchConfiguration("median_blur_ksize")
    queue_size           = LaunchConfiguration("queue_size")

    return LaunchDescription([
        # Required-ish
        DeclareLaunchArgument(
            "track_class_ids",
            default_value="[0, 2, 5, 7]",
            description="List or string for COCO class IDs to track (e.g., \"[0,2,5,7]\")"
        ),
        DeclareLaunchArgument("detection_topic",   default_value="yolo/detections"),
        DeclareLaunchArgument("depth_topic",       default_value="/zed/zed_node/depth/depth_registered"),
        DeclareLaunchArgument("camera_info_topic", default_value="/zed/zed_node/left/camera_info"),

        # Optional tuning
        DeclareLaunchArgument("bev_frame",            default_value="base_link"),
        DeclareLaunchArgument("slop_sec",             default_value="0.08"),
        DeclareLaunchArgument("min_mask_pixels",      default_value="30"),
        DeclareLaunchArgument("min_cluster_points",   default_value="50"),
        DeclareLaunchArgument("max_depth_m",          default_value="20.0"),
        DeclareLaunchArgument("sample_points_max",    default_value="300"),
        DeclareLaunchArgument("dbscan_min_samples",   default_value="10"),
        DeclareLaunchArgument("dbscan_eps_per_meter", default_value="0.02"),
        DeclareLaunchArgument("use_morphology",       default_value="true"),
        DeclareLaunchArgument("use_median_blur_depth",default_value="true"),
        DeclareLaunchArgument("median_blur_ksize",    default_value="3"),
        DeclareLaunchArgument("queue_size",           default_value="40"),

        Node(
            package="object_depth_tracker",
            executable="object_depth_tracker_node",
            name="object_depth_tracker",
            output="screen",
            parameters=[{
                "track_class_ids":   track_class_ids,     # <-- 런치 인자 반영
                "detection_topic":   detection_topic,
                "depth_topic":       depth_topic,
                "camera_info_topic": camera_info_topic,

                # optional tuning (노드 파라미터와 이름 동일)
                "bev_frame":            bev_frame,
                "slop_sec":             slop_sec,
                "min_mask_pixels":      min_mask_pixels,
                "min_cluster_points":   min_cluster_points,
                "max_depth_m":          max_depth_m,
                "sample_points_max":    sample_points_max,
                "dbscan_min_samples":   dbscan_min_samples,
                "dbscan_eps_per_meter": dbscan_eps_per_meter,
                "use_morphology":       use_morphology,
                "use_median_blur_depth":use_median_blur,
                "median_blur_ksize":    median_blur_ksize,
                "queue_size":           queue_size,
            }],
        ),
    ])
