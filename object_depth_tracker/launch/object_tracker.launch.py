#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    track_class_ids = LaunchConfiguration("track_class_ids")
    detection_topic = LaunchConfiguration("detection_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    filter_type = LaunchConfiguration("filter_type")

    return LaunchDescription([
        DeclareLaunchArgument(
            "track_class_ids", 
            default_value="[0, 1, 2, 3, 5, 6, 7, 28]",
            description="Class IDs to track (person=0, car=2, bus=5, truck=7)"
        ),
        DeclareLaunchArgument(
            "detection_topic", 
            default_value="yolo/detections"
        ),
        DeclareLaunchArgument(
            "depth_topic", 
            default_value="/zed/zed_node/depth/depth_registered"
        ),
        DeclareLaunchArgument(
            "camera_info_topic", 
            default_value="/zed/zed_node/left/camera_info"
        ),
        DeclareLaunchArgument(
            "filter_type", 
            default_value="kalman_6d_v2",
            description="Filter type: centroid, kalman_6d_v2"
        ),

        Node(
            package="object_depth_tracker",
            executable="object_depth_tracker_node",
            name="object_depth_tracker",
            output="screen",
            parameters=[{
                "track_class_ids": [0, 1, 2, 3, 5, 6, 7, 28],
                "detection_topic": detection_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
                "filter_type": filter_type,
            }],
        ),
    ])