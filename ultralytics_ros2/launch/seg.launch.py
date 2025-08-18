#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- launch args ----
    yolo_model = LaunchConfiguration("yolo_model")
    input_topic = LaunchConfiguration("input_topic")
    result_image_topic = LaunchConfiguration("result_image_topic")
    conf_thres = LaunchConfiguration("conf_thres")
    iou_thres = LaunchConfiguration("iou_thres")
    max_det = LaunchConfiguration("max_det")
    device = LaunchConfiguration("device")
    classes = LaunchConfiguration("classes")  # 문자열로 받았다가 파라미터 서버에 그대로 넘김

    return LaunchDescription([
        DeclareLaunchArgument("yolo_model", default_value="yolo11n-seg.pt",
                              description="Model file name under share/ultralytics_ros2/model or absolute path"),
        DeclareLaunchArgument("input_topic", default_value="/zed/zed_node/left/image_rect_color"),
        DeclareLaunchArgument("result_image_topic", default_value="yolo/seg_image"),
        DeclareLaunchArgument("conf_thres", default_value="0.25"),
        DeclareLaunchArgument("iou_thres", default_value="0.45"),
        DeclareLaunchArgument("max_det", default_value="300"),
        DeclareLaunchArgument("device", default_value=""),
        DeclareLaunchArgument("classes", default_value=""),  # 예: "0,1" (문자열)

        Node(
            package="ultralytics_ros2",
            executable="seg_run",
            name="ultralytics_seg_node",
            output="screen",
            parameters=[{
                "yolo_model": yolo_model,
                "input_topic": input_topic,
                "result_image_topic": result_image_topic,
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
                "max_det": max_det,
                "device": device,
                "classes": classes,   # 문자열 그대로; 노드에서 가공하고 싶으면 파싱 로직 추가
                "result_conf": True,
                "result_line_width": 3,
                "result_font_size": 16,
                "result_font": "Arial.ttf",
                "result_labels": True,
                "result_boxes": True,
            }],
        ),
    ])
