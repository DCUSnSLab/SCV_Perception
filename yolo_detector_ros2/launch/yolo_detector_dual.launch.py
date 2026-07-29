from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def detector_node(name, image_topic, depth_topic, detections_topic, annotated_topic):
    return Node(
        package='yolo_detector_ros2',
        executable='yolo_detector_node',
        name=name,
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'image_topic': image_topic,
            'use_compressed_image': True,
            'depth_topic': depth_topic,
            'detections_topic': detections_topic,
            'annotated_image_topic': annotated_topic,
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'device': LaunchConfiguration('device'),
            'imgsz': LaunchConfiguration('imgsz'),
            'frame_skip': LaunchConfiguration('frame_skip'),
            'use_half': LaunchConfiguration('use_half'),
            'publish_every_n_frames': LaunchConfiguration('publish_every_n_frames'),
        }],
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/scv/SCV/src/perception/yolov8n.pt',
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.35',
        ),
        DeclareLaunchArgument(
            'device',
            default_value='auto',
        ),
        DeclareLaunchArgument(
            'imgsz',
            default_value='640',
        ),
        DeclareLaunchArgument(
            'frame_skip',
            default_value='0',
        ),
        DeclareLaunchArgument(
            'use_half',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'publish_every_n_frames',
            default_value='1',
        ),
        detector_node(
            'left_yolo_detector',
            '/realsense_2/d435i_left/color/image_raw/compressed',
            '/realsense_2/d435i_left/aligned_depth_to_color/image_raw',
            '/perception/left/detections',
            '/perception/left/annotated_image',
        ),
        detector_node(
            'right_yolo_detector',
            '/realsense_1/d435i_right/color/image_raw/compressed',
            '/realsense_1/d435i_right/aligned_depth_to_color/image_raw',
            '/perception/right/detections',
            '/perception/right/annotated_image',
        ),
    ])
