from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/scv/SCV/src/perception/yolov8n.pt',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/realsense_1/color/image_raw/compressed',
        ),
        DeclareLaunchArgument(
            'use_compressed_image',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/realsense_1/aligned_depth_to_color/image_raw',
        ),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/perception/detections',
        ),
        DeclareLaunchArgument(
            'annotated_image_topic',
            default_value='/perception/annotated_image',
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
        Node(
            package='yolo_detector_ros2',
            executable='yolo_detector_node',
            name='yolo_detector_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'use_compressed_image': LaunchConfiguration('use_compressed_image'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'annotated_image_topic': LaunchConfiguration('annotated_image_topic'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'device': LaunchConfiguration('device'),
                'imgsz': LaunchConfiguration('imgsz'),
                'frame_skip': LaunchConfiguration('frame_skip'),
                'use_half': LaunchConfiguration('use_half'),
                'publish_every_n_frames': LaunchConfiguration('publish_every_n_frames'),
            }],
        ),
    ])
