from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path

from mando_tools.workspace_paths import default_runtime_image_topic


def _default_tl_model() -> str:
    launch_file = Path(__file__).resolve()
    for root in [launch_file.parent, *launch_file.parents]:
        candidate = root / 'model' / 'best.pt'
        if candidate.exists():
            return str(candidate)
        if root.name == 'traffic_light':
            return str(candidate)
    return 'best.pt'

def generate_launch_description() -> LaunchDescription:
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=_default_tl_model(),
        description='Path to the YOLO model checkpoint.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value=default_runtime_image_topic(),
        description='Image topic consumed by the model.',
    )
    annotated_topic_arg = DeclareLaunchArgument(
        'annotated_topic',
        default_value='/mando/yolo/annotated',
        description='Output topic for images with detections drawn.',
    )
    detections_topic_arg = DeclareLaunchArgument(
        'detections_topic',
        default_value='/mando/yolo/detections',
        description='Output topic for Detection2DArray messages.',
    )
    conf_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.25',
        description='Minimum detection confidence.',
    )
    image_size_arg = DeclareLaunchArgument(
        'image_size',
        default_value='640',
        description='Inference image size. Lower is faster, higher is more accurate.',
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda:0',
        description='Inference device. Examples: cuda:0, cpu, auto.',
    )
    fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='15.0',
        description='Maximum processed frames per second.',
    )
    draw_labels_arg = DeclareLaunchArgument(
        'draw_labels',
        default_value='true',
        description='Draw class labels on the annotated image.',
    )
    draw_confidence_arg = DeclareLaunchArgument(
        'draw_confidence',
        default_value='true',
        description='Draw confidence scores on the annotated image.',
    )

    validator = Node(
        package='mando_tools',
        executable='mando_yolo_validate',
        name='mando_yolo_validator',
        output='screen',
        parameters=[
            {
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'annotated_topic': LaunchConfiguration('annotated_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'conf_threshold': ParameterValue(
                    LaunchConfiguration('conf_threshold'),
                    value_type=float,
                ),
                'image_size': ParameterValue(
                    LaunchConfiguration('image_size'),
                    value_type=int,
                ),
                'device': LaunchConfiguration('device'),
                'max_fps': ParameterValue(
                    LaunchConfiguration('max_fps'),
                    value_type=float,
                ),
                'draw_labels': ParameterValue(
                    LaunchConfiguration('draw_labels'),
                    value_type=bool,
                ),
                'draw_confidence': ParameterValue(
                    LaunchConfiguration('draw_confidence'),
                    value_type=bool,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            image_topic_arg,
            annotated_topic_arg,
            detections_topic_arg,
            conf_arg,
            image_size_arg,
            device_arg,
            fps_arg,
            draw_labels_arg,
            draw_confidence_arg,
            validator,
        ]
    )
