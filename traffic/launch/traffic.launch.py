from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def default_model_path() -> str:
    suffix = Path('src/perception/traffic_light/model/best.pt')
    for parent in Path(__file__).resolve().parents:
        candidate = parent / suffix
        if candidate.is_file():
            return str(candidate)
    return str(Path.home() / 'SSC' / suffix)


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument('model_path', default_value=default_model_path()),
        DeclareLaunchArgument('image_topic', default_value='/panorama/image_raw'),
        DeclareLaunchArgument('device', default_value='auto'),
        DeclareLaunchArgument('conf_threshold', default_value='0.20'),
        DeclareLaunchArgument('image_size', default_value='416'),
    ]
    detector = Node(
        package='traffic',
        executable='detector',
        name='traffic_detector',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'device': LaunchConfiguration('device'),
            'conf_threshold': ParameterValue(
                LaunchConfiguration('conf_threshold'), value_type=float
            ),
            'image_size': ParameterValue(LaunchConfiguration('image_size'), value_type=int),
        }],
    )
    return LaunchDescription([*arguments, detector])
