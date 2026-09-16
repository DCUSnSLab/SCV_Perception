from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def default_model_path() -> str:
    suffix = Path('analysis/jin_yolo/yolo26m_sign_ceiling_20260910/best.pt')
    for parent in Path(__file__).resolve().parents:
        candidate = parent / suffix
        if candidate.is_file():
            return str(candidate)
    return str(Path.home() / 'SSC' / suffix)


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument('model_path', default_value=default_model_path()),
        DeclareLaunchArgument('image_topic', default_value='/panorama/image_raw'),
        DeclareLaunchArgument('annotated_topic', default_value='/sign_truck/annotated'),
        DeclareLaunchArgument('device', default_value='auto'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('iou_threshold', default_value='0.70'),
        DeclareLaunchArgument('image_size', default_value='1280'),
        DeclareLaunchArgument('current_lane', default_value='auto'),
        DeclareLaunchArgument('left_lane_anchor_ratio', default_value='0.35'),
        DeclareLaunchArgument('right_lane_anchor_ratio', default_value='0.65'),
        DeclareLaunchArgument('ego_anchor_ratio', default_value='0.50'),
        DeclareLaunchArgument('lane_match_max_ratio', default_value='0.22'),
    ]
    detector = Node(
        package='sign_truck',
        executable='detector',
        name='sign_truck_detector',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'image_topic': LaunchConfiguration('image_topic'),
            'annotated_topic': LaunchConfiguration('annotated_topic'),
            'device': LaunchConfiguration('device'),
            'conf_threshold': ParameterValue(LaunchConfiguration('conf_threshold'), value_type=float),
            'iou_threshold': ParameterValue(LaunchConfiguration('iou_threshold'), value_type=float),
            'image_size': ParameterValue(LaunchConfiguration('image_size'), value_type=int),
            'current_lane': LaunchConfiguration('current_lane'),
            'left_lane_anchor_ratio': ParameterValue(
                LaunchConfiguration('left_lane_anchor_ratio'), value_type=float
            ),
            'right_lane_anchor_ratio': ParameterValue(
                LaunchConfiguration('right_lane_anchor_ratio'), value_type=float
            ),
            'ego_anchor_ratio': ParameterValue(
                LaunchConfiguration('ego_anchor_ratio'), value_type=float
            ),
            'lane_match_max_ratio': ParameterValue(
                LaunchConfiguration('lane_match_max_ratio'), value_type=float
            ),
            'roi_top_ratio': 0.40,
            'roi_left_ratio': 0.30,
            'roi_right_ratio': 0.70,
        }],
    )
    return LaunchDescription([*arguments, detector])
