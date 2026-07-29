from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_launch = PathJoinSubstitution(
        [FindPackageShare('tem_realsense'), 'launch', 'tem_realsense.launch.py']
    )
    yolo_launch = PathJoinSubstitution(
        [FindPackageShare('yolo_detector_ros2'), 'launch', 'yolo_detector_dual.launch.py']
    )
    safety_launch = PathJoinSubstitution(
        [FindPackageShare('crosswalk_safety_ros2'), 'launch', 'crosswalk_safety.launch.py']
    )
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(realsense_launch)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(yolo_launch)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(safety_launch)),
    ])
