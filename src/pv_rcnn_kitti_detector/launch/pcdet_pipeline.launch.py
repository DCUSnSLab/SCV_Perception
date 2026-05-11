from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    tracker_type = LaunchConfiguration('tracker_type')
    enable_csv_logging = LaunchConfiguration('enable_csv_logging')
    sequence_id = LaunchConfiguration('sequence_id')
    detection_output_dir = LaunchConfiguration('detection_output_dir')
    track_output_dir = LaunchConfiguration('track_output_dir')

    return LaunchDescription([
        DeclareLaunchArgument(
            'tracker_type',
            default_value='kitti',
            description='Tracker mode: basic, ab3dmot, or kitti',
        ),
        DeclareLaunchArgument(
            'enable_csv_logging',
            default_value='false',
            description='Enable CSV export in jay_tracker',
        ),
        DeclareLaunchArgument(
            'sequence_id',
            default_value='seq01',
            description='Sequence id used in exported CSV filenames',
        ),
        DeclareLaunchArgument(
            'detection_output_dir',
            default_value='results/detections',
            description='Directory for detection CSV export',
        ),
        DeclareLaunchArgument(
            'track_output_dir',
            default_value='results/tracks',
            description='Directory for track CSV export',
        ),
        Node(
            package='pcd_ground_filter',
            executable='ground_removal_node',
            name='ground_removal_node',
            output='screen',
        ),
        Node(
            package='pv_rcnn_kitti_detector',
            executable='kitti_detector_node',
            name='kitti_detector_node',
            output='screen',
        ),
        Node(
            package='pcdet_tracker',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", tracker_type, "' == 'basic'"])
            ),
        ),
        Node(
            package='pcdet_tracker',
            executable='ab3dmot_node',
            name='ab3dmot_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", tracker_type, "' == 'ab3dmot'"])
            ),
        ),
        Node(
            package='pcdet_tracker',
            executable='jay_tracker',
            name='jay_tracker',
            output='screen',
            arguments=[
                '--enable_csv_logging', enable_csv_logging,
                '--sequence_id', sequence_id,
                '--detection_output_dir', detection_output_dir,
                '--track_output_dir', track_output_dir,
            ],
            condition=IfCondition(
                PythonExpression(["'", tracker_type, "' == 'kitti'"])
            ),
        ),
    ])
