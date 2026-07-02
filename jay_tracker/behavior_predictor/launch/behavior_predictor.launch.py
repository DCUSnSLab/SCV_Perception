from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("tracked_topic", default_value="/tracked_objects_3d"),
        DeclareLaunchArgument("prediction_marker_topic", default_value="/behavior/prediction_markers"),
        DeclareLaunchArgument("history_seconds", default_value="1.5"),
        DeclareLaunchArgument("prediction_horizon", default_value="3.0"),
        DeclareLaunchArgument("prediction_step", default_value="0.5"),
        DeclareLaunchArgument("stationary_speed_thresh", default_value="0.35"),
        DeclareLaunchArgument("turn_yaw_rate_thresh", default_value="0.20"),
        DeclareLaunchArgument("enable_csv_logging", default_value="false"),
        DeclareLaunchArgument("sequence_id", default_value="seq01"),
        DeclareLaunchArgument("prediction_output_dir", default_value="results/behavior_predictions"),
        Node(
            package="behavior_predictor",
            executable="behavior_predictor_node",
            name="behavior_predictor",
            output="screen",
            arguments=[
                "--tracked_topic", LaunchConfiguration("tracked_topic"),
                "--prediction_marker_topic", LaunchConfiguration("prediction_marker_topic"),
                "--history_seconds", LaunchConfiguration("history_seconds"),
                "--prediction_horizon", LaunchConfiguration("prediction_horizon"),
                "--prediction_step", LaunchConfiguration("prediction_step"),
                "--stationary_speed_thresh", LaunchConfiguration("stationary_speed_thresh"),
                "--turn_yaw_rate_thresh", LaunchConfiguration("turn_yaw_rate_thresh"),
                "--enable_csv_logging", LaunchConfiguration("enable_csv_logging"),
                "--sequence_id", LaunchConfiguration("sequence_id"),
                "--prediction_output_dir", LaunchConfiguration("prediction_output_dir"),
            ],
        ),
    ])
