from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/no_ground_points"),
        DeclareLaunchArgument("detection_topic", default_value="/detected_objects_3d"),
        DeclareLaunchArgument("marker_topic", default_value="/detections/visual_markers"),
        DeclareLaunchArgument("score_thresh", default_value="0.65"),
        DeclareLaunchArgument("z_offset", default_value="0.0"),
        DeclareLaunchArgument("perf_output_dir", default_value="results/perf"),
        DeclareLaunchArgument(
            "openpcdet_path",
            default_value="/home/scv/SCV_Perception/third_party/OpenPCDet",
            description="Path containing the OpenPCDet pcdet Python package",
        ),
        DeclareLaunchArgument(
            "ckpt_file",
            default_value="/home/scv/SCV/src/perception/jay_tracker/models/pointpillars_coda.pth",
        ),
        DeclareLaunchArgument(
            "cuda_ops_path",
            default_value=(
                "/home/scv/SCV/src/perception/jay_tracker/third_party/"
                "OpenPCDet/pcdet/ops"
            ),
            description="CUDA 12 OpenPCDet extension directory",
        ),
        DeclareLaunchArgument("run_tracker", default_value="true"),
        # behavior_predictor is an optional downstream package and is not part
        # of this workspace. Keep the detector/tracker launch self-contained.
        DeclareLaunchArgument("run_behavior_predictor", default_value="false"),
        DeclareLaunchArgument("bbox_topic", default_value="/pcdet/coda_tracks"),
        DeclareLaunchArgument("tracked_topic", default_value="/tracked_objects_3d"),
        DeclareLaunchArgument("prediction_marker_topic", default_value="/behavior/prediction_markers"),
        DeclareLaunchArgument("behavior_history_seconds", default_value="1.5"),
        DeclareLaunchArgument("behavior_prediction_horizon", default_value="3.0"),
        DeclareLaunchArgument("behavior_prediction_step", default_value="0.5"),
        DeclareLaunchArgument("behavior_stationary_speed_thresh", default_value="0.35"),
        DeclareLaunchArgument("behavior_turn_yaw_rate_thresh", default_value="0.20"),
        DeclareLaunchArgument("behavior_enable_csv_logging", default_value="false"),
        DeclareLaunchArgument("behavior_sequence_id", default_value="seq01"),
        DeclareLaunchArgument("behavior_prediction_output_dir", default_value="results/behavior_predictions"),
        DeclareLaunchArgument("odom_topic", default_value="/odometry/wheel"),
        DeclareLaunchArgument("imu_topic", default_value="/vectornav/imu"),
        DeclareLaunchArgument(
            "adaptive_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("pcdet_tracker"), "config",
                "adaptive_tracking.yaml",
            ]),
        ),
        Node(
            package="pointpillars_coda_detector",
            executable="coda_pointpillar_node",
            name="coda_pointpillar_node",
            output="screen",
            additional_env={
                "OPENPCDET_PATH": LaunchConfiguration("openpcdet_path"),
                "OPENPCDET_CUDA_OPS_PATH": LaunchConfiguration(
                    "cuda_ops_path"),
            },
            parameters=[
                {
                    "input_topic": LaunchConfiguration("input_topic"),
                    "detection_topic": LaunchConfiguration("detection_topic"),
                    "marker_topic": LaunchConfiguration("marker_topic"),
                    "score_thresh": LaunchConfiguration("score_thresh"),
                    "z_offset": LaunchConfiguration("z_offset"),
                    "perf_output_dir": LaunchConfiguration("perf_output_dir"),
                    "ckpt_file": LaunchConfiguration("ckpt_file"),
                }
            ],
        ),
        Node(
            package="pcdet_tracker",
            executable="jay_tracker",
            name="jay_tracker",
            output="screen",
            arguments=[
                "--adaptive_config", LaunchConfiguration("adaptive_config"),
                "--detection_topic", LaunchConfiguration("detection_topic"),
                "--tracked_topic", LaunchConfiguration("tracked_topic"),
                "--bbox_topic", LaunchConfiguration("bbox_topic"),
                "--odom_topic", LaunchConfiguration("odom_topic"),
                "--imu_topic", LaunchConfiguration("imu_topic"),
            ],
            condition=IfCondition(LaunchConfiguration("run_tracker")),
        ),
        Node(
            package="behavior_predictor",
            executable="behavior_predictor_node",
            name="behavior_predictor",
            output="screen",
            arguments=[
                "--tracked_topic", LaunchConfiguration("tracked_topic"),
                "--prediction_marker_topic", LaunchConfiguration("prediction_marker_topic"),
                "--history_seconds", LaunchConfiguration("behavior_history_seconds"),
                "--prediction_horizon", LaunchConfiguration("behavior_prediction_horizon"),
                "--prediction_step", LaunchConfiguration("behavior_prediction_step"),
                "--stationary_speed_thresh", LaunchConfiguration("behavior_stationary_speed_thresh"),
                "--turn_yaw_rate_thresh", LaunchConfiguration("behavior_turn_yaw_rate_thresh"),
                "--enable_csv_logging", LaunchConfiguration("behavior_enable_csv_logging"),
                "--sequence_id", LaunchConfiguration("behavior_sequence_id"),
                "--prediction_output_dir", LaunchConfiguration("behavior_prediction_output_dir"),
            ],
            condition=IfCondition(LaunchConfiguration("run_behavior_predictor")),
        ),
    ])
