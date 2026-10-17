from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = Path.home()
    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/velodyne_points"),
        DeclareLaunchArgument("detection_topic", default_value="/detected_objects_3d"),
        DeclareLaunchArgument("marker_topic", default_value="/detections/visual_markers"),
        DeclareLaunchArgument("score_thresh", default_value="0.65"),
        DeclareLaunchArgument("reliable_input", default_value="false"),
        DeclareLaunchArgument("z_offset", default_value="0.0"),
        DeclareLaunchArgument("perf_output_dir", default_value="results/perf"),
        DeclareLaunchArgument(
            "openpcdet_path",
            default_value=str(home / "OpenPCDet-restored"),
            description="Path containing the OpenPCDet pcdet Python package",
        ),
        DeclareLaunchArgument(
            "ckpt_file",
            default_value=str(
                home / "pcdet_ros2_ws/models/pointpillars_coda.pth"),
        ),
        DeclareLaunchArgument(
            "cuda_ops_path",
            default_value=str(home / "OpenPCDet-restored/pcdet/ops"),
            description="CUDA 12 OpenPCDet extension directory",
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
                    "reliable_input": LaunchConfiguration("reliable_input"),
                    "z_offset": LaunchConfiguration("z_offset"),
                    "perf_output_dir": LaunchConfiguration("perf_output_dir"),
                    "ckpt_file": LaunchConfiguration("ckpt_file"),
                }
            ],
        ),
    ])
