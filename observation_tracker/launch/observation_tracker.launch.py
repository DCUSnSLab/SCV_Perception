"""Launch the standalone observation-adaptive tracker."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_parameters = PathJoinSubstitution([
        FindPackageShare("observation_tracker"), "config",
        "observation_tracker.yaml"])
    arguments = [
        DeclareLaunchArgument("params_file", default_value=default_parameters),
        DeclareLaunchArgument("detection_topic", default_value="/detected_objects_3d"),
        DeclareLaunchArgument("tracked_topic", default_value="/tracked_objects_3d"),
        DeclareLaunchArgument("bbox_topic", default_value="/pcdet/observation_tracks"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("imu_topic", default_value="/vectornav/imu"),
        DeclareLaunchArgument("sequence_id", default_value="seq01"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("enable_lidar_observation", default_value="true"),
        DeclareLaunchArgument("enable_camera_observation", default_value="true"),
        DeclareLaunchArgument("enable_depth_consistency", default_value="true"),
        DeclareLaunchArgument("enable_odometry", default_value="true"),
        DeclareLaunchArgument("enable_imu", default_value="true"),
        DeclareLaunchArgument("enable_csv_logging", default_value="true"),
        DeclareLaunchArgument(
            "enable_debug_visualization", default_value="false"),
        DeclareLaunchArgument(
            "enable_fused_detection_visualization", default_value="false"),
        DeclareLaunchArgument("lidar_points_topic", default_value="/velodyne_points"),
        DeclareLaunchArgument(
            "camera_rgb_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("camera_detection_topic", default_value="/yolo/detections"),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/camera/camera/color/camera_info"),
        DeclareLaunchArgument(
            "aligned_depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument(
            "compressed_depth_topic",
            default_value=(
                "/camera/camera/aligned_depth_to_color/"
                "image_raw/compressedDepth")),
        DeclareLaunchArgument(
            "fused_detection_marker_topic",
            default_value="/observation_tracker/fused_detection_markers"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("observation_tracker"), "config",
                "observation_tracker_bev.rviz"])),
    ]
    tracker = Node(
        package="observation_tracker",
        executable="observation_tracker_node",
        name="observation_tracker",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
                "enable_lidar_observation": ParameterValue(
                    LaunchConfiguration("enable_lidar_observation"), value_type=bool),
                "enable_camera_observation": ParameterValue(
                    LaunchConfiguration("enable_camera_observation"), value_type=bool),
                "enable_depth_consistency": ParameterValue(
                    LaunchConfiguration("enable_depth_consistency"), value_type=bool),
                "enable_odometry": ParameterValue(
                    LaunchConfiguration("enable_odometry"), value_type=bool),
                "enable_imu": ParameterValue(
                    LaunchConfiguration("enable_imu"), value_type=bool),
                "enable_csv_logging": ParameterValue(
                    LaunchConfiguration("enable_csv_logging"), value_type=bool),
                "enable_debug_visualization": ParameterValue(
                    LaunchConfiguration("enable_debug_visualization"),
                    value_type=bool),
                "enable_fused_detection_visualization": ParameterValue(
                    LaunchConfiguration("enable_fused_detection_visualization"),
                    value_type=bool),
                "lidar_points_topic": LaunchConfiguration("lidar_points_topic"),
                "camera_rgb_topic": LaunchConfiguration("camera_rgb_topic"),
                "camera_detection_topic": LaunchConfiguration("camera_detection_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "aligned_depth_topic": LaunchConfiguration("aligned_depth_topic"),
                "compressed_depth_topic": LaunchConfiguration(
                    "compressed_depth_topic"),
                "fused_detection_marker_topic": LaunchConfiguration(
                    "fused_detection_marker_topic"),
            },
        ],
        arguments=[
            "--detection_topic", LaunchConfiguration("detection_topic"),
            "--tracked_topic", LaunchConfiguration("tracked_topic"),
            "--bbox_topic", LaunchConfiguration("bbox_topic"),
            "--odom_topic", LaunchConfiguration("odom_topic"),
            "--imu_topic", LaunchConfiguration("imu_topic"),
            "--sequence_id", LaunchConfiguration("sequence_id"),
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="observation_tracker_bev",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )
    return LaunchDescription(arguments + [tracker, rviz])
