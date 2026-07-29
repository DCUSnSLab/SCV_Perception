from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def create_camera_launch(serial_arg, namespace_arg, name_arg, config_arg):
    rs_launch = PathJoinSubstitution(
        [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py']
    )

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            'serial_no': LaunchConfiguration(serial_arg),
            'camera_namespace': LaunchConfiguration(namespace_arg),
            'camera_name': LaunchConfiguration(name_arg),
            'config_file': LaunchConfiguration(config_arg),
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
        }.items(),
    )


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('tem_realsense'), 'config', 'realsense_common.yaml']
    )

    serial_1_arg = DeclareLaunchArgument(
        'serial_no_1',
        default_value="'233522076130'",
        description='RealSense serial number for camera 1',
    )
    serial_2_arg = DeclareLaunchArgument(
        'serial_no_2',
        default_value="'327122078834'",
        description='RealSense serial number for camera 2',
    )
    namespace_1_arg = DeclareLaunchArgument(
        'camera_namespace_1',
        default_value='realsense_1',
        description='ROS namespace for camera 1',
    )
    namespace_2_arg = DeclareLaunchArgument(
        'camera_namespace_2',
        default_value='realsense_2',
        description='ROS namespace for camera 2',
    )
    name_1_arg = DeclareLaunchArgument(
        'camera_name_1',
        default_value='d435i_right',
        description='Node name for the right-side camera on realsense_1',
    )
    name_2_arg = DeclareLaunchArgument(
        'camera_name_2',
        default_value='d435i_left',
        description='Node name for the left-side camera on realsense_2',
    )
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Shared realsense2_camera YAML config file',
    )

    camera_1 = create_camera_launch(
        'serial_no_1', 'camera_namespace_1', 'camera_name_1', 'config_file'
    )
    camera_2 = create_camera_launch(
        'serial_no_2', 'camera_namespace_2', 'camera_name_2', 'config_file'
    )

    return LaunchDescription([
        serial_1_arg,
        serial_2_arg,
        namespace_1_arg,
        namespace_2_arg,
        name_1_arg,
        name_2_arg,
        config_arg,
        camera_1,
        camera_2,
    ])
