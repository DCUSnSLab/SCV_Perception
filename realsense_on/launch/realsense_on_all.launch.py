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
            'usb_port_id': LaunchConfiguration(serial_arg.replace('serial_no', 'usb_port_id')),
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
        [FindPackageShare('realsense_on'), 'config', 'realsense_555.yaml']
    )

    front_serial_arg = DeclareLaunchArgument(
        'front_serial_no',
        default_value="'419222301550'",
        description='RealSense serial number for the front D555 camera',
    )
    front_usb_port_arg = DeclareLaunchArgument(
        'front_usb_port_id',
        default_value="'2-8'",
        description='USB port ID for the front D555 camera',
    )
    right_serial_arg = DeclareLaunchArgument(
        'right_serial_no',
        default_value="'233522076130'",
        description='RealSense serial number for the right D435i camera',
    )
    right_usb_port_arg = DeclareLaunchArgument(
        'right_usb_port_id',
        default_value="'4-2.1'",
        description='USB port ID for the right D435i camera',
    )
    left_serial_arg = DeclareLaunchArgument(
        'left_serial_no',
        default_value="'327122078834'",
        description='RealSense serial number for the left D435i camera',
    )
    left_usb_port_arg = DeclareLaunchArgument(
        'left_usb_port_id',
        default_value="'4-2.2'",
        description='USB port ID for the left D435i camera',
    )

    front_namespace_arg = DeclareLaunchArgument(
        'front_camera_namespace',
        default_value='realsense_on',
        description='ROS namespace for the front camera',
    )
    right_namespace_arg = DeclareLaunchArgument(
        'right_camera_namespace',
        default_value='realsense_1',
        description='ROS namespace for the right camera',
    )
    left_namespace_arg = DeclareLaunchArgument(
        'left_camera_namespace',
        default_value='realsense_2',
        description='ROS namespace for the left camera',
    )

    front_name_arg = DeclareLaunchArgument(
        'front_camera_name',
        default_value='d555_front',
        description='Node name for the front camera',
    )
    right_name_arg = DeclareLaunchArgument(
        'right_camera_name',
        default_value='d435i_right',
        description='Node name for the right camera',
    )
    left_name_arg = DeclareLaunchArgument(
        'left_camera_name',
        default_value='d435i_left',
        description='Node name for the left camera',
    )

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Shared realsense2_camera YAML config file',
    )

    front_camera = create_camera_launch(
        'front_serial_no',
        'front_camera_namespace',
        'front_camera_name',
        'config_file',
    )
    right_camera = create_camera_launch(
        'right_serial_no',
        'right_camera_namespace',
        'right_camera_name',
        'config_file',
    )
    left_camera = create_camera_launch(
        'left_serial_no',
        'left_camera_namespace',
        'left_camera_name',
        'config_file',
    )

    return LaunchDescription([
        front_serial_arg,
        front_usb_port_arg,
        right_serial_arg,
        right_usb_port_arg,
        left_serial_arg,
        left_usb_port_arg,
        front_namespace_arg,
        right_namespace_arg,
        left_namespace_arg,
        front_name_arg,
        right_name_arg,
        left_name_arg,
        config_arg,
        front_camera,
        right_camera,
        left_camera,
    ])
