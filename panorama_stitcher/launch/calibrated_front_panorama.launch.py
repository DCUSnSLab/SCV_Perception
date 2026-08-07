import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def static_tf(name, parent, child, translation, quaternion):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        output='screen',
        arguments=[
            '--x', str(translation[0]),
            '--y', str(translation[1]),
            '--z', str(translation[2]),
            '--qx', str(quaternion[0]),
            '--qy', str(quaternion[1]),
            '--qz', str(quaternion[2]),
            '--qw', str(quaternion[3]),
            '--frame-id', parent,
            '--child-frame-id', child,
        ],
    )


def load_runtime_tf_poses():
    """Load every calibrated runtime TF from the source-of-truth YAML."""
    calibration_path = os.path.join(
        get_package_share_directory('panorama_stitcher'),
        'config',
        'rig_extrinsics.yaml',
    )
    with open(calibration_path, 'r', encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream) or {}
    poses = calibration.get('runtime_tf', {})

    pose_keys = {
        'front_camera_rig': 'front_camera_rig_in_velodyne',
        'front_left': 'front_left_link_in_front_camera_rig',
        'front_right': 'front_right_link_in_front_camera_rig',
    }
    result = {}
    for name, key in pose_keys.items():
        pose = poses.get(key, {})
        translation = pose.get('translation_m')
        quaternion = pose.get('quaternion_xyzw')
        if not isinstance(translation, list) or len(translation) != 3:
            raise RuntimeError(
                f'{calibration_path}: runtime_tf.{key}.translation_m '
                'must contain 3 values')
        if not isinstance(quaternion, list) or len(quaternion) != 4:
            raise RuntimeError(
                f'{calibration_path}: runtime_tf.{key}.quaternion_xyzw '
                'must contain 4 values')
        result[name] = (translation, quaternion)
    return result


def generate_launch_description():
    runtime_tf_poses = load_runtime_tf_poses()
    default_config = PathJoinSubstitution([
        FindPackageShare('panorama_stitcher'),
        'config',
        'rgbd_panorama.yaml',
    ])

    # External calibration convention:
    #   p_velodyne = R * p_front_camera_rig + t
    # front_camera_rig: x forward, y left, z up; origin at the midpoint of
    # the two RGB lens centers. The pose is loaded from the 2026-08-06
    # remounted-rig calibration recorded in rig_extrinsics.yaml.
    rig_in_velodyne = static_tf(
        'velodyne_to_front_camera_rig',
        'velodyne',
        'front_camera_rig',
        *runtime_tf_poses['front_camera_rig'],
    )

    # These attach the RealSense trees at their role-specific root frames.
    # The driver retains its per-device factory RGB/depth extrinsics.
    left_link_in_rig = static_tf(
        'front_camera_rig_to_front_left_link',
        'front_camera_rig',
        'front_left_link',
        *runtime_tf_poses['front_left'],
    )
    right_link_in_rig = static_tf(
        'front_camera_rig_to_front_right_link',
        'front_camera_rig',
        'front_right_link',
        *runtime_tf_poses['front_right'],
    )

    panorama_optical_in_rig = static_tf(
        'front_camera_rig_to_panorama_optical',
        'front_camera_rig',
        'panorama_optical_frame',
        (0.0, 0.0, 0.0),
        (-0.5, 0.5, -0.5, 0.5),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Calibrated front panorama parameter file',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/panorama/image_raw',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
        ),
        rig_in_velodyne,
        left_link_in_rig,
        right_link_in_rig,
        panorama_optical_in_rig,
        Node(
            package='panorama_stitcher',
            executable='rgbd_panorama_stitcher_node',
            # Keep this name matched to the YAML root key.
            name='panorama_stitcher',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'output_topic': LaunchConfiguration('output_topic'),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'),
                        value_type=bool,
                    ),
                },
            ],
        ),
    ])
