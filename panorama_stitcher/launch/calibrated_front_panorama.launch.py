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


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('panorama_stitcher'),
        'config',
        'rgbd_panorama.yaml',
    ])

    # External calibration convention:
    #   p_velodyne = R * p_front_camera_rig + t
    # front_camera_rig: x forward, y left, z up; origin at the midpoint of
    # the two RGB lens centers. 2026-08-04 new-mount recalibration; see
    # lidar_rig_extrinsics_20260804.yaml and rig_extrinsics.yaml.
    rig_in_velodyne = static_tf(
        'velodyne_to_front_camera_rig',
        'velodyne',
        'front_camera_rig',
        (-0.00448541435124259, 0.02500756951604277,
         0.21625076381657354),
        (0.010795577032945527, 0.03790650713358004,
         -0.04021168161650252, 0.9984135280008133),
    )

    # These attach the RealSense trees at their role-specific root frames.
    # The driver retains its per-device factory RGB/depth extrinsics.
    left_link_in_rig = static_tf(
        'front_camera_rig_to_front_left_link',
        'front_camera_rig',
        'front_left_link',
        (0.0027668551625554174, 0.022891964978233523,
         0.022070886433021613),
        (-0.004055179658881763, -0.0018700761474020188,
         0.27668413709010087, 0.9609505432725678),
    )
    right_link_in_rig = static_tf(
        'front_camera_rig_to_front_right_link',
        'front_camera_rig',
        'front_right_link',
        (-0.0020299669272732714, -0.047836572251912846,
         -0.02196040437160633),
        (-0.003141950239487204, -0.00632642612279132,
         -0.2787819097627938, 0.9603284600959273),
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
