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
    # the two RGB lens centers.
    rig_in_velodyne = static_tf(
        'velodyne_to_front_camera_rig',
        'velodyne',
        'front_camera_rig',
        (-0.075936702386557, 0.005907846329035, -0.051649662111118),
        (0.011788154607001, 0.030677511911544,
         -0.029048897759901, 0.999037582482668),
    )

    # These attach the existing RealSense driver trees at their camera_link
    # roots. The driver retains its per-device factory RGB/depth extrinsics.
    left_link_in_rig = static_tf(
        'front_camera_rig_to_front_link',
        'front_camera_rig',
        'front_link',
        (0.007812144921651, 0.071382099650485, 0.001682236771369),
        (0.961186571253649, 0.275863070335251,
         0.000568893289566, 0.004429224230944),
    )
    right_link_in_rig = static_tf(
        'front_camera_rig_to_camera_link',
        'front_camera_rig',
        'camera_link',
        (-0.006982041272288, -0.046847119018253, -0.002483045576121),
        (0.958223094280192, -0.284887283435533,
         -0.004655850233539, 0.025021198680711),
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
