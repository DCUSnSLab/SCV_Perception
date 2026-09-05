from setuptools import find_packages, setup


package_name = 'mando_tools'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/launch',
            [
                'launch/green_down_arrow.launch.py',
                'launch/play_mando_bag.launch.py',
                'launch/traffic_light.launch.py',
                'launch/tl_fusion.launch.py',
                'launch/tl_roi_hist.launch.py',
                'launch/validate_mando_bag.launch.py',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ki',
    maintainer_email='ki@localhost',
    description='Utility package for the mando ROS 2 workspace.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mando_bag = mando_tools.bag_cli:main',
            'mando_green_down_arrow = mando_tools.green_down_arrow:main',
            'mando_tl_fusion = mando_tools.tl_fusion:main',
            'mando_tl_roi_hist = mando_tools.tl_roi_hist:main',
            'workspace_info = mando_tools.workspace_info:main',
            'mando_yolo_validate = mando_tools.yolo_validator:main',
        ],
    },
)
