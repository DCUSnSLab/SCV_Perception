from setuptools import find_packages, setup

package_name = 'pcdet_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='anam3200@cu.ac.kr',
    description='3D object tracking package for OpenPCDet detections',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tracker_node = pcdet_tracker.tracker_node:main',
            'visualizer_node = pcdet_tracker.visualizer_node:main',
            'ab3dmot_node = pcdet_tracker.ab3dmot_node:main',
            'jay_tracker = pcdet_tracker.jay_tracker:main',
            'kitti_based_tracker = pcdet_tracker.jay_tracker:main',
        ],
    },
)
