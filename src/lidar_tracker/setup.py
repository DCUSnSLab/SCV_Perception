from setuptools import setup
import os
from glob import glob

package_name = 'lidar_tracker'
submodules = 'lidar_tracker/AB3DMOT_libs'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='The lidar_tracker package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = lidar_tracker.tracker_node:main',
            'visualizer_node = lidar_tracker.visualizer_node:main',
        ],
    },
)