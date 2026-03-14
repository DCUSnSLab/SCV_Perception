from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bae',
    maintainer_email='bae@todo.todo',
    description='Lane detection package for ROS2 Humble',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_node = lane_detection.lane_node:main',
            'yolop_lane_detection = lane_detection.yolop_lane_detection:main',
        ],
    },
)
