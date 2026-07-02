from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'ultralytics_ros2'

# 수동 수집
model_files = glob('model/*.pt')
launch_files = glob('launch/*.py')
param_files = glob('params/*.yaml')

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', launch_files),
    ('share/' + package_name + '/model', model_files),
    ('share/' + package_name + '/params', param_files),
]

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=[
        'setuptools',
        # pip 패키지(환경에 따라 직접 설치 필요):
        # 'ultralytics', 'opencv-python', 'numpy'
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='ROS 2 wrapper node for Ultralytics segmentation overlay',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seg_run = ultralytics_ros2.seg_run:main',
        ],
    },
)
