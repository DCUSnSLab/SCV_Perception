import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'terrain_seg'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scv',
    maintainer_email='pcdpcd100@gmail.com',
    description=(
        'SegFormer semantic segmentation and RGB-D semantic point cloud '
        'generation for the SCV RealSense D555.'),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'terrain_seg_node = terrain_seg.terrain_seg_node:main',
        ],
    },
)
