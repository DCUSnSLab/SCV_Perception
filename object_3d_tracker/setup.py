from setuptools import find_packages, setup

package_name = 'object_3d_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/object_3d_tracker.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='d2-521-30',
    maintainer_email='zxc81808080@gmail.com',
    description='3D object tracking from segmentation masks and depth images',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_3d_tracker_node = object_3d_tracker.object_3d_tracker_node:main',
        ],
    },
)
