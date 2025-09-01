from setuptools import find_packages, setup

package_name = 'object_depth_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/object_tracker.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='d2-521-30',
    maintainer_email='zxc81808080@gmail.com',
    description='3D object depth tracking with multiple filtering algorithms',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_depth_tracker_node = object_depth_tracker.object_depth_tracker_node:main'
        ],
    },
)
