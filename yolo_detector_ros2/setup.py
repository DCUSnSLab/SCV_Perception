from setuptools import find_packages, setup


package_name = 'yolo_detector_ros2'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scv',
    maintainer_email='scv@example.com',
    description='ROS2 YOLOv8 detector node for vehicle and pedestrian detection.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector_node = yolo_detector_ros2.yolo_detector_node:main',
        ],
    },
)
