from setuptools import find_packages, setup


package_name = 'crosswalk_safety_ros2'


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
    description='Crosswalk safety estimator using left and right detections with depth-derived distance.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'crosswalk_safety_node = crosswalk_safety_ros2.crosswalk_safety_node:main',
        ],
    },
)
