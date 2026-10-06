from setuptools import setup

package_name = 'pcd_ground_filter'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/curb_detection.launch.py', 'launch/curb_costmap.launch.py']),
        ('share/' + package_name + '/config', ['config/curb_params.yaml']),
    ],
    install_requires=['setuptools', 'numpy', 'scikit-learn'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jay@example.com',
    description='LiDAR ground removal using slope/elevation + height cutoff',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ground_removal_node = pcd_ground_filter.ground_removal_node:main',
            'curb_detection_node = pcd_ground_filter.curb_detection_node:main',
        ],
    },
)
