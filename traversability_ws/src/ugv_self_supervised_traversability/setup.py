from setuptools import find_packages, setup

package_name = 'ugv_self_supervised_traversability'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/traversability_labeling.launch.py']),
        ('share/' + package_name + '/config', ['config/traversability.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UGV Traversability Maintainer',
    maintainer_email='maintainer@example.com',
    description='Delayed self-supervised positive traversability label generation for UGVs.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'trajectory_recorder = ugv_self_supervised_traversability.trajectory_recorder:main',
            'footprint_generator = ugv_self_supervised_traversability.footprint_generator:main',
            'traversability_labeler = ugv_self_supervised_traversability.traversability_labeler:main',
            'label_visualizer = ugv_self_supervised_traversability.label_visualizer:main',
        ],
    },
)
