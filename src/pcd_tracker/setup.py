from setuptools import setup

package_name = 'pcd_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','numpy','scikit-learn'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jay@example.com',
    description='Simple nearest-neighbor tracker for clustered obstacles',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tracker_node = pcd_tracker.tracker_node:main',
        ],
    },
)
