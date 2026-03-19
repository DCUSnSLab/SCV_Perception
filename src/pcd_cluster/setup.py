from setuptools import setup

package_name = 'pcd_cluster'

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
    description='Euclidean clustering of obstacle point cloud',
    license='MIT',
    entry_points={
        'console_scripts': [
            'cluster_node = pcd_cluster.cluster_node:main',
        ],
    },
)
