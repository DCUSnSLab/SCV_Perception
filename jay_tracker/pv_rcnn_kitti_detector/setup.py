from glob import glob
from setuptools import find_packages, setup

package_name = 'pv_rcnn_kitti_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'easydict'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='anam3200@cu.ac.kr',
    description='OpenPCDet-based 3D detector package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pointpillars_node = pv_rcnn_kitti_detector.pointpillars_node:main',
            'kitti_detector_node = pv_rcnn_kitti_detector.kitti_detector_node:main',
        ],
    },
)
