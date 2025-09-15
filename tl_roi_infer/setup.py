from setuptools import find_packages, setup
from glob import glob

package_name = 'tl_roi_infer'

setup(
    name=package_name,
    version='0.1.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',  glob('launch/*.py')),
        ('share/' + package_name + '/models',  glob('models/*')),
        ('share/' + package_name + '/params',  glob('params/*.yaml')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='jjs523',
    maintainer_email='jjs523@todo.todo',
    description='Traffic light ROI inference node (ROS2, YOLO)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'roi_infer = tl_roi_infer.roi_infer:main',
        ],
    },
)
