from setuptools import setup

package_name = 'tl_roi_hist'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/tl_crop_only.param.yaml']),
        ('share/' + package_name + '/launch', ['launch/tl_crop_only.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Traffic-light ROI crop + Hue histogram state estimator (YOLO + HSV + LT rule)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tl_crop_only = tl_roi_hist.tl_crop_only_node:main',
        ],
    },
)
