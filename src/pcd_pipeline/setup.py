from setuptools import setup
import os
from glob import glob

package_name = 'pcd_pipeline'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # ament index 등록용
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),

        # package.xml 설치
        ('share/' + package_name, ['package.xml']),

        # launch 파일 설치
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join(package_name, 'launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jay',
    maintainer_email='jay@example.com',
    description='Launch pipeline for ground removal, clustering, and tracking',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
