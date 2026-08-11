from setuptools import find_packages
from setuptools import setup

setup(
    name='memsort_ros2',
    version='0.1.0',
    packages=find_packages(
        include=('memsort_ros2', 'memsort_ros2.*')),
)
