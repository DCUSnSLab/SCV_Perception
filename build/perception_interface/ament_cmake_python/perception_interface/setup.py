from setuptools import find_packages
from setuptools import setup

setup(
    name='perception_interface',
    version='0.0.0',
    packages=find_packages(
        include=('perception_interface', 'perception_interface.*')),
)
