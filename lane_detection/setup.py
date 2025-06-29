from setuptools import setup, find_packages

setup(
    name='lane_detection',
    version='0.0.0',
    packages=find_packages(  # lib/‥ 하위까지 자동 포함
        include=("lane_detection*", "lib*")
    ),
    install_requires=['rospy', 'numpy', 'torch'],  # 필요한 것만
)
