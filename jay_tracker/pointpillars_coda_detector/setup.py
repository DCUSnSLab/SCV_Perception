from glob import glob
from setuptools import find_packages, setup

package_name = "pointpillars_coda_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "easydict"],
    zip_safe=True,
    maintainer="jay",
    maintainer_email="anam3200@cu.ac.kr",
    description="ROS2 detector package for CODa-trained PointPillars",
    license="MIT",
    entry_points={
        "console_scripts": [
            "coda_pointpillar_node = pointpillars_coda_detector.coda_pointpillar_node:main",
        ],
    },
)
