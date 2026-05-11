from glob import glob
from setuptools import find_packages, setup

package_name = "behavior_predictor"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="jay",
    maintainer_email="anam3200@cu.ac.kr",
    description="Basic behavior prediction package using tracked trajectories",
    license="MIT",
    entry_points={
        "console_scripts": [
            "behavior_predictor_node = behavior_predictor.behavior_predictor_node:main",
        ],
    },
)
