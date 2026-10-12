from setuptools import find_packages, setup


package_name = "observation_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/observation_tracker.yaml"]),
        ("share/" + package_name + "/launch", ["launch/observation_tracker.launch.py"]),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="SCV",
    maintainer_email="scv@example.com",
    description="Observation-adaptive 3-D multi-object tracker for VLP-32C and D555",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "observation_tracker_node = observation_tracker.observation_tracker_node:main",
        ],
    },
)

