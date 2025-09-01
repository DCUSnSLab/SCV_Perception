from setuptools import setup, find_packages

package_name = 'memsort_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/memsort_ros2']),
        ('share/memsort_ros2', ['package.xml']),
    ],
    install_requires=[
        # pip 패키지는 rosdep로는 안깔립니다. 실행 환경에서 미리 설치하세요.
        # 'ultralytics','opencv-python','torch','numpy' 등
    ],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Memory-SORT tracker ROS2 node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'memory_sort_node = memsort_ros2.memory_sort_node:main',
        ],
    },
)
