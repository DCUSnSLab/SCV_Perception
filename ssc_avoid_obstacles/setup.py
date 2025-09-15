from setuptools import find_packages, setup

package_name = 'ssc_avoid_obstacles'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/ssc_avoid_obstacles/launch', ['launch/send_stop_flag.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choe',
    maintainer_email='yujeongchoe20@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_stop_flag = ssc_avoid_obstacles.send_stop_flag:main',
        ],
    },
)
