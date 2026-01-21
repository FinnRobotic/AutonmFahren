from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'trajectory_planner_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='finn',
    maintainer_email='finn.ole.flemming@gmail.com',
    description='A package for planning a trajectory form a given map',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_node = trajectory_planner_py.trajectory_node:main',
        ],
    },
)
