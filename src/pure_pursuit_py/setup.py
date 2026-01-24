from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'pure_pursuit_py'

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
    description='A package for trajectory following',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = pure_pursuit_py.pure_pursuit_node:main',
            'tf_to_odom_node = pure_pursuit_py.tf_to_odom:main',
        ],
    },
)
