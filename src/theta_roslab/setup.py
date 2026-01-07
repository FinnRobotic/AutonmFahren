from setuptools import setup
import os
from glob import glob


package_name = 'theta_roslab'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'waypoints'), glob('waypoints/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='Simple lidar node',
    license='MIT',
    tests_require=['pytest'],



    package_data={
    'theta_roslab': ['waypoints/*.csv'],    
    },
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'lidar_example_node=theta_roslab.LidarExampleNode:main',
            'lidar_example_copy_node=theta_roslab.LidarExampleNodeCopy:main',
            'keyboard_control_talker=theta_roslab.KeyboardControlTalker:main',
            'key_listener=theta_roslab.keyListener:main',   
            'safety_node=theta_roslab.SafetyNode:main',
            'wall_follow_node=theta_roslab.wall_follow:main',
            'pure_pursuit_node=theta_roslab.pure_pursuit:main',
            'vicon_to_odom=theta_roslab.vicon_to_odom:main',
            'vicon_to_odom_2=theta_roslab.vicon_to_odom2:main',
        ],
    },
)
