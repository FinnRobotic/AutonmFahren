from setuptools import setup

package_name = 'track_planner_grid'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name + "/launch", ["launch/build_track.launch.py"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='finn.ole.flemming@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grid_track_builder_node=track_planner_grid.grid_track_builder:main',
        ],
    },
)
