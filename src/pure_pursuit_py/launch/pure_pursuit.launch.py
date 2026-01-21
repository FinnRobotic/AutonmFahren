from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = 'pure_pursuit_py'

    # Default-Config für REAL
    cfg_default = os.path.join(
        get_package_share_directory(pkg),
        'config',
        'pure_pursuit.yaml'
    )
    cfg_default = os.path.normpath(cfg_default)

    cfg_arg = DeclareLaunchArgument(
        'config',
        default_value=cfg_default,
        description='YAML config file for pure pursuit'
    )

    return LaunchDescription([
        cfg_arg,
        Node(
            package=pkg,
            executable='pure_pursuit_node',
            name='pure_pursuit_controller',
            parameters=[LaunchConfiguration('config')],
            output='screen',
        ),
    ])