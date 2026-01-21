from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = 'pure_pursuit_py'

    # Default-Config für PURE PURSUIT (Sim)
    cfg_default = os.path.join(
        get_package_share_directory(pkg),
        'config',
        'pure_pursuit.yaml'
    )
    cfg_default = os.path.normpath(cfg_default)

    cfg_arg = DeclareLaunchArgument(
        'pp_config',
        default_value=cfg_default,
        description='YAML config file for pure pursuit controller'
    )

    pure_pursuit_node = Node(
        package=pkg,
        executable='pure_pursuit_node',
        name='pure_pursuit_controller',
        parameters=[LaunchConfiguration('pp_config')],
        output='screen',
    )

    actuator_to_fsds_node = Node(
        package=pkg,
        executable='actuator_to_fsds',
        name='actuator_to_fsds',
        parameters=[LaunchConfiguration('pp_config')],
        output='screen',
    )

    return LaunchDescription([
        cfg_arg,
        pure_pursuit_node,
        actuator_to_fsds_node,
    ])