from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = 'trajectory_planner_py'
    default_cfg = os.path.join(get_package_share_directory(pkg), 'config', 'trajectory.yaml')

    cfg_arg = DeclareLaunchArgument('cfg', default_value=default_cfg)

    return LaunchDescription([
        cfg_arg,
        Node(
            package=pkg,
            executable='trajectory_node',
            name='trajectory_node',
            parameters=[LaunchConfiguration('cfg')],
            output='screen'
        )
    ])