import launch
from launch import LaunchDescription
from launch_ros.actions import Node



def generate_launch_description():
    bringup_path = '/home/nvidia/theta_ws/src/f1tenth_system/f1tenth_stack/launch/bringup_launch.py'

    bringup_source = launch.launch_description_sources.PythonLaunchDescriptionSource(bringup_path)

    return LaunchDescription(
        [
            launch.actions.IncludeLaunchDescription(bringup_source),
            Node(


                package = 'theta_roslab',
                namespace= 'lidar_example_node',
                executable='lidar_example_copy_node',
                name='lidar_example_copy_node'
            ),
            Node(
                package='rviz2',
                namespace='',
                executable='rviz2',
                name='rviz2',
                arguments=['-d' + '/home/nvidia/<group_name_ws>/config/config_file.rviz']
            ),
        ]
    )

