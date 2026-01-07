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
                namespace= 'wall_follow_node',
                executable='wall_follow_node',
                name='wall_follow_node'
            ),
            Node(
                package='rviz2',
                namespace='',
                executable='rviz2',
                name='rviz2',
                arguments=['-d' + '/home/nvidia/<group_name_ws>/config/config_file.rviz']
            ),
            Node(


                package = 'theta_roslab',
                namespace= 'safety_node',
                executable='safety_node',
                name='safety_node'
            ),
          
            

        ]
    )