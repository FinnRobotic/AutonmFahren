import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    bringup_path = '/home/nvidia/theta_ws/src/f1tenth_system/f1tenth_stack/launch/bringup_launch.py'

    bringup_source = launch.launch_description_sources.PythonLaunchDescriptionSource(bringup_path)

    bringup_path2 = '/home/nvidia/vicon_receiver_ws/src/vrpn_client_ros/launch/sample.launch.py'

    bringup_source2 = launch.launch_description_sources.PythonLaunchDescriptionSource(bringup_path2)

    return LaunchDescription(
        [
            launch.actions.IncludeLaunchDescription(bringup_source),
            Node(


                package = 'theta_roslab',
                namespace= 'pure_pursuit_node',
                executable='pure_pursuit_node',
                name='pure_pursuit_node'
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