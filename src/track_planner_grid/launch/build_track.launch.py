# launch/build_track.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="track_planner_grid",
            executable="grid_track_builder_node",
            name="grid_track_builder_node",
            output="screen",
            parameters=[{
                "start_xy_m": [0.0, 0.0],
                "inflate_radius_m": 0.30,
                "safe_margin_m": 0.05,
                "spur_prune_m": 1.0,
                "resample_ds_m": 0.10,
                "min_loop_length_m": 20.0,
                "require_through_start": True,
                "closed_loop": True,
            }]
        )
    ])