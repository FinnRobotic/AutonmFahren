#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import csv
import os
import yaml
import cv2

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from nav_msgs.msg import Odometry

from theta_msgs.msg import Mission, PathWithCurvature, PoseWithCurvature

from fsd_path_planning import PathPlanner, MissionTypes


def _color(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')

        # -------------------------
        # PARAMETERS
        # -------------------------
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('veh_state_topic', '/zed/zed_node/odom')
        self.declare_parameter('mission_topic', '/master/mission')
        self.declare_parameter('path_topic', '/trajectory/middle_line')
        self.declare_parameter('markers_topic', '/traj/markers')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('cones_csv', '/home/nvidia/theta_ws/src/track_planner_grid/cones/cones.csv')
        self.declare_parameter('cone_radius', 0.15)
        self.declare_parameter('cone_height', 0.5)
        self.declare_parameter('target_ds', 0.5)
        self.declare_parameter('use_csv', True)
        self.declare_parameter('experimental_perf', False)

        # -------------------------
        # PARAMETER READ
        # -------------------------
        self.frame_id = self.get_parameter('frame_id').value
        self.veh_state_topic = self.get_parameter('veh_state_topic').value
        self.mission_topic = self.get_parameter('mission_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.cones_csv = self.get_parameter('cones_csv').value
        self.cone_radius = self.get_parameter('cone_radius').value
        self.cone_height = self.get_parameter('cone_height').value
        self.target_ds = float(self.get_parameter('target_ds').value)
        self.use_csv = bool(self.get_parameter('use_csv').value)
        self.experimental_perf = bool(self.get_parameter('experimental_perf').value)

        # -------------------------
        # STATE
        # -------------------------
        self.car_pos = None
        self.car_yaw = None
        self.global_cones = None
        self.map_msg = None

        # -------------------------
        # PUBLISHERS
        # -------------------------
        self.pub_path = self.create_publisher(PathWithCurvature, self.path_topic, 1)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, 1)
        self.pub_map = self.create_publisher(OccupancyGrid, self.map_topic, 1)

        # -------------------------
        # SUBSCRIBERS
        # -------------------------
        self.sub_state = self.create_subscription(Odometry, self.veh_state_topic, self.cb_state, 50)
        self.sub_mission = self.create_subscription(Mission, self.mission_topic, self.cb_mission, 10)

        # -------------------------
        # CSV Cones
        # -------------------------
        if self.use_csv:
            self.create_timer(0.5, self._read_csv)

        # -------------------------
        # PLANNER
        # -------------------------
        self.current_mission = MissionTypes.trackdrive
        self._planner = PathPlanner(self.current_mission, experimental_performance_improvements=self.experimental_perf)

        # -------------------------
        # LOG
        # -------------------------
        self.get_logger().info("TrajectoryVisualizer initialized.")

    # ==========================
    # CSV CONES
    # ==========================
    def _read_csv(self):
        if not os.path.exists(self.cones_csv):
            self.get_logger().warn(f"CSV file not found: {self.cones_csv}")
            return

        left, right, osmall, obig, unk = [], [], [], [], []
        try:
            with open(self.cones_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x, y = float(row['x']), float(row['y'])
                    color = row.get('color', 'unknown').lower()
                    if color == 'blue':
                        left.append((x, y))
                    elif color == 'yellow':
                        right.append((x, y))
                    elif color == 'orange_small':
                        osmall.append((x, y))
                    elif color == 'orange_big':
                        obig.append((x, y))
                    else:
                        unk.append((x, y))
            A = lambda L: np.asarray(L, dtype=np.float64).reshape(-1, 2)
            self.global_cones = [A(unk), A(right), A(left), A(osmall), A(obig)]
            self._publish_all()
        except Exception as e:
            self.get_logger().error(f"CSV read error: {e}")

    # ==========================
    # VEHICLE STATE CALLBACK
    # ==========================
    def cb_state(self, msg: Odometry):
        self.car_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # Yaw from quaternion
        q = msg.pose.pose.orientation
        self.car_yaw = math.atan2(2 * (q.w * q.z), 1 - 2 * (q.z ** 2))
        self._publish_all()

    # ==========================
    # MISSION CALLBACK
    # ==========================
    def cb_mission(self, msg: Mission):
        # Not implemented fully
        pass

    # ==========================
    # MAP LOADING (PGM + YAML)
    # ==========================
    def load_map(self, pgm_file, yaml_file):
        if not os.path.exists(pgm_file) or not os.path.exists(yaml_file):
            self.get_logger().warn("Map files not found.")
            return

        with open(yaml_file, 'r') as f:
            map_info = yaml.safe_load(f)

        resolution = map_info['resolution']
        origin = map_info['origin']
        occupied_thresh = map_info.get('occupied_thresh', 0.65)
        free_thresh = map_info.get('free_thresh', 0.196)

        img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().warn("PGM could not be read")
            return

        height, width = img.shape
        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = origin[0]
        grid.info.origin.position.y = origin[1]
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        data = []
        for y in range(height):
            for x in range(width):
                pixel = img[height - y - 1, x] / 255.0
                if pixel > free_thresh:
                    data.append(0)
                elif pixel < occupied_thresh:
                    data.append(100)
                else:
                    data.append(-1)
        grid.data = data
        self.map_msg = grid

    # ==========================
    # PUBLISH ALL
    # ==========================
    def _publish_all(self):
        now = self.get_clock().now().to_msg()

        # --- Map ---
        if self.map_msg is not None:
            self.map_msg.header.stamp = now
            self.pub_map.publish(self.map_msg)

        # --- Cones ---
        marker_array = MarkerArray()
        if self.global_cones is not None:
            for i, cone_group in enumerate(self.global_cones):
                for j, (x, y) in enumerate(cone_group):
                    marker = Marker()
                    marker.header.frame_id = self.frame_id
                    marker.header.stamp = now
                    marker.ns = f"cones_{i}"
                    marker.id = i * 1000 + j
                    marker.type = Marker.CYLINDER
                    marker.action = Marker.ADD
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = self.cone_height / 2.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = self.cone_radius * 2.0
                    marker.scale.y = self.cone_radius * 2.0
                    marker.scale.z = self.cone_height

                    # Farbe je nach Gruppe
                    if i == 1:   # Right - Yellow
                        marker.color = _color(1.0, 0.9, 0.1)
                    elif i == 2: # Left - Blue
                        marker.color = _color(0.2, 0.3, 1.0)
                    elif i == 3: # Orange small
                        marker.color = _color(1.0, 0.5, 0.0)
                    elif i == 4: # Orange big
                        marker.color = _color(1.0, 0.35, 0.0)
                    else:       # Unknown
                        marker.color = _color(0.7, 0.7, 0.7)
                    marker_array.markers.append(marker)

        # --- Vehicle Position ---
        if self.car_pos is not None:
            car_marker = Marker()
            car_marker.header.frame_id = self.frame_id
            car_marker.header.stamp = now
            car_marker.ns = "car_pos"
            car_marker.id = 9999
            car_marker.type = Marker.SPHERE
            car_marker.action = Marker.ADD
            car_marker.pose.position.x = self.car_pos[0]
            car_marker.pose.position.y = self.car_pos[1]
            car_marker.pose.position.z = 0.2
            car_marker.pose.orientation.w = 1.0
            car_marker.scale.x = 0.3
            car_marker.scale.y = 0.3
            car_marker.scale.z = 0.3
            car_marker.color = _color(0.0, 1.0, 0.0, 1.0)
            marker_array.markers.append(car_marker)

        self.pub_markers.publish(marker_array)


def main():
    rclpy.init()
    node = TrajectoryVisualizer()

    # Optionally load map once
    pgm_file = '/home/nvidia/theta_ws/maps/track_map3.pgm'
    yaml_file = '/home/nvidia/theta_ws/maps/track_map3.yaml'
    node.load_map(pgm_file, yaml_file)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

