#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import yaml
import csv
import cv2
import numpy as np
import os

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose


class MapAndConesPublisher(Node):

    def __init__(self):
        super().__init__('map_and_cones_publisher')

        # =========================
        # PARAMETER
        # =========================
        self.declare_parameter('pgm_file', '/home/nvidia/theta_ws/maps/track_map3.pgm')
        self.declare_parameter('yaml_file', '/home/nvidia/theta_ws/maps/track_map3.yaml')
        self.declare_parameter('cones_csv', '/home/nvidia/theta_ws/src/track_planner_grid/cones/cones.csv')
        self.declare_parameter('frame_id', 'map')

        self.declare_parameter('cone_radius', 0.15)
        self.declare_parameter('cone_height', 0.5)

        self.frame_id = self.get_parameter('frame_id').value
        self.cone_radius = self.get_parameter('cone_radius').value
        self.cone_height = self.get_parameter('cone_height').value

        pgm_file = self.get_parameter('pgm_file').value
        yaml_file = self.get_parameter('yaml_file').value
        cones_csv = self.get_parameter('cones_csv').value

        # =========================
        # PUBLISHER
        # =========================
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.cone_pub = self.create_publisher(MarkerArray, '/virtual_cones', 10)

        # =========================
        # LADEN
        # =========================
        self.map_msg = self.load_map(pgm_file, yaml_file)
        self.cones = self.load_cones(cones_csv)

        self.timer = self.create_timer(1.0, self.publish_all)

        self.get_logger().info("Map + virtuelle Cones werden published")

    # ==========================================================
    # MAP LADEN (PGM + YAML → OccupancyGrid)
    # ==========================================================
    def load_map(self, pgm_file, yaml_file):

        if not os.path.exists(pgm_file):
            raise RuntimeError("PGM-Datei nicht gefunden")
        if not os.path.exists(yaml_file):
            raise RuntimeError("YAML-Datei nicht gefunden")

        with open(yaml_file, 'r') as f:
            map_info = yaml.safe_load(f)

        resolution = map_info['resolution']
        origin = map_info['origin']
        occupied_thresh = map_info.get('occupied_thresh', 0.65)
        free_thresh = map_info.get('free_thresh', 0.196)

        img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("PGM konnte nicht gelesen werden")

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
                    data.append(0)        # frei
                elif pixel < occupied_thresh:
                    data.append(100)      # belegt
                else:
                    data.append(-1)       # unbekannt

        grid.data = data
        return grid

    # ==========================================================
    # CONES LADEN
    # ==========================================================
    def load_cones(self, csv_file):
        cones = []
        if not os.path.exists(csv_file):
            self.get_logger().warn("Keine Cone-CSV gefunden")
            return cones

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cones.append((float(row['x']), float(row['y'])))

        return cones

    # ==========================================================
    # PUBLISH
    # ==========================================================
    def publish_all(self):
        now = self.get_clock().now().to_msg()

        # Map
        self.map_msg.header.stamp = now
        self.map_pub.publish(self.map_msg)

        # Cones
        marker_array = MarkerArray()

        for i, (x, y) in enumerate(self.cones):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = now

            marker.ns = "virtual_cones"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = self.cone_height / 2.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = self.cone_radius * 2.0
            marker.scale.y = self.cone_radius * 2.0
            marker.scale.z = self.cone_height

            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.cone_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MapAndConesPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
