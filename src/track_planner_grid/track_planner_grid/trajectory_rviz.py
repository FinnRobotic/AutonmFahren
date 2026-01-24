#!/usr/bin/env python3
import csv
import math
import os
import yaml
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class TrajFullViz(Node):
    def __init__(self):
        super().__init__("traj_full_viz")

        # ---------------- Params ----------------
        self.declare_parameter("csv_file", "/home/nvidia/theta_ws/src/track_planner_grid/track/trajectory.csv")
        self.declare_parameter("yaml_file", "maps/track_map6.yaml")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("z", 0.05)
        self.declare_parameter("rate", 1.0)

        self.csv_file = self.get_parameter("csv_file").value
        self.yaml_file = self.get_parameter("yaml_file").value
        self.frame_id = self.get_parameter("frame_id").value
        self.z = float(self.get_parameter("z").value)
        self.rate = float(self.get_parameter("rate").value)

        # ---------------- Publishers (4 Topics) ----------------
        self.pub_map = self.create_publisher(OccupancyGrid, "/viz/map", 1)
        self.pub_curv_marker = self.create_publisher(Marker, "/viz/middle_curvature_marker", 1)
        self.pub_prog_marker = self.create_publisher(Marker, "/viz/trajectory_progress_marker", 1)
        self.pub_path = self.create_publisher(Path, "/viz/path", 1)

        # ---------------- Load data ----------------
        self.xy, self.yaw, self.curv = self.load_csv(self.csv_file)
        self.map_msg = self.load_map(self.yaml_file)

        period = 1.0 / max(0.1, self.rate)
        self.timer = self.create_timer(period, self.publish_all)

        self.get_logger().info("Trajectory Full Viz started.")

    # ==========================================================
    # Load CSV
    # ==========================================================
    def load_csv(self, path):
        xy, yaw, curv = [], [], []
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                xy.append([float(row["x"]), float(row["y"])])
                yaw.append(float(row.get("yaw", "0.0")))
                curv.append(float(row.get("curvature", "0.0")))
        return np.array(xy), np.array(yaw), np.array(curv)

    # ==========================================================
    # Load Map
    # ==========================================================
    def load_map(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        image_path = data["image"]
        if not image_path.startswith("/"):
            image_path = os.path.join(os.path.dirname(yaml_path), image_path)

        img = Image.open(image_path).convert("L")
        pixels = np.array(img, dtype=np.float32) / 255.0
        h, w = pixels.shape

        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id

        info = MapMetaData()
        info.resolution = float(data["resolution"])
        info.width = w
        info.height = h
        info.origin.position.x = data["origin"][0]
        info.origin.position.y = data["origin"][1]
        info.origin.orientation.w = 1.0
        grid.info = info

        occ_th = float(data.get("occupied_thresh", 0.65))
        free_th = float(data.get("free_thresh", 0.196))

        out = []
        for y in range(h - 1, -1, -1):
            for x in range(w):
                v = 1 - pixels[y, x]
                if v > occ_th:
                    out.append(100)
                elif v < free_th:
                    out.append(0)
                else:
                    out.append(-1)

        grid.data = out
        return grid

    # ==========================================================
    # Color ramps
    # ==========================================================
    def ramp_white_yellow_orange_red(self, t):
        t = float(clamp(t, 0.0, 1.0))
        c = ColorRGBA()
        c.a = 1.0

        if t < 1.0 / 3.0:
            u = t / (1.0 / 3.0)
            c.r, c.g, c.b = 1.0, 1.0, 1.0 - u
        elif t < 2.0 / 3.0:
            u = (t - 1.0 / 3.0) / (1.0 / 3.0)
            c.r, c.g, c.b = 1.0, 1.0 - 0.5 * u, 0.0
        else:
            u = (t - 2.0 / 3.0) / (1.0 / 3.0)
            c.r, c.g, c.b = 1.0, 0.5 - 0.5 * u, 0.0

        return c

    def ramp_curvature(self, k, kmax):
        t = abs(k) / max(kmax, 1e-6)
        return self.ramp_white_yellow_orange_red(t)

    # ==========================================================
    # Publish
    # ==========================================================
    def publish_all(self):
        now = self.get_clock().now().to_msg()

        # ---------- MAP ----------
        self.map_msg.header.stamp = now
        self.pub_map.publish(self.map_msg)

        # ---------- PATH ----------
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = now

        for i in range(len(self.xy)):
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = now
            ps.pose.position.x = float(self.xy[i, 0])
            ps.pose.position.y = float(self.xy[i, 1])
            ps.pose.position.z = self.z
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.pub_path.publish(path)

        # ---------- CURVATURE MARKER ----------
        kabs = np.abs(self.curv)
        kmax = np.percentile(kabs, 95) if len(kabs) > 5 else np.max(kabs)

        mc = Marker()
        mc.header.frame_id = self.frame_id
        mc.header.stamp = now
        mc.ns = "middle_curvature"
        mc.id = 0
        mc.type = Marker.LINE_STRIP
        mc.action = Marker.ADD
        mc.pose.orientation.w = 1.0
        mc.scale.x = 0.06

        for i in range(len(self.xy)):
            mc.points.append(Point(x=self.xy[i, 0], y=self.xy[i, 1], z=self.z + 0.02))
            mc.colors.append(self.ramp_curvature(self.curv[i], kmax))

        self.pub_curv_marker.publish(mc)

        # ---------- PROGRESS MARKER ----------
        mp = Marker()
        mp.header.frame_id = self.frame_id
        mp.header.stamp = now
        mp.ns = "trajectory_progress"
        mp.id = 0
        mp.type = Marker.LINE_STRIP
        mp.action = Marker.ADD
        mp.pose.orientation.w = 1.0
        mp.scale.x = 0.08

        N = len(self.xy)
        for i in range(N):
            mp.points.append(Point(x=self.xy[i, 0], y=self.xy[i, 1], z=self.z + 0.05))
            t = i / max(1, N - 1)
            mp.colors.append(self.ramp_white_yellow_orange_red(t))

        self.pub_prog_marker.publish(mp)


def main(args=None):
    rclpy.init(args=args)
    node = TrajFullViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
