#!/usr/bin/env python3
import csv
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


class TrajectoryCsvViz(Node):
    def __init__(self):
        super().__init__("trajectory_csv_viz")

        self.declare_parameter("csv_file", "/home/finn/Desktop/AutonmFahren/src/track_planner_grid/track/trajectory.csv")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("path_topic", "/traj_path")
        self.declare_parameter("marker_topic", "/traj_marker")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("z", 0.05)

        self.csv_file = self.get_parameter("csv_file").value
        self.frame_id = self.get_parameter("frame_id").value
        self.path_topic = self.get_parameter("path_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.rate = float(self.get_parameter("publish_rate_hz").value)
        self.z = float(self.get_parameter("z").value)

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)

        self.xy, self.yaw, self.curv = self.load_csv(self.csv_file)

        if len(self.xy) < 2:
            self.get_logger().warn("CSV has too few points to visualize.")
        else:
            self.get_logger().info(f"Loaded {len(self.xy)} points from {self.csv_file}")

        period = 1.0 / max(0.1, self.rate)
        self.timer = self.create_timer(period, self.publish)

    def load_csv(self, path):
        xy = []
        yaw = []
        curv = []
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                xy.append([float(row["x"]), float(row["y"])])
                yaw.append(float(row.get("yaw", "0.0")))
                curv.append(float(row.get("curvature", "0.0")))
        return np.array(xy, dtype=np.float32), np.array(yaw, dtype=np.float32), np.array(curv, dtype=np.float32)

    def quat_from_yaw(self, yaw):
        # planar yaw to quaternion
        # q = [0,0,sin(y/2),cos(y/2)]
        return (0.0, 0.0, float(math.sin(yaw * 0.5)), float(math.cos(yaw * 0.5)))

    def publish(self):
        now = self.get_clock().now().to_msg()

        # ---- Path
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = now

        for i in range(len(self.xy)):
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = now
            ps.pose.position.x = float(self.xy[i, 0])
            ps.pose.position.y = float(self.xy[i, 1])
            ps.pose.position.z = float(self.z)
            qx, qy, qz, qw = self.quat_from_yaw(float(self.yaw[i]) if i < len(self.yaw) else 0.0)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)

        self.path_pub.publish(path)

        # ---- Marker with curvature coloring
        # Map curvature magnitude -> color:
        # low |k| => white, high |k| => red
        kabs = np.abs(self.curv) if len(self.curv) == len(self.xy) else np.zeros((len(self.xy),), np.float32)
        kmax = float(np.percentile(kabs, 95)) if len(kabs) > 5 else float(np.max(kabs) if len(kabs) else 1.0)
        if kmax < 1e-9:
            kmax = 1.0

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = now
        m.ns = "traj"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06  # line width

        for i in range(len(self.xy)):
            m.points.append(Point(x=float(self.xy[i, 0]), y=float(self.xy[i, 1]), z=float(self.z)))

            t = float(min(1.0, kabs[i] / kmax))  # 0..1
            # white -> red: (1,1,1) -> (1,0,0)
            c = ColorRGBA()
            c.r = 1.0
            c.g = 1.0 - t
            c.b = 1.0 - t
            c.a = 1.0
            m.colors.append(c)

        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCsvViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
