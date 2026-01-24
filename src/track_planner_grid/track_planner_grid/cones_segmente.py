#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

import csv
import numpy as np
from sklearn.cluster import DBSCAN

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class ConeSegmentation(Node):

    def __init__(self):
        super().__init__('cone_segmentation')

        self.declare_parameter('csv_path', '/home/nvidia/theta_ws/src/track_planner_grid/cones/cones.csv')
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value

        if csv_path == '':
            self.get_logger().error("csv_path parameter missing")
            return

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub = self.create_publisher(
            MarkerArray,
            '/cone_segments',
            qos
        )

        points = self.load_csv(csv_path)
        self.marker_msg = self.cluster_and_build_markers(points)

        self.timer = self.create_timer(0.5, self.publish)
        self.get_logger().info("Cone segmentation with distance clustering running")

    # -----------------------------

    def load_csv(self, path):
        pts = []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                pts.append([float(r[0]), float(r[1])])
        return np.array(pts)

    # -----------------------------

    def cluster_and_build_markers(self, points):
        clustering = DBSCAN(
            eps=0.4,        # 🔥 30 cm Regel
            min_samples=3
        ).fit(points)

        labels = clustering.labels_
        unique_labels = sorted(set(labels) - {-1})

        msg = MarkerArray()
        marker_id = 0

        for lbl in unique_labels:
            seg_pts = points[labels == lbl]

            m = Marker()
            m.header.frame_id = "map"
            m.ns = "cone_segments"
            m.id = marker_id
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.color = self.color_from_label(lbl)
            m.lifetime.sec = 0

            # sortiert entlang der Strecke (grob)
            center = np.mean(seg_pts, axis=0)
            angles = np.arctan2(seg_pts[:,1]-center[1], seg_pts[:,0]-center[0])
            seg_pts = seg_pts[np.argsort(angles)]

            for x, y in seg_pts:
                m.points.append(Point(x=float(x), y=float(y), z=0.05))

            # schließen
            m.points.append(m.points[0])

            msg.markers.append(m)
            marker_id += 1

        return msg

    # -----------------------------

    def publish(self):
        now = self.get_clock().now().to_msg()
        for m in self.marker_msg.markers:
            m.header.stamp = now
        self.pub.publish(self.marker_msg)

    # -----------------------------

    def color_from_label(self, lbl):
        colors = [
            (0.1, 0.9, 0.1),
            (0.1, 0.3, 0.9),
            (0.9, 0.1, 0.1),
            (0.9, 0.9, 0.1)
        ]
        r, g, b = colors[lbl % len(colors)]
        return ColorRGBA(r=r, g=g, b=b, a=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ConeSegmentation()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
