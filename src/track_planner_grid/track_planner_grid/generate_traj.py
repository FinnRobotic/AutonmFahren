#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import yaml
import csv
import cv2
import numpy as np
import os
from math import acos

from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point

from scipy.interpolate import splprep, splev


def clamp(x, a, b):
    return max(a, min(b, x))


class GenerateTraj(Node):
    def __init__(self):
        super().__init__("generate_traj")

        # =========================
        # PARAMETER
        # =========================
        self.declare_parameter('pgm_file', '/home/nvidia/theta_ws/maps/track_map3.pgm')
        self.declare_parameter('yaml_file', '/home/nvidia/theta_ws/maps/track_map3.yaml')
        self.declare_parameter('cones_csv', '/home/nvidia/theta_ws/src/track_planner_grid/cones/cones.csv')
        self.declare_parameter('frame_id', 'map')

        # cone viz
        self.declare_parameter('cone_radius', 0.15)
        self.declare_parameter('cone_height', 0.5)

        # graph / geometry
        self.declare_parameter('same_line_radius', 0.30)   # <— WICHTIG: gleiche Linie
        self.declare_parameter('min_edge_dist', 0.05)      # gegen doppelte Punkte
        self.declare_parameter('resample_points', 400)

        # smoothing
        self.declare_parameter('smoothing', 0.8)
        self.declare_parameter('closed_track', False)      # wenn true: per=True in spline

        self.frame_id = self.get_parameter('frame_id').value
        self.cone_radius = float(self.get_parameter('cone_radius').value)
        self.cone_height = float(self.get_parameter('cone_height').value)

        self.same_line_radius = float(self.get_parameter('same_line_radius').value)
        self.min_edge_dist = float(self.get_parameter('min_edge_dist').value)
        self.resample_points = int(self.get_parameter('resample_points').value)

        self.smoothing = float(self.get_parameter('smoothing').value)
        self.closed_track = bool(self.get_parameter('closed_track').value)

        pgm_file = self.get_parameter('pgm_file').value
        yaml_file = self.get_parameter('yaml_file').value
        cones_csv = self.get_parameter('cones_csv').value

        # =========================
        # PUBLISHER
        # =========================
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.cone_pub = self.create_publisher(MarkerArray, '/virtual_cones', 10)

        # Boundary-Lines + Midline
        self.debug_pub = self.create_publisher(MarkerArray, '/track_debug_markers', 10)

        # Final trajectory
        self.path_pub = self.create_publisher(Path, '/trajectory_path', 10)

        # =========================
        # LOAD
        # =========================
        self.map_msg = self.load_map(pgm_file, yaml_file)
        self.cones = self.load_cones(cones_csv)
        if len(self.cones) < 4:
            raise RuntimeError("Zu wenige Cones in CSV")

        # =========================
        # BUILD CENTERLINE
        # =========================
        adj = self.build_graph(self.cones, self.same_line_radius, self.min_edge_dist)
        comps = self.connected_components(adj)

        # nehme die zwei größten Komponenten als Grenzen
        comps = sorted(comps, key=lambda c: len(c), reverse=True)
        if len(comps) < 2:
            raise RuntimeError(f"Nur {len(comps)} Connected Component(s) gefunden. same_line_radius prüfen!")

        compA = comps[0]
        compB = comps[1]

        boundaryA = self.order_component(self.cones, compA, adj)
        boundaryB = self.order_component(self.cones, compB, adj)

        # Resample entlang Bogenlänge auf gleiche Anzahl Punkte
        A_rs = self.resample_polyline(boundaryA, self.resample_points, closed=False)
        B_rs = self.resample_polyline(boundaryB, self.resample_points, closed=False)

        # Paarung: gleiche "Laufrichtung" erzwingen (falls invertiert)
        if np.linalg.norm(A_rs[0] - B_rs[0]) > np.linalg.norm(A_rs[0] - B_rs[-1]):
            B_rs = B_rs[::-1].copy()

        mid = 0.5 * (A_rs + B_rs)

        # Kreuzungen / Zickzack: Richtungs-Glättung (zusätzlich zur Spline)
        mid = self.direction_filter(mid, max_turn_deg=65.0)

        # Spline glätten
        self.trajectory = self.spline_smooth(mid, self.resample_points, self.smoothing, closed=self.closed_track)

        # für Visualisierung
        self.boundaryA = A_rs
        self.boundaryB = B_rs
        self.midpoints = mid

        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info("✅ Trajektorie generiert (Graph Components → Centerline).")

    # ==========================================================
    # MAP
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
        free_thresh = map_info.get('free_thresh', 0.196)
        occupied_thresh = map_info.get('occupied_thresh', 0.65)

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
                    data.append(0)
                elif pixel < occupied_thresh:
                    data.append(100)
                else:
                    data.append(-1)

        grid.data = data
        return grid

    # ==========================================================
    # CONES
    # ==========================================================
    def load_cones(self, csv_file):
        cones = []
        if not os.path.exists(csv_file):
            raise RuntimeError("cones_csv nicht gefunden")

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cones.append([float(row['x']), float(row['y'])])

        return np.array(cones, dtype=np.float32)

    # ==========================================================
    # GRAPH: connect cones on SAME boundary line
    # ==========================================================
    def build_graph(self, pts, radius, min_dist):
        n = len(pts)
        adj = [[] for _ in range(n)]
        for i in range(n):
            di = pts - pts[i]
            d = np.linalg.norm(di, axis=1)
            for j in range(n):
                if i == j:
                    continue
                if min_dist < d[j] <= radius:
                    adj[i].append(j)
        # optional: sort neighbor lists by distance (stabiler)
        for i in range(n):
            adj[i].sort(key=lambda j: np.linalg.norm(pts[j] - pts[i]))
        return adj

    def connected_components(self, adj):
        n = len(adj)
        vis = [False]*n
        comps = []
        for i in range(n):
            if vis[i]:
                continue
            stack = [i]
            vis[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not vis[v]:
                        vis[v] = True
                        stack.append(v)
            comps.append(comp)
        return comps

    # ==========================================================
    # ORDER COMPONENT as a polyline
    # - if there are endpoints (degree 1), start there
    # - if branches, choose neighbor with smallest turning angle
    # ==========================================================
    def order_component(self, pts, comp, adj):
        comp_set = set(comp)

        # degree within component
        deg = {i: sum((j in comp_set) for j in adj[i]) for i in comp}

        endpoints = [i for i in comp if deg[i] == 1]
        if len(endpoints) >= 1:
            start = endpoints[0]
        else:
            # loop or all deg==2: start at arbitrary
            start = comp[0]

        ordered = [start]
        used = set([start])

        # initial direction: pick closest neighbor in component
        neigh = [j for j in adj[start] if j in comp_set]
        if not neigh:
            return pts[comp]

        current = start
        prev = None
        nxt = neigh[0]

        ordered.append(nxt)
        used.add(nxt)
        prev = current
        current = nxt

        while len(used) < len(comp_set):
            cand = [j for j in adj[current] if j in comp_set and j not in used]
            if not cand:
                # if stuck (branch artifacts), allow revisit but prevent infinite:
                cand = [j for j in adj[current] if j in comp_set and j != prev]
                if not cand:
                    break

            # choose best continuation by smallest turning angle
            v1 = pts[current] - pts[prev]
            n1 = np.linalg.norm(v1)
            if n1 < 1e-6:
                v1 = np.array([1.0, 0.0], dtype=np.float32)
                n1 = 1.0
            v1 = v1 / n1

            best = None
            best_ang = 1e9
            for j in cand:
                v2 = pts[j] - pts[current]
                n2 = np.linalg.norm(v2)
                if n2 < 1e-6:
                    continue
                v2 = v2 / n2
                dot = clamp(float(np.dot(v1, v2)), -1.0, 1.0)
                ang = acos(dot)  # radians
                if ang < best_ang:
                    best_ang = ang
                    best = j

            if best is None:
                break

            ordered.append(best)
            used.add(best)
            prev, current = current, best

        return pts[np.array(ordered, dtype=np.int32)]

    # ==========================================================
    # RESAMPLE polyline by arclength
    # ==========================================================
    def resample_polyline(self, poly, M, closed=False):
        if len(poly) < 2:
            return poly

        if closed:
            p = np.vstack([poly, poly[0]])
        else:
            p = poly

        seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1]
        if L < 1e-6:
            return np.repeat(poly[:1], M, axis=0)

        t = np.linspace(0.0, L, M)
        out = np.zeros((M, 2), dtype=np.float32)

        # piecewise linear interpolation
        k = 0
        for i in range(M):
            while k < len(s)-2 and t[i] > s[k+1]:
                k += 1
            a = s[k]
            b = s[k+1]
            w = 0.0 if (b-a) < 1e-9 else (t[i]-a)/(b-a)
            out[i] = (1.0-w)*p[k] + w*p[k+1]
        return out

    # ==========================================================
    # Direction filter to keep "straight through" at crossings
    # ==========================================================
    def direction_filter(self, pts, max_turn_deg=65.0):
        if len(pts) < 3:
            return pts
        max_turn = np.deg2rad(max_turn_deg)
        keep = [pts[0], pts[1]]
        for i in range(2, len(pts)):
            a = keep[-2]
            b = keep[-1]
            c = pts[i]
            v1 = b - a
            v2 = c - b
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                keep.append(c)
                continue
            v1 /= n1
            v2 /= n2
            ang = acos(clamp(float(np.dot(v1, v2)), -1.0, 1.0))
            if ang > max_turn:
                # too sharp → skip this point (forces straighter path)
                continue
            keep.append(c)
        return np.array(keep, dtype=np.float32)

    # ==========================================================
    # SPLINE
    # ==========================================================
    def spline_smooth(self, pts, M, smoothing, closed=False):
        if len(pts) < 4:
            return pts
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, per=closed)
            u = np.linspace(0, 1, M)
            x, y = splev(u, tck)
            return np.vstack([x, y]).T.astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"Spline failed: {e}")
            return pts

    # ==========================================================
    # PUBLISH
    # ==========================================================
    def publish_all(self):
        now = self.get_clock().now().to_msg()

        # Map
        self.map_msg.header.stamp = now
        self.map_pub.publish(self.map_msg)

        # Cones
        cones_ma = MarkerArray()
        for i, (x, y) in enumerate(self.cones):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "cones"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(self.cone_height / 2.0)
            m.pose.orientation.w = 1.0
            m.scale.x = float(self.cone_radius * 2.0)
            m.scale.y = float(self.cone_radius * 2.0)
            m.scale.z = float(self.cone_height)
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 1.0
            cones_ma.markers.append(m)
        self.cone_pub.publish(cones_ma)

        # Debug markers: boundaryA (blue), boundaryB (yellow), mid (red points), traj (green line)
        dbg = MarkerArray()

        def make_line(ns, mid, pts, rgba, width=0.05):
            line = Marker()
            line.header.frame_id = self.frame_id
            line.header.stamp = now
            line.ns = ns
            line.id = mid
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = width
            line.color.r, line.color.g, line.color.b, line.color.a = rgba
            for p in pts:
                pt = Point()
                pt.x = float(p[0]); pt.y = float(p[1]); pt.z = 0.05
                line.points.append(pt)
            return line

        dbg.markers.append(make_line("boundaryA", 0, self.boundaryA, (0.2, 0.6, 1.0, 1.0), width=0.06))
        dbg.markers.append(make_line("boundaryB", 1, self.boundaryB, (1.0, 1.0, 0.2, 1.0), width=0.06))
        dbg.markers.append(make_line("trajectory", 2, self.trajectory, (0.0, 1.0, 0.0, 1.0), width=0.07))

        # midpoints as spheres
        for i, p in enumerate(self.midpoints[:300]):  # limit marker count
            s = Marker()
            s.header.frame_id = self.frame_id
            s.header.stamp = now
            s.ns = "midpoints"
            s.id = 1000 + i
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.scale.x = 0.08
            s.scale.y = 0.08
            s.scale.z = 0.08
            s.color.r = 1.0
            s.color.g = 0.0
            s.color.b = 0.0
            s.color.a = 0.9
            s.pose.position.x = float(p[0])
            s.pose.position.y = float(p[1])
            s.pose.position.z = 0.05
            s.pose.orientation.w = 1.0
            dbg.markers.append(s)

        self.debug_pub.publish(dbg)

        # Path
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = now
        for p in self.trajectory:
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.05
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = GenerateTraj()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
