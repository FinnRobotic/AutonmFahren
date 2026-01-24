#!/usr/bin/env python3
"""
ROS2 node: Track centerline from OccupancyGrid map (PGM+YAML)
Pipeline:
  OccupancyGrid -> Track mask -> remove outside region -> skeletonize -> prune spurs
  -> trace closed loop (choose "middle/straight" arm at junctions)
  -> publish debug maps + centerline markers + Path
  -> save CSV with x,y,yaw,curvature,s

Topics:
  /map_raw          nav_msgs/OccupancyGrid
  /map_track_mask   nav_msgs/OccupancyGrid
  /map_skeleton     nav_msgs/OccupancyGrid
  /track_graph_markers   visualization_msgs/MarkerArray (only centerline gradient)
  /track_centerline_path nav_msgs/Path

Params:
  pgm_file, yaml_file, frame_id
  track_is_free (true: grid==0 is track), treat_unknown_as_obstacle
  spur_prune_iters, node_merge_radius_cells
  publish_rate_hz, marker_line_width
  publish_centerline_path
  save_csv, csv_out, csv_stride
"""

import os
import csv
import yaml
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


_HAVE_SKIMAGE = False
try:
    from skimage.morphology import skeletonize as sk_skeletonize
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


def clamp(x, a, b):
    return max(a, min(b, x))


class TrackGraphFromGrid(Node):
    def __init__(self):
        super().__init__("track_graph_from_grid")

        # -------------------------
        # PARAMETERS
        # -------------------------
        self.declare_parameter("pgm_file", "maps/track_map3.pgm")
        self.declare_parameter("yaml_file", "maps/track_map3.yaml")
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("track_is_free", True)
        self.declare_parameter("treat_unknown_as_obstacle", False)

        self.declare_parameter("spur_prune_iters", 8)
        self.declare_parameter("node_merge_radius_cells", 2)

        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("marker_line_width", 0.06)
        self.declare_parameter("publish_centerline_path", True)

        self.declare_parameter("save_csv", True)

        self.declare_parameter("csv_out", "/home/nvidia/theta_ws/src/track_planner_grid/track/trajectory.csv")
        self.declare_parameter("csv_stride", 1)

        # -------------------------
        # READ PARAMS
        # -------------------------
        self.pgm_file = self.get_parameter("pgm_file").value
        self.yaml_file = self.get_parameter("yaml_file").value
        self.frame_id = self.get_parameter("frame_id").value

        self.track_is_free = bool(self.get_parameter("track_is_free").value)
        self.treat_unknown_as_obstacle = bool(self.get_parameter("treat_unknown_as_obstacle").value)

        self.spur_prune_iters = int(self.get_parameter("spur_prune_iters").value)
        self.node_merge_radius_cells = int(self.get_parameter("node_merge_radius_cells").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.marker_line_width = float(self.get_parameter("marker_line_width").value)
        self.publish_centerline_path = bool(self.get_parameter("publish_centerline_path").value)

        # -------------------------
        # PUBLISHERS
        # -------------------------
        self.pub_map_raw = self.create_publisher(OccupancyGrid, "/map_raw", 1)
        self.pub_map_mask = self.create_publisher(OccupancyGrid, "/map_track_mask", 1)
        self.pub_map_skel = self.create_publisher(OccupancyGrid, "/map_skeleton", 1)

        self.pub_markers = self.create_publisher(MarkerArray, "/track_graph_markers", 10)
        self.pub_path = self.create_publisher(Path, "/track_centerline_path", 10)

        # -------------------------
        # LOAD + PROCESS ONCE
        # -------------------------
        self.map_raw, _ = self.load_map_as_occupancygrid(self.pgm_file, self.yaml_file)
        self.H = self.map_raw.info.height
        self.W = self.map_raw.info.width

        grid_np = np.array(self.map_raw.data, dtype=np.int16).reshape((self.H, self.W))

        # Track mask (255=track)
        track_mask = self.make_track_mask(grid_np)
        track_mask = self.remove_outside_region(track_mask)

        # Skeletonize + prune
        skel = self.skeletonize(track_mask)
        skel_pruned = self.prune_spurs(skel, iters=self.spur_prune_iters)

        # Loop trace
        loop_pixels = self.trace_loop_simple(
            skel_pruned,
            merge_radius=self.node_merge_radius_cells,
            restarts=80
        )

        if loop_pixels and len(loop_pixels) > 20:
            loop_pixels = self.rotate_list_random(loop_pixels)
            self.centerline_pixels = loop_pixels
            self.get_logger().info(f"✅ Loop found: {len(self.centerline_pixels)} points")
        else:
            self.centerline_pixels = []
            self.get_logger().warn("❌ No loop found (simple).")

        # Save CSV (world coords + yaw + curvature)
        if bool(self.get_parameter("save_csv").value) and self.centerline_pixels:
            out_path = self.get_parameter("csv_out").value
            stride = int(self.get_parameter("csv_stride").value)
            self.save_centerline_csv(out_path, stride=stride)
            self.get_logger().info(f"💾 Saved trajectory CSV: {out_path}")

        # RViz occupancy grids
        self.map_mask = self.mask_to_occgrid(track_mask, topic_frame=self.frame_id)
        self.map_skel = self.mask_to_occgrid(skel_pruned, topic_frame=self.frame_id)

        # Timer
        period = 1.0 / max(0.1, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.publish_all)

        self.get_logger().info(f"✅ Ready. (skimage={'yes' if _HAVE_SKIMAGE else 'no'})")

    # ==========================================================
    # MAP LOADING (PGM+YAML -> OccupancyGrid)
    # ==========================================================
    def load_map_as_occupancygrid(self, pgm_file, yaml_file):
        if not os.path.exists(pgm_file):
            raise RuntimeError(f"PGM not found: {pgm_file}")
        if not os.path.exists(yaml_file):
            raise RuntimeError(f"YAML not found: {yaml_file}")

        with open(yaml_file, "r") as f:
            info = yaml.safe_load(f)

        resolution = float(info["resolution"])
        origin = info["origin"]
        negate = int(info.get("negate", 0))
        free_thresh = float(info.get("free_thresh", 0.25))
        occupied_thresh = float(info.get("occupied_thresh", 0.65))

        img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Could not read PGM image")

        H, W = img.shape
        img_u = (255 - img).astype(np.uint8) if negate else img
        img_f = img_u.astype(np.float32) / 255.0

        data = np.full((H, W), -1, dtype=np.int16)
        data[img_f > free_thresh] = 0
        data[img_f < occupied_thresh] = 100

        # Align with RViz (same as earlier code)
        data = np.flipud(data)

        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = resolution
        msg.info.width = W
        msg.info.height = H
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = data.reshape(-1).tolist()
        return msg, {"resolution": resolution, "origin": origin, "H": H, "W": W}

    # ==========================================================
    # TRACK MASK
    # ==========================================================
    def make_track_mask(self, grid_np: np.ndarray) -> np.ndarray:
        if self.track_is_free:
            track = (grid_np == 0)
        else:
            track = (grid_np == 100)

        if self.treat_unknown_as_obstacle:
            track = track & (grid_np != -1)

        return (track.astype(np.uint8) * 255)

    # ==========================================================
    # REMOVE OUTSIDE REGION
    # ==========================================================
    def remove_outside_region(self, track_mask255: np.ndarray) -> np.ndarray:
        m = (track_mask255 > 0).astype(np.uint8)
        H, W = m.shape

        outside = np.zeros((H, W), dtype=np.uint8)
        stack = []

        for c in range(W):
            if m[0, c]: stack.append((0, c))
            if m[H - 1, c]: stack.append((H - 1, c))
        for r in range(H):
            if m[r, 0]: stack.append((r, 0))
            if m[r, W - 1]: stack.append((r, W - 1))

        N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while stack:
            r, c = stack.pop()
            if outside[r, c]:
                continue
            outside[r, c] = 1
            for dr, dc in N4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and m[rr, cc] and not outside[rr, cc]:
                    stack.append((rr, cc))

        inside = (m == 1) & (outside == 0)
        return (inside.astype(np.uint8) * 255)

    # ==========================================================
    # SKELETONIZE
    # ==========================================================
    def skeletonize(self, mask255: np.ndarray) -> np.ndarray:
        bin01 = (mask255 > 0).astype(np.uint8)

        try:
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                skel01 = cv2.ximgproc.thinning(
                    bin01 * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                )
                skel01 = (skel01 > 0).astype(np.uint8)
                return skel01 * 255
        except Exception:
            pass

        if not _HAVE_SKIMAGE:
            raise RuntimeError(
                "No cv2.ximgproc.thinning AND no skimage.skeletonize available. "
                "Install python3-opencv-contrib or python3-skimage."
            )
        skel01 = sk_skeletonize(bin01.astype(bool)).astype(np.uint8)
        return skel01 * 255

    # ==========================================================
    # PRUNE SPURS
    # ==========================================================
    def prune_spurs(self, skel255: np.ndarray, iters: int = 8) -> np.ndarray:
        sk = (skel255 > 0).astype(np.uint8)
        H, W = sk.shape

        def neighbors_count(r, c):
            nb = 0
            for rr in range(max(0, r - 1), min(H - 1, r + 1) + 1):
                for cc in range(max(0, c - 1), min(W - 1, c + 1) + 1):
                    if rr == r and cc == c:
                        continue
                    nb += int(sk[rr, cc] != 0)
            return nb

        for _ in range(max(0, iters)):
            to_remove = []
            ys, xs = np.where(sk != 0)
            for r, c in zip(ys, xs):
                if neighbors_count(r, c) <= 1:
                    to_remove.append((r, c))
            if not to_remove:
                break
            for r, c in to_remove:
                sk[r, c] = 0

        return sk * 255

    # ==========================================================
    # JUNCTION CLUSTERING
    # ==========================================================
    def cluster_junctions(self, skel01: np.ndarray, merge_radius=2):
        sk = skel01.astype(np.uint8)
        H, W = sk.shape
        N8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

        # degree
        deg = np.zeros((H, W), np.uint8)
        ys, xs = np.where(sk)
        for r, c in zip(ys, xs):
            cnt = 0
            for dr, dc in N8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and sk[rr, cc]:
                    cnt += 1
            deg[r, c] = cnt

        junc = (sk == 1) & (deg >= 3)

        # merge nearby by dilating junction mask
        if merge_radius > 0:
            k = 2 * merge_radius + 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            junc = cv2.dilate(junc.astype(np.uint8) * 255, ker) > 0
            junc = junc & (sk == 1)

        cluster_id = -np.ones((H, W), np.int32)
        cid = 0
        for r, c in zip(*np.where(junc)):
            if cluster_id[r, c] != -1:
                continue
            stack = [(r, c)]
            cluster_id[r, c] = cid
            while stack:
                rr, cc = stack.pop()
                for dr, dc in N8:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < H and 0 <= c2 < W and junc[r2, c2] and cluster_id[r2, c2] == -1:
                        cluster_id[r2, c2] = cid
                        stack.append((r2, c2))
            cid += 1

        return cluster_id, cid  # cid = number of clusters

    def get_ports(self, skel01, cluster_id, cid):
        H, W = skel01.shape
        N8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

        ports = set()
        ys, xs = np.where(cluster_id == cid)
        for r, c in zip(ys, xs):
            for dr, dc in N8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and skel01[rr, cc] and cluster_id[rr, cc] != cid:
                    ports.add((rr, cc))
        return list(ports)

    def follow_arm_direction(self, skel01, cluster_id, start, steps=8):
        H, W = skel01.shape
        N8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

        prev = None
        cur = start

        for _ in range(steps):
            nbrs = []
            for dr, dc in N8:
                rr, cc = cur[0] + dr, cur[1] + dc
                if 0 <= rr < H and 0 <= cc < W and skel01[rr, cc]:
                    if prev is not None and (rr, cc) == prev:
                        continue
                    # don't go into junction clusters
                    if cluster_id[rr, cc] != -1:
                        continue
                    nbrs.append((rr, cc))
            if not nbrs:
                break

            # pick neighbor that is straightest relative to prev->cur if possible
            nxt = nbrs[0]
            if prev is not None and len(nbrs) > 1:
                v_in = np.array([cur[0] - prev[0], cur[1] - prev[1]], np.float32)
                n = np.linalg.norm(v_in)
                v_in = v_in / n if n > 1e-6 else np.array([1.0, 0.0], np.float32)
                best_dot = -1e9
                for cand in nbrs:
                    v_out = np.array([cand[0] - cur[0], cand[1] - cur[1]], np.float32)
                    m = np.linalg.norm(v_out)
                    if m < 1e-6:
                        continue
                    v_out /= m
                    d = float(np.dot(v_in, v_out))
                    if d > best_dot:
                        best_dot = d
                        nxt = cand

            prev, cur = cur, nxt

        v = np.array([cur[0] - start[0], cur[1] - start[1]], np.float32)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.array([1.0, 0.0], np.float32)
        return v / n

    def choose_middle_arm(self, prev_pix, cur_pix, ports, skel01, cluster_id):
        v_in = np.array([cur_pix[0] - prev_pix[0], cur_pix[1] - prev_pix[1]], np.float32)
        n = np.linalg.norm(v_in)
        v_in = v_in / n if n > 1e-6 else np.array([1.0, 0.0], np.float32)

        best_port = None
        best_dot = -1e9
        for p in ports:
            if p == prev_pix:  # avoid immediate U-turn
                continue
            v_out = self.follow_arm_direction(skel01, cluster_id, p, steps=8)
            d = float(np.dot(v_in, v_out))
            if d > best_dot:
                best_dot = d
                best_port = p
        return best_port

    def trace_loop_simple(self, skel255, merge_radius=2, restarts=50, max_steps=200000):
        sk = (skel255 > 0).astype(np.uint8)
        H, W = sk.shape
        N8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

        cluster_id, _num = self.cluster_junctions(sk, merge_radius=merge_radius)

        ys, xs = np.where(sk)
        if len(ys) == 0:
            return []

        rng = np.random.default_rng()

        def neighbors(p):
            r, c = p
            out = []
            for dr, dc in N8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and sk[rr, cc]:
                    out.append((rr, cc))
            return out

        for _ in range(restarts):
            start = (int(rng.choice(ys)), int(rng.choice(xs)))
            if cluster_id[start[0], start[1]] != -1:
                continue

            path = [start]
            prev = None
            cur = start
            visited_edges = set()

            for _step in range(max_steps):
                nbrs = neighbors(cur)
                if prev is not None:
                    nbrs = [p for p in nbrs if p != prev]
                if not nbrs:
                    break

                # choose straight locally if multiple
                nxt = nbrs[0]
                if prev is not None and len(nbrs) > 1:
                    v_in = np.array([cur[0] - prev[0], cur[1] - prev[1]], np.float32)
                    n = np.linalg.norm(v_in)
                    v_in = v_in / n if n > 1e-6 else np.array([1.0, 0.0], np.float32)
                    best_dot = -1e9
                    for cand in nbrs:
                        v_out = np.array([cand[0] - cur[0], cand[1] - cur[1]], np.float32)
                        m = np.linalg.norm(v_out)
                        if m < 1e-6:
                            continue
                        v_out /= m
                        d = float(np.dot(v_in, v_out))
                        if d > best_dot:
                            best_dot = d
                            nxt = cand

                # entering junction? decide arm and jump to port
                cid = cluster_id[nxt[0], nxt[1]]
                if cid != -1 and prev is not None:
                    ports = self.get_ports(sk, cluster_id, cid)
                    best_port = self.choose_middle_arm(prev_pix=cur, cur_pix=nxt, ports=ports,
                                                       skel01=sk, cluster_id=cluster_id)
                    if best_port is None:
                        break
                    path.append(nxt)  # for visualization continuity
                    prev, cur = cur, best_port
                    path.append(cur)
                else:
                    prev, cur = cur, nxt
                    path.append(cur)

                edge = (prev, cur)
                if edge in visited_edges:
                    break
                visited_edges.add(edge)

                if cur == start and len(path) > 80:
                    return path

        return []

    def rotate_list_random(self, lst):
        if not lst:
            return lst
        k = int(np.random.randint(0, len(lst)))
        return lst[k:] + lst[:k]

    # ==========================================================
    # CSV: curvature + yaw
    # ==========================================================
    def compute_curvature_and_yaw(
        self,
        xy: np.ndarray,
        closed: bool = True,
        ds_target: float = 0.10,     # gewünschter Punktabstand [m]
        smooth_win: int = 21,        # Fenstergröße (ungerade!) in Samples nach Resample
        curv_lpf_win: int = 31       # extra Glättung nur für curvature
    ):
        """
        Smooth yaw + curvature for a (closed) loop trajectory.

        Steps:
        1) Resample by arc length to approximately constant spacing ds_target
        2) Smooth x,y (Savitzky-Golay if available, else circular moving average)
        3) Compute yaw + curvature from derivatives wrt s
        4) Lowpass curvature again (circular moving average)

        Returns:
        yaw (N,), curvature (N,), s (N,), xy_rs (N,2)
        """
        xy = np.asarray(xy, dtype=np.float32)
        N0 = len(xy)
        if N0 < 5:
            yaw = np.zeros((N0,), np.float32)
            curv = np.zeros((N0,), np.float32)
            s = np.zeros((N0,), np.float32)
            return yaw, curv, s, xy

        # --------------------------
        # 1) Arc-length resample
        # --------------------------
        if closed:
            pts = np.vstack([xy, xy[0]])
        else:
            pts = xy.copy()

        seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        L = float(np.sum(seg))
        if L < 1e-6:
            yaw = np.zeros((N0,), np.float32)
            curv = np.zeros((N0,), np.float32)
            s = np.zeros((N0,), np.float32)
            return yaw, curv, s, xy

        s_src = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)

        M = max(10, int(round(L / max(1e-3, ds_target))))
        s_tgt = np.linspace(0.0, L, M, dtype=np.float32)

        # linear interpolation x(s), y(s)
        x_src = pts[:, 0].astype(np.float32)
        y_src = pts[:, 1].astype(np.float32)
        x = np.interp(s_tgt, s_src, x_src).astype(np.float32)
        y = np.interp(s_tgt, s_src, y_src).astype(np.float32)

        xy_rs = np.stack([x, y], axis=1)

        # --------------------------
        # helper: circular moving average
        # --------------------------
        def circ_movavg(a: np.ndarray, win: int) -> np.ndarray:
            win = int(win)
            if win < 3:
                return a
            if win % 2 == 0:
                win += 1
            pad = win // 2
            ap = np.concatenate([a[-pad:], a, a[:pad]])
            k = np.ones((win,), dtype=np.float32) / float(win)
            out = np.convolve(ap, k, mode="valid").astype(np.float32)
            return out

        # --------------------------
        # 2) Smooth x,y
        # --------------------------
        win = int(smooth_win)
        if win % 2 == 0:
            win += 1
        win = max(5, min(win, M - (1 - M % 2)))  # <= M and odd

        used_savgol = False
        try:
            # optional: Savitzky-Golay for smooth derivatives
            from scipy.signal import savgol_filter
            # polyorder 3 is usually stable for tracks
            x_s = savgol_filter(x, window_length=win, polyorder=3, mode="wrap" if closed else "interp").astype(np.float32)
            y_s = savgol_filter(y, window_length=win, polyorder=3, mode="wrap" if closed else "interp").astype(np.float32)
            used_savgol = True
        except Exception:
            x_s = circ_movavg(x, win) if closed else np.convolve(x, np.ones(win)/win, mode="same").astype(np.float32)
            y_s = circ_movavg(y, win) if closed else np.convolve(y, np.ones(win)/win, mode="same").astype(np.float32)

        # --------------------------
        # 3) Derivatives wrt s (ds ~ constant)
        # --------------------------
        ds = float(L / max(1, (M - 1)))  # approx constant spacing

        if used_savgol:
            # if we already have savgol, we can get smooth derivatives directly
            try:
                from scipy.signal import savgol_filter
                dx = savgol_filter(x, window_length=win, polyorder=3, deriv=1, delta=ds,
                                mode="wrap" if closed else "interp").astype(np.float32)
                dy = savgol_filter(y, window_length=win, polyorder=3, deriv=1, delta=ds,
                                mode="wrap" if closed else "interp").astype(np.float32)
                ddx = savgol_filter(x, window_length=win, polyorder=3, deriv=2, delta=ds,
                                    mode="wrap" if closed else "interp").astype(np.float32)
                ddy = savgol_filter(y, window_length=win, polyorder=3, deriv=2, delta=ds,
                                    mode="wrap" if closed else "interp").astype(np.float32)
            except Exception:
                # fallback if deriv call fails for any reason
                dx = np.gradient(x_s, ds).astype(np.float32)
                dy = np.gradient(y_s, ds).astype(np.float32)
                ddx = np.gradient(dx, ds).astype(np.float32)
                ddy = np.gradient(dy, ds).astype(np.float32)
        else:
            dx = np.gradient(x_s, ds).astype(np.float32)
            dy = np.gradient(y_s, ds).astype(np.float32)
            ddx = np.gradient(dx, ds).astype(np.float32)
            ddy = np.gradient(dy, ds).astype(np.float32)

        yaw = np.arctan2(dy, dx).astype(np.float32)

        # curvature formula
        denom = (dx * dx + dy * dy) ** 1.5
        denom = np.maximum(denom, 1e-8).astype(np.float32)
        curv = (dx * ddy - dy * ddx) / denom
        curv = curv.astype(np.float32)

        # --------------------------
        # 4) Extra lowpass on curvature (really helps)
        # --------------------------
        win2 = int(curv_lpf_win)
        if win2 % 2 == 0:
            win2 += 1
        win2 = max(5, min(win2, M - (1 - M % 2)))

        if closed:
            curv = circ_movavg(curv, win2)
            # yaw also a bit (optional, but usually nice)
            # use sin/cos averaging to respect wrap-around
            yaw_sin = circ_movavg(np.sin(yaw).astype(np.float32), win2)
            yaw_cos = circ_movavg(np.cos(yaw).astype(np.float32), win2)
            yaw = np.arctan2(yaw_sin, yaw_cos).astype(np.float32)
        else:
            k = np.ones((win2,), dtype=np.float32) / float(win2)
            curv = np.convolve(curv, k, mode="same").astype(np.float32)

        # s output
        s = s_tgt.astype(np.float32)

        # return resampled trajectory outputs (note: now length M)
        return yaw, curv, s, np.stack([x_s, y_s], axis=1).astype(np.float32)


    def save_centerline_csv(self, csv_path: str, stride: int = 1):
        if not self.centerline_pixels:
            self.get_logger().warn("No centerline_pixels to save.")
            return

        stride = max(1, int(stride))

        # rc -> world xy
        xy = []
        for (r, c) in self.centerline_pixels[::stride]:
            x, y = self.rc_to_world(r, c)
            xy.append([x, y])
        xy = np.array(xy, dtype=np.float32)

        # NEW: smooth + resample + curvature
        yaw, curv, s, xy_smooth = self.compute_curvature_and_yaw(
            xy,
            closed=True,
            ds_target=0.10,      # <- tunen
            smooth_win=21,       # <- tunen
            curv_lpf_win=31      # <- tunen
        )

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "yaw", "curvature", "s"])
            for i in range(len(xy_smooth)):
                w.writerow([
                    float(xy_smooth[i, 0]),
                    float(xy_smooth[i, 1]),
                    float(yaw[i]),
                    float(curv[i]),
                    float(s[i]),
                ])


    # ==========================================================
    # OccupancyGrid helper
    # ==========================================================
    def mask_to_occgrid(self, mask255: np.ndarray, topic_frame="map") -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = topic_frame
        msg.info = self.map_raw.info
        occ = np.full(mask255.shape, 100, dtype=np.int16)
        occ[mask255 > 0] = 0
        msg.data = occ.reshape(-1).tolist()
        return msg

    # ==========================================================
    # GRID -> WORLD
    # ==========================================================
    def rc_to_world(self, r, c):
        res = self.map_raw.info.resolution
        ox = self.map_raw.info.origin.position.x
        oy = self.map_raw.info.origin.position.y
        return float(ox + (c + 0.5) * res), float(oy + (r + 0.5) * res)

    # ==========================================================
    # Color ramp: white -> yellow -> orange -> red
    # ==========================================================
    def ramp_white_yellow_orange_red(self, t: float) -> ColorRGBA:
        t = float(clamp(t, 0.0, 1.0))
        c = ColorRGBA()
        c.a = 1.0

        # 0.0 .. 0.33: white (1,1,1) -> yellow (1,1,0)
        if t < 1.0 / 3.0:
            u = t / (1.0 / 3.0)
            c.r = 1.0
            c.g = 1.0
            c.b = 1.0 - u  # 1 -> 0
            return c

        # 0.33 .. 0.66: yellow (1,1,0) -> orange (1,0.5,0)
        if t < 2.0 / 3.0:
            u = (t - 1.0 / 3.0) / (1.0 / 3.0)
            c.r = 1.0
            c.g = 1.0 - 0.5 * u  # 1 -> 0.5
            c.b = 0.0
            return c

        # 0.66 .. 1.0: orange (1,0.5,0) -> red (1,0,0)
        u = (t - 2.0 / 3.0) / (1.0 / 3.0)
        c.r = 1.0
        c.g = 0.5 - 0.5 * u  # 0.5 -> 0
        c.b = 0.0
        return c

    # ==========================================================
    # PUBLISH
    # ==========================================================
    def publish_all(self):
        now = self.get_clock().now().to_msg()

        self.map_raw.header.stamp = now
        self.map_mask.header.stamp = now
        self.map_skel.header.stamp = now

        self.pub_map_raw.publish(self.map_raw)
        self.pub_map_mask.publish(self.map_mask)
        self.pub_map_skel.publish(self.map_skel)

        ma = MarkerArray()

        # Centerline marker (gradient)
        if self.centerline_pixels:
            cl = Marker()
            cl.header.frame_id = self.frame_id
            cl.header.stamp = now
            cl.ns = "centerline_grad"
            cl.id = 0
            cl.type = Marker.LINE_STRIP
            cl.action = Marker.ADD
            cl.scale.x = self.marker_line_width
            cl.pose.orientation.w = 1.0

            N = len(self.centerline_pixels)
            for i, (r, c) in enumerate(self.centerline_pixels):
                x, y = self.rc_to_world(r, c)
                cl.points.append(Point(x=x, y=y, z=0.07))
                t = i / max(1, (N - 1))
                cl.colors.append(self.ramp_white_yellow_orange_red(t))

            ma.markers.append(cl)

        self.pub_markers.publish(ma)

        # Path
        if self.publish_centerline_path and self.centerline_pixels:
            path = Path()
            path.header.frame_id = self.frame_id
            path.header.stamp = now
            for (r, c) in self.centerline_pixels:
                x, y = self.rc_to_world(r, c)
                ps = PoseStamped()
                ps.header.frame_id = self.frame_id
                ps.header.stamp = now
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.position.z = 0.05
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.pub_path.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = TrackGraphFromGrid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
