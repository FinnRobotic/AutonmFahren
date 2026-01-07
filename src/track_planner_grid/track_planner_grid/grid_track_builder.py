# track_planner_grid/nodes/grid_track_builder.py
import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

from track_planner_grid.occgrid import occgrid_to_numpy, to_binary_occupied
from track_planner_grid.inflate import inflate
from track_planner_grid.edt import compute_edt
from track_planner_grid.skeleton_graph import skeletonize_free, build_skeleton_graph, prune_spurs
from track_planner_grid.loop_select import select_best_loop_through_start
from track_planner_grid.spline import fit_and_resample_closed, fit_and_resample_open
from track_planner_grid.raceline_opt import optimize_raceline_phase1
from track_planner_grid.speed_profile import compute_speed_profile

class GridTrackBuilder(Node):
    def __init__(self):
        super().__init__("grid_track_builder")

        self.sub_map = self.create_subscription(OccupancyGrid, "/map", self.on_map, 1)

        self.pub_center = self.create_publisher(Path, "/centerline", 1)
        self.pub_race = self.create_publisher(Path, "/raceline", 1)
        self.pub_speed = self.create_publisher(Float32MultiArray, "/speed_profile", 1)

        # --- Parameters
        self.declare_parameter("occ_thresh", 50)
        self.declare_parameter("treat_unknown_as_occupied", True)

        self.declare_parameter("inflate_radius_m", 0.30)      # Fahrzeug + Sicherheitsmargin grob
        self.declare_parameter("safe_margin_m", 0.05)         # zusätzlicher Abstand zur Wand
        self.declare_parameter("spur_prune_m", 1.0)           # Skeleton-Spurs entfernen < X m

        self.declare_parameter("resample_ds_m", 0.10)         # Path Punktabstand
        self.declare_parameter("closed_loop", True)           # wenn False: offene Route (später)

        # Loop Auswahl / Start
        self.declare_parameter("start_xy_m", [0.0, 0.0])      # in map frame
        self.declare_parameter("require_through_start", True)
        self.declare_parameter("min_loop_length_m", 20.0)

        # Optimizer
        self.declare_parameter("opt_iters", 80)
        self.declare_parameter("opt_lambda_smooth", 1.0)
        self.declare_parameter("opt_lambda_wall", 5.0)

        # Speed profile
        self.declare_parameter("a_lat_max", 7.0)              # m/s^2 (anpassen)
        self.declare_parameter("a_long_max", 3.0)             # m/s^2
        self.declare_parameter("v_max", 12.0)                 # m/s

        self._built = False

    def on_map(self, msg: OccupancyGrid):
        if self._built:
            return

        grid, res, origin_xy = occgrid_to_numpy(msg)
        occ = to_binary_occupied(
            grid,
            occ_thresh=int(self.get_parameter("occ_thresh").value),
            treat_unknown_as_occupied=bool(self.get_parameter("treat_unknown_as_occupied").value),
        )

        inflate_r_m = float(self.get_parameter("inflate_radius_m").value)
        r_cells = int(np.ceil(inflate_r_m / res))
        occ_inf = inflate(occ, r_cells)

        d = compute_edt(occ_inf, res)
        free = ~occ_inf

        # Skeleton -> Graph
        skel = skeletonize_free(free)
        G = build_skeleton_graph(skel)
        G = prune_spurs(G, min_len_m=float(self.get_parameter("spur_prune_m").value), resolution=res)

        start_xy = np.array(self.get_parameter("start_xy_m").value, dtype=float)
        require_start = bool(self.get_parameter("require_through_start").value)
        min_loop_len = float(self.get_parameter("min_loop_length_m").value)

        loop_poly_world = select_best_loop_through_start(
            G, start_xy, resolution=res, origin_xy=origin_xy,
            require_through_start=require_start,
            min_loop_length_m=min_loop_len,
            edt=d,
        )

        if loop_poly_world is None or len(loop_poly_world) < 10:
            self.get_logger().error("No feasible loop/route found from skeleton graph.")
            return

        closed = bool(self.get_parameter("closed_loop").value)
        ds = float(self.get_parameter("resample_ds_m").value)

        if closed:
            center = fit_and_resample_closed(loop_poly_world, ds=ds)
        else:
            center = fit_and_resample_open(loop_poly_world, ds=ds)

        # Publish centerline
        self.pub_center.publish(self._to_path(center, msg.header.frame_id))

        # Raceline optimization (phase 1)
        d_safe = inflate_r_m + float(self.get_parameter("safe_margin_m").value)
        race = optimize_raceline_phase1(
            center_xy=center,
            edt=d,
            resolution=res,
            origin_xy=origin_xy,
            d_safe=d_safe,
            iters=int(self.get_parameter("opt_iters").value),
            lam_smooth=float(self.get_parameter("opt_lambda_smooth").value),
            lam_wall=float(self.get_parameter("opt_lambda_wall").value),
            closed=closed,
        )
        self.pub_race.publish(self._to_path(race, msg.header.frame_id))

        # Speed profile
        v = compute_speed_profile(
            xy=race,
            ds=ds,
            a_lat_max=float(self.get_parameter("a_lat_max").value),
            a_long_max=float(self.get_parameter("a_long_max").value),
            v_max=float(self.get_parameter("v_max").value),
            closed=closed,
        )
        sp = Float32MultiArray()
        sp.data = [float(x) for x in v]
        self.pub_speed.publish(sp)

        self._built = True
        self.get_logger().info(f"Built: center {len(center)} pts, race {len(race)} pts, speed {len(v)}.")

    def _to_path(self, xy, frame_id):
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

def main():
    rclpy.init()
    node = GridTrackBuilder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
