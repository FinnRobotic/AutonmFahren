import math, numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from theta_msgs.msg import PathWithCurvature
from ackermann_msgs.msg import AckermannDriveStamped


from std_msgs.msg import Bool


def wrap_pi(a): 
    return (a + math.pi) % (2*math.pi) - math.pi


class PurePursuitPID(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # Parameters
        self.declare_parameter('vehicle_state_topic', '/slam/vehicle_state')
        self.declare_parameter('path_topic', '/trajectory/middle_line')
        self.declare_parameter('cmd_topic', '/controller/actuator_request')

        self.declare_parameter('wheelbase', 0.325)         # m
        self.declare_parameter('lookahead', 0.3)         # m (smaller at low v, larger at high v)
        self.declare_parameter('v_ref', 1.5)             # m/s

        # --- Dynamic Driving parameters ---
        # v_ref = clamp( sqrt(a_lat_max / |kappa|), v_ref_min, v_ref_max )
        self.declare_parameter("a_lat_max", 6.0)  # m/s^2
        self.declare_parameter("v_ref_min", 0.1)  # m/s
        self.declare_parameter("v_ref_max", 4.0)  # m/s
        self.declare_parameter("v_ref_smoothing_alpha", 0.3)  # 0..1
        self.declare_parameter("use_curvature_speed", True)

        # --- Dynamic lookahead distance when dynamic driving allowed ---
        # Ld = clamp( Ld_base + Ld_gain * v, Ld_min, Ld_max )
        self.declare_parameter("Ld_base", 0.4)  # m
        self.declare_parameter("Ld_gain", 0.1)  # m per (m/s)
        self.declare_parameter("Ld_min", 0.1)   # m
        self.declare_parameter("Ld_max", 3.0)  # m

        self.declare_parameter('kp_steer', 0.0)
        self.declare_parameter('ki_steer', 0.0)
        self.declare_parameter('steer_int_limit', 2.0)
        self.declare_parameter('kd_steer', 0.0)
        self.declare_parameter('kp_psi_steer', 0.2)
        self.declare_parameter('pid_clip_deg', 10.0)
        self.declare_parameter('steer_limit_deg', 30.0)
        self.declare_parameter('alpha', 0.3)

        self.declare_parameter('kp_speed', 0.0)
        self.declare_parameter('ki_speed', 0.0)
        self.declare_parameter('speed_int_limit', 2.0)
        self.declare_parameter('kd_speed', 0.0)
        self.declare_parameter('torque_limit', 100.0)

        self.declare_parameter('min_path_size', 5)

        # Resolve params
        self.vehicle_state_topic = self.get_parameter('vehicle_state_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.cmd_topic  = self.get_parameter('cmd_topic').value

        self.L          = float(self.get_parameter('wheelbase').value)
        self.Ld         = float(self.get_parameter('lookahead').value)
        self.v_ref      = float(self.get_parameter('v_ref').value)

        # dyn speed params
        self.a_lat_max = float(self.get_parameter("a_lat_max").value)
        self.v_ref_min = float(self.get_parameter("v_ref_min").value)
        self.v_ref_max = float(self.get_parameter("v_ref_max").value)
        self.vref_alpha = float(self.get_parameter("v_ref_smoothing_alpha").value)
        self.use_curv_speed = bool(self.get_parameter("use_curvature_speed").value)

        # dyn lookahead params
        self.Ld_base = float(self.get_parameter("Ld_base").value)
        self.Ld_gain = float(self.get_parameter("Ld_gain").value)
        self.Ld_min = float(self.get_parameter("Ld_min").value)
        self.Ld_max = float(self.get_parameter("Ld_max").value)

        # steering pid
        self.kp_steer       = float(self.get_parameter('kp_steer').value)
        self.ki_steer       = float(self.get_parameter('ki_steer').value)
        self.steer_i_lim    = float(self.get_parameter('steer_int_limit').value)
        self.kd_steer       = float(self.get_parameter('kd_steer').value)
        self.kp_psi_steer   = float(self.get_parameter('kp_psi_steer').value)
        self.pid_clip_deg   = float(self.get_parameter('pid_clip_deg').value)
        self.steer_lim      = math.radians(float(self.get_parameter('steer_limit_deg').value))
        self.alpha          = float(self.get_parameter('alpha').value)

        # speed pid
        self.kp_speed       = float(self.get_parameter('kp_speed').value)
        self.ki_speed       = float(self.get_parameter('ki_speed').value)
        self.speed_i_lim    = float(self.get_parameter('speed_int_limit').value)
        self.kd_speed       = float(self.get_parameter('kd_speed').value)
        self.torque_limit   = float(self.get_parameter('torque_limit').value)

        self.min_path   = int(self.get_parameter('min_path_size').value)

        # IO
        self.sub_path = self.create_subscription(
            PathWithCurvature, 
            self.path_topic, 
            self.on_path, 
            1
        )

        # immer ActuatorRequest (real commands)
        self.pub_cmd = self.create_publisher(
            AckermannDriveStamped, 
            '/drive', 
            1
        )

        self.sub_veh_state = self.create_subscription(
            Odometry, 
            '/zed/zed_node/odom', 
            self.on_vehicle_state, 
            10
        )

        self.stop_sub = self.create_subscription(
            Bool, 
            '/master/stop',
            self.stop_callback,
            10
        )
        self.stop = True

        self.steer_acutation_allowed_sub = self.create_subscription(
            Bool, 
            '/master/steering_actuation_allowed', 
            self.steer_allowed_callback, 
            10
        )
        self.steer_allowed = False

        self.dynamic_driving_allowed_sub = self.create_subscription(
            Bool,
            '/master/allow_dynamic_driving',
            self.dynamic_driving_callback,
            10
        )
        self.dynamic_driving_allowed = False

        # State
        self.path_xy = None
        self.path_s  = None
        self.path_curv = None
        self.last_odom_t = None

        # PID memory
        self.int_y = 0.0
        self.prev_y = None
        self.last_delta = 0.0
        
        # Speed controller integrator
        self.int_torque = 0.0
        self.prev_v_error = None
        self.last_v_ref_cmd = None

        self.get_logger().info("PurePursuitPID ready.")


    def stop_callback(self, msg: Bool):
        self.stop = msg.data
        self.get_logger().info(f"Received Stop-MSG, stop = {msg.data}")

    def steer_allowed_callback(self, msg: Bool):
        self.steer_allowed = msg.data
        self.get_logger().info(f"Received Steer-MSG, steer_allowed = {msg.data}")

    def dynamic_driving_callback(self, msg: Bool):
        self.dynamic_driving_allowed = msg.data
        self.get_logger().info(f"Received Dynamic Driving-MSG | Dynamic Driving allowed = {msg.data}")

    def on_path(self, msg: PathWithCurvature):
        if len(msg.poses) < self.min_path:
            self.path_xy = None
            self.path_s = None
            self.path_curv = None
            return

        xy = np.array([[p.position.x, p.position.y] for p in msg.poses], dtype=float)
        c = np.array([float(p.curvature) for p in msg.poses], dtype=float)

        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        self.path_xy, self.path_s, self.path_curv = xy, s, c

    def on_vehicle_state(self, state: Odometry):
        if self.path_xy is None:
            return

        if self.stop:
            self.stop_vehicle(state)
            return
        
        # Current pose
        x = state.pose.pose.position.x
        y = state.pose.pose.position.y


        w = state.pose.orientation.w
        z = state.pose.orientation.z
        yaw = math.atan2(2 * (w * z), 1 - 2 * (z ** 2))
        # Projection point on path
        p = np.array([x, y])
        d2 = np.sum((self.path_xy - p) ** 2, axis=1)
        i0 = int(np.argmin(d2))


        # Lookahead distance
        if self.dynamic_driving_allowed:
            Ld_eff = min(max(self.Ld_base + self.Ld_gain * v, self.Ld_min), self.Ld_max)
        else:
            Ld_eff = self.Ld

        # Lookahead target
        s0 = self.path_s[i0]
        s_target = s0 + self.Ld
        j = np.searchsorted(self.path_s, s_target, side='left')
        j = min(max(j, i0 + 1), len(self.path_s) - 1)
        p_star = self.path_xy[j]

        # Tangent / heading error
        if i0 < len(self.path_xy) - 1:
            t_vec = self.path_xy[i0 + 1] - self.path_xy[i0]
        else:
            t_vec = self.path_xy[i0] - self.path_xy[i0 - 1]

        psi_path = math.atan2(t_vec[1], t_vec[0])
        e_psi = wrap_pi(yaw - psi_path)

        # Lateral error e_y (in path frame)
        R = np.array(
            [
                [math.cos(-psi_path), -math.sin(-psi_path)],
                [math.sin(-psi_path),  math.cos(-psi_path)]
            ]
        )
        e_xy = R @ (p - self.path_xy[i0])
        e_y = e_xy[1]

        # Lookahead point relative to vehicle
        Rcar = np.array([
            [math.cos(-yaw), -math.sin(-yaw)],
            [math.sin(-yaw),  math.cos(-yaw)]
        ])
        p_rel = Rcar @ (p_star - p)
        x_lh, y_lh = float(p_rel[0]), float(p_rel[1])

        # Pure Pursuit curvature -> steering
        if Ld_eff < 1e-3 or x_lh < 1e-3:
            delta_pp = 0.0
        else:
            kappa_pp = 2.0 * y_lh / (Ld_eff ** 2)
            delta_pp = math.atan(self.L * kappa_pp)

        # PID correction
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, now_sec - self.last_odom_t) if self.last_odom_t else 0.0
        self.last_odom_t = now_sec

        dy = (e_y - self.prev_y) / dt if self.prev_y is not None else 0.0
        self.prev_y = e_y

        self.int_y += e_y * dt
        self.int_y = float(np.clip(self.int_y, -self.steer_i_lim, self.steer_i_lim))

        delta_pid = (
            self.kp_steer * e_y +
            self.kd_steer * dy +
            self.ki_steer * self.int_y +
            self.kp_psi_steer * e_psi
        )
        delta_pid = float(np.clip(
            delta_pid,
            -math.radians(self.pid_clip_deg),
            math.radians(self.pid_clip_deg)
        ))

        delta = delta_pp - delta_pid
        delta = float(np.clip(delta, -self.steer_lim, self.steer_lim))

        # Smooth steering
        delta = self.alpha * delta + (1.0 - self.alpha) * self.last_delta
        self.last_delta = delta

        if self.dynamic_driving_allowed and self.use_curv_speed and (self.path_curv is not None):
            curv_lh = float(self.path_curv[j])
            abs_curv = abs(curv_lh)
            if abs_curv <= 1e-6:
                v_cmd = self.v_ref_max
            else:
                v_cmd = math.sqrt(max(self.a_lat_max, 0.0) / abs_curv)
            v_cmd = float(np.clip(v_cmd, self.v_ref_min, self.v_ref_max))
            if self.last_v_ref_cmd is None:
                v_ref_cmd = v_cmd
            else:
                a = float(np.clip(self.vref_alpha, 0.0, 1.0))
                v_ref_cmd = a * v_cmd + (1.0 - a) * self.last_v_ref_cmd
            self.last_v_ref_cmd = v_ref_cmd
        else:
            v_ref_cmd = self.v_ref

       
        drive = AckermannDriveStamped()

        drive.drive.steering_angle = delta
        drive.drive.speed = v_ref_cmd
        drive.header.stamp = self.get_clock().now().to_msg()
        self.pub_drive.publish(drive)
        
        self.get_logger().info(
            f"CMD: | target={v_ref_cmd:.2f} | lookahead={Ld_eff:.2f} | steering={delta:.2f} rad ({math.degrees(delta):.2f} deg)"
        )

    def stop_vehicle(self, state: Odometry):
        # Current pose
        x = state.pose.pose.position.x
        y = state.pose.pose.position.y


        w = state.pose.orientation.w
        z = state.pose.orientation.z
        yaw = math.atan2(2 * (w * z), 1 - 2 * (z ** 2))

        p = np.array([x, y])
        d2 = np.sum((self.path_xy - p) ** 2, axis=1)
        i0 = int(np.argmin(d2))

        s0 = self.path_s[i0]
        s_target = s0 + self.Ld
        j = np.searchsorted(self.path_s, s_target, side='left')
        j = min(max(j, i0 + 1), len(self.path_s) - 1)
        p_star = self.path_xy[j]

        if i0 < len(self.path_xy) - 1:
            t_vec = self.path_xy[i0 + 1] - self.path_xy[i0]
        else:
            t_vec = self.path_xy[i0] - self.path_xy[i0 - 1]
        psi_path = math.atan2(t_vec[1], t_vec[0])
        e_psi = wrap_pi(yaw - psi_path)

        R = np.array([
            [math.cos(-psi_path), -math.sin(-psi_path)],
            [math.sin(-psi_path),  math.cos(-psi_path)]
        ])
        e_xy = R @ (p - self.path_xy[i0])
        e_y = e_xy[1]

        Rcar = np.array([
            [math.cos(-yaw), -math.sin(-yaw)],
            [math.sin(-yaw),  math.cos(-yaw)]
        ])
        p_rel = Rcar @ (p_star - p)
        x_lh, y_lh = float(p_rel[0]), float(p_rel[1])

        if self.Ld < 1e-3 or x_lh < 1e-3:
            delta_pp = 0.0
        else:
            kappa = 2.0 * y_lh / (self.Ld ** 2)
            delta_pp = math.atan(self.L * kappa)

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, now_sec - self.last_odom_t) if self.last_odom_t else 0.0
        self.last_odom_t = now_sec

        dy = (e_y - self.prev_y) / dt if self.prev_y is not None else 0.0
        self.prev_y = e_y

        self.int_y += e_y * dt
        self.int_y = float(np.clip(self.int_y, -self.steer_i_lim, self.steer_i_lim))

        delta_pid = (
            self.kp_steer * e_y +
            self.kd_steer * dy +
            self.ki_steer * self.int_y +
            self.kp_psi_steer * e_psi
        )
        delta_pid = float(np.clip(
            delta_pid,
            -math.radians(self.pid_clip_deg),
            math.radians(self.pid_clip_deg)
        ))

        delta = delta_pp - delta_pid
        delta = float(np.clip(delta, -self.steer_lim, self.steer_lim))

        if self.steer_allowed:
            delta = self.alpha * delta + (1.0 - self.alpha) * self.last_delta
            self.last_delta = delta
        else:
            delta = self.last_delta


        drive = AckermannDriveStamped()

        drive.drive.steering_angle = delta
        drive.drive.speed = 0.0
        drive.header.stamp = self.get_clock().now().to_msg()
        self.pub_drive.publish(drive)

        self.get_logger().info(
            f"Braking-CMD: target={0.0:.2f} | steering={delta:.2f}"
        )


def main():
    rclpy.init()
    node = PurePursuitPID()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()