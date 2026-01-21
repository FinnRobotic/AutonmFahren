import math
import numpy as np

import rclpy
from rclpy.node import Node

from as_msgs.msg import ActuatorRequest
from fs_msgs.msg import ControlCommand


class ActuatorToFsds(Node):
    def __init__(self):
        super().__init__('actuator_to_fsds')

        # Parameter
        self.declare_parameter('in_topic', '/controller/actuator_request')
        self.declare_parameter('out_topic', '/control_command')
        self.declare_parameter('steer_limit_deg', 30.0)
        self.declare_parameter('torque_limit', 100.0)

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value
        steer_limit_deg = float(self.get_parameter('steer_limit_deg').value)
        self.torque_limit = float(self.get_parameter('torque_limit').value)

        self.steer_lim = math.radians(steer_limit_deg)

        self.sub_cmd = self.create_subscription(
            ActuatorRequest,
            in_topic,
            self.on_actuator_request,
            10
        )

        self.pub_cmd = self.create_publisher(
            ControlCommand,
            out_topic,
            10
        )

        self.get_logger().info(
            f"ActuatorToFsds ready. Subscribing on '{in_topic}', publishing FSDS ControlCommand on '{out_topic}'."
        )

    def on_actuator_request(self, msg: ActuatorRequest):

        torque = float(msg.backleft + msg.backright)
        delta = float(msg.steering)  # [rad]


        steering = -float(np.clip(delta / self.steer_lim, -1.0, 1.0))

        # Throttle / Brake aus Torque
        if torque >= 0.0:
            throttle = float(np.clip(torque / self.torque_limit, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip((-torque) / self.torque_limit, 0.0, 1.0))

        cmd = ControlCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.steering = steering
        cmd.throttle = throttle
        cmd.brake = brake

        self.pub_cmd.publish(cmd)
        self.get_logger().debug(
            f"FSDS-CMD: torque={torque:.2f} -> throttle={throttle:.2f}, brake={brake:.2f}, steering={steering:.2f}"
        )


def main():
    rclpy.init()
    node = ActuatorToFsds()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()