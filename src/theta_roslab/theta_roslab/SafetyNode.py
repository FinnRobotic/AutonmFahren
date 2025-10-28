import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64

import numpy as np
import math

TTC_TRESHOLD = 0.5


def timte_to_collision(velocity, angle, dist):

    
    v = np.cos(angle) * velocity

    if v==0:
        return 1000.0
    
    return dist / max(v, 0.00000001)


class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')


        self.sub_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.cb_lidar,
            10
        )
        self.lidar_msg = None
        self.valid_ranges = None

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.cb_odom,
            10
        )
        self.odom_msg = None

        self.pub_ttc = self.create_publisher(
            Float64,
            '/ttc',
            10
        )


        self.pub_brake = self.create_publisher(
            AckermannDriveStamped,
            '/brake',
            10
        )




        self.threshold_violated = False
        self.last_v = None


        self.ttc_timer = self.create_timer(
            0.05,
            self.ttc_cb
        )

        self.pub_timer = self.create_timer(
            0.01,
            self.pub_cb
        )

    def cb_lidar(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        self.valid_ranges = ranges[np.isfinite(ranges)]
        self.lidar_msg = msg

    def cb_odom(self, msg: Odometry):
        self.odom_msg = msg


    def ttc_cb(self):
        if self.odom_msg is None or self.lidar_msg is None:
            return
        

        angle = self.lidar_msg.angle_min
        angle_incr = self.lidar_msg.angle_increment

        vx = -self.odom_msg.twist.twist.linear.x
        vy = -self.odom_msg.twist.twist.linear.y

        if self.threshold_violated:
            vx = self.last_v[0]
            vy = self.last_v[1]

        v = math.sqrt(vx**2 + vy**2)


        if abs(v) == 0:
            normed_vx = 0
            normed_vy = 0
        else:
            normed_vx = vx / v
            normed_vy = vy / v

        theta_v = math.atan2(normed_vy, normed_vx)

        ttc_min = 10000000.0


        for range in self.valid_ranges:

            ttc = timte_to_collision(v, theta_v - angle, range)
            if ttc < TTC_TRESHOLD:
                self.threshold_violated = True
                ttc_msg = Float64()
                ttc_msg.data = ttc
                self.last_v = np.array([vx, vy])
                self.pub_ttc.publish(ttc_msg)
                return
            elif ttc < ttc_min:
                ttc_min = ttc

            angle = angle + angle_incr

        self.threshold_violated = False


        ttc_msg = Float64()
        ttc_msg.data = ttc_min
        self.pub_ttc.publish(ttc_msg)



    def pub_cb(self):
        print(self.threshold_violated)
        if self.threshold_violated:
            msg = AckermannDriveStamped()
            msg.drive.speed = 0.0
            self.pub_brake.publish(msg)

def main():
    try:
        rclpy.init()
        node = SafetyNode()
        rclpy.spin(node)
        

    finally:
        node.destroy_node()
        rclpy.shutdown()    