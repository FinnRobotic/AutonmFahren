from std_msgs.msg import Float64

from sensor_msgs.msg import LaserScan


import numpy as np


import rclpy
from rclpy.node import Node


class LidarExampleNode(Node):


    def __init__(self):
        super().__init__('lidar_example_node')


        self.pub_farthest = self.create_publisher(
            Float64,
            'lidar_example/farthest',
            10
        )

        self.pub_shortest = self.create_publisher(
            Float64,
            'lidar_example/shortest',
            10
        )

        self.sub_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.cb_lidar,
            10
        )


    def cb_lidar(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        shortest = Float64()
        farthest = Float64()

        shortest.data = 10000000000.0
        farthest.data = 0.0


        for range in valid_ranges:
            
            if range > farthest.data:
                farthest.data = float(range)

            if range < shortest.data:
                shortest.data = float(range)

        
        self.pub_farthest.publish(farthest)
        self.pub_shortest.publish(shortest)

        


def main():
    try:
        rclpy.init()
        node = LidarExampleNode()
        rclpy.spin(node)
        

    finally:
        node.destroy_node()
        rclpy.shutdown()    

