from std_msgs.msg import Float64

from sensor_msgs.msg import LaserScan
from theta_msgs.msg import ScanRange


import numpy as np


import rclpy
from rclpy.node import Node


class LidarExampleNode(Node):


    def __init__(self):
        super().__init__('lidar_example_node')


        self.pub_own = self.create_publisher(
            ScanRange,
            '/ScanRange',
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

        scanRange = ScanRange()

        scanRange.shortest = 10000000000.0
        scanRange.farthest = 0.0


        for range in valid_ranges:
            
            if range > scanRange.farthest:
                scanRange.farthest = float(range)

            if range < scanRange.shortest:
                scanRange.shortest = float(range)

        
        self.pub_own.publish(scanRange)

        


def main():
    try:
        rclpy.init()
        node = LidarExampleNode()
        rclpy.spin(node)
        

    finally:
        node.destroy_node()
        rclpy.shutdown()    

