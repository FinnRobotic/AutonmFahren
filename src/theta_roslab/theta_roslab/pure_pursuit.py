import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

from tf_transformations import euler_from_quaternion

import os
import csv
from ament_index_python.packages import get_package_share_directory

import numpy as np
import math
from scipy.interpolate import splprep, splev

LOOKAHEADDIST = 0.6
VEHICLE_SPEED = 1.5 # in m/s
WHEELBASE = 0.325

package_share_directory = get_package_share_directory('theta_roslab')
filename = os.path.join(package_share_directory, 'waypoints', 'levine-waypoints.csv')

# Import waypoints.csv into a list (path_points)
with open(filename, 'r') as f:
    path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(p[0]) - 1.5 , float(p[1]) - 1.5 , float(p[2])) for p in path_points]


def interpolate_path(points, resolution=0.05):
    """
    Interpoliert den gegebenen Pfad (x,y,z) mit Spline-Interpolation.
    resolution = gewünschter Abstand zwischen Punkten in Metern.
    """
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]

    # Spline erstellen
    tck, u = splprep([x, y, z], s=0)
    # Länge des Pfads schätzen (für gleichmäßige Punkte)
    num_points = int(np.ceil(sum(np.hypot(np.diff(x), np.diff(y))) / resolution))
    print(num_points)
    # Neue gleichmäßig verteilte u-Werte erzeugen
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new, z_new = splev(u_new, tck)

    return list(zip(x_new, y_new, z_new))


class PurePursuitNode(Node):


    def __init__(self):
        super().__init__('pure_pursuit_node')


        self.marker_publisher = self.create_publisher(MarkerArray, 'csv_point', 10)
        self.goal_marker_publisher = self.create_publisher(Marker, 'current_goal_point', 10)


        self.sub_pose = self.create_subscription(
            PoseStamped,
            '/Praktikum25_car3/pose',
            self.cb_pose,
            10
        )


        self.pub_drive = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )


        #todo: Import waypoints.csv into a list (path_points)
        
        self.path_points = path_points


        self.marker_msg = self.publish_points()



    # This method helps you to visualize the points in RViz. It returns a message which can be published.
    # The points need to be in the list self.path_points in order for it to work.
    def publish_points(self):
        marker_array = MarkerArray()
        marker_id = 0
        for p in self.path_points:
            marker = Marker()
            marker.header.frame_id = "/world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = 2
            marker.id = marker_id
            marker_id +=1

            marker.scale.x =0.1
            marker.scale.y =0.1
            marker.scale.z =0.1

            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.position.z = p[2]*0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)
            
        return marker_array
    
    # If you call this method with a x, y and z coordinate, it will publish a message to show a single point in RViz
    # This can be useful if you want to check which point the car is heading at currently.
    def publish_single_point(self, x, y, z):
        global nearest

        marker = Marker()
        marker.header.frame_id = "/world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = 2
        marker.id = 9999

        marker.scale.x =0.2
        marker.scale.y =0.2
        marker.scale.z =0.2

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        self.goal_marker_publisher.publish(marker)

    def cb_pose(self, msg: PoseStamped):

        x = msg.pose.position.x
        y = msg.pose.position.y

        w = msg.pose.orientation.w
        z = msg.pose.orientation.z
        theta = math.atan2(2 * (w * z), 1 - 2 * (z ** 2))



        xy = np.array([-x, -y, 1])
        # todo: find the current waypoint to track using methods mentioned in the paper.
        T = np.array(
            [
                [np.cos(-theta), -np.sin(-theta),0],
                [np.sin(-theta),np.cos(-theta),0],
                [0,0,1]
            ]
        )
        xy = T @ xy

        T[0,2] = xy[0]
        T[1,2] = xy[1]
        shortest_p =np.array( [100,100,1])
        for point in self.path_points:
            p = np.array([point[0],point[1],1])
            p = T@p
            dist = math.sqrt(p[0]**2+p[1]**2)
            
            if p[0]>=0 and dist>=LOOKAHEADDIST :
                if math.sqrt(shortest_p[0]**2+shortest_p[1]**2) >= dist:
                    shortest_p=p

        shortest_p[0] = shortest_p[0] - 0.37
        alpha = math.atan2(shortest_p[1] , shortest_p[0])
        dist = math.sqrt(shortest_p[0]**2+shortest_p[1]**2)
        delta = np.arctan(2 * WHEELBASE * math.sin(alpha) / dist)

        drive = AckermannDriveStamped()

        drive.drive.steering_angle = delta
        drive.drive.speed = VEHICLE_SPEED
        drive.header.stamp = self.get_clock().now().to_msg()
        self.pub_drive.publish(drive)

        # marker message to visualize waypoints
        p = np.linalg.inv(T) @ shortest_p
        self.publish_single_point(p[0], p[1], 0.0)
        self.marker_publisher.publish(self.marker_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    PurePursuitNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
