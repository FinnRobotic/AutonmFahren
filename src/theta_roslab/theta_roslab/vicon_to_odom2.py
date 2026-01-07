#!/usr/bin/env python3
# vicon_to_odom.py + TF alignment integration
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
import tf2_ros


class ViconToOdom(Node):
    def __init__(self):
        super().__init__('vicon_to_odom')

        # ---- VICON INPUT ----
        self.create_subscription(PoseStamped, '/Praktikum25_car3/pose', self.pose_cb, 10)
        self.create_subscription(TwistStamped, '/Praktikum25_car3/twist', self.twist_cb, 10)

        # ---- BUTTON FOR MAP RESET ----
        # Passe das Topic an dein Fahrzeug an!
        self.create_subscription(Joy, '/joy', self.joy_cb, 10)

        # Publisher für /odom (Cartographer braucht /odom)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Zwischenspeicher Twist
        self.last_twist = None

        # ---- TF BROADCASTER ----
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Map-Offset (world → map)
        self.map_offset = None
        self.reset_requested = False

        self.get_logger().info("vicon_to_odom with TF alignment gestartet.")

    
    def joy_cb(self, msg: Joy):
        # Beispiel: Button 3 (Index 2) löst Reset aus
        if len(msg.buttons) > 2 and msg.buttons[2] == 1:
            self.reset_requested = True
            self.get_logger().info("Map Reset Button gedrückt!")

    # -----------------------------------------------------------
    # Pose Callback von Vicon
    # -----------------------------------------------------------
    def pose_cb(self, msg: PoseStamped):

        
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose = msg.pose

        if self.last_twist is not None:
            odom.twist.twist = self.last_twist.twist

        self.odom_pub.publish(odom)

        # --------------------------------------------------------
        # 2) TF: odom → base_link (für RVIZ)
        # --------------------------------------------------------
        tf_odom = TransformStamped()
        tf_odom.header.stamp = self.get_clock().now().to_msg()
        tf_odom.header.frame_id = "odom"
        tf_odom.child_frame_id = "base_link"

        tf_odom.transform.translation.x = msg.pose.position.x
        tf_odom.transform.translation.y = msg.pose.position.y
        tf_odom.transform.translation.z = msg.pose.position.z

        tf_odom.transform.rotation = msg.pose.orientation

        self.tf_broadcaster.sendTransform(tf_odom)

        # --------------------------------------------------------
        # 3) TF: world → base_link  (Vicon-Pose)
        # --------------------------------------------------------
        tf_world_bl = TransformStamped()
        tf_world_bl.header.stamp = self.get_clock().now().to_msg()
        tf_world_bl.header.frame_id = "world"
        tf_world_bl.child_frame_id = "base_link"

        tf_world_bl.transform.translation.x = msg.pose.position.x
        tf_world_bl.transform.translation.y = msg.pose.position.y
        tf_world_bl.transform.translation.z = msg.pose.position.z
        tf_world_bl.transform.rotation = msg.pose.orientation

        self.tf_broadcaster.sendTransform(tf_world_bl)

        # --------------------------------------------------------
        # 4) MAP RESET
        # --------------------------------------------------------
        if self.reset_requested or self.map_offset is None:
            # Neuer Offset = aktuelle Vicon-Pose
            self.map_offset = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.orientation
            )
            self.reset_requested = False

            self.get_logger().info(
                f"Map Offset gesetzt: x={self.map_offset[0]:.3f}, "
                f"y={self.map_offset[1]:.3f}"
            )

        # --------------------------------------------------------
        # 5) TF: world → map (Offset)
        # --------------------------------------------------------
        tf_world_map = TransformStamped()
        tf_world_map.header.stamp = self.get_clock().now().to_msg()
        tf_world_map.header.frame_id = "world"
        tf_world_map.child_frame_id = "map"

        # world → map ist NEGATIV der Fahrzeugpose zum Reset-Zeitpunkt
        tf_world_map.transform.translation.x = self.map_offset[0]
        tf_world_map.transform.translation.y = self.map_offset[1]
        tf_world_map.transform.translation.z = 0.0
        tf_world_map.transform.rotation = self.map_offset[2]

        self.tf_broadcaster.sendTransform(tf_world_map)

    # -----------------------------------------------------------
    # Twist Callback
    # -----------------------------------------------------------
    def twist_cb(self, msg: TwistStamped):
        self.last_twist = msg


def main(args=None):
    rclpy.init(args=args)
    node = ViconToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
