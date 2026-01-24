import rclpy
from rclpy.node import Node

import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped


class TfToOdometry(Node):
    def __init__(self):
        super().__init__('tf_to_odometry')

        # Parameters
        self.declare_parameter("parent_frame", "map")
        self.declare_parameter("child_frame", "zed_camera_link")
        self.declare_parameter("odom_topic", "/tf_odom")
        self.declare_parameter("rate", 50.0)

        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.rate = self.get_parameter("rate").value

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher
        self.pub_odom = self.create_publisher(Odometry, self.odom_topic, 10)

        # Timer
        self.timer = self.create_timer(1.0 / self.rate, self.timer_cb)

        self.get_logger().info(
            f"TF → Odom: {self.parent_frame} -> {self.child_frame} @ {self.rate} Hz"
        )

    # -------------------------------------------------

    def timer_cb(self):
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                rclpy.time.Time()
            )
        except Exception:
            return

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.parent_frame
        odom.child_frame_id = self.child_frame

        odom.pose.pose.position.x = tf.transform.translation.x
        odom.pose.pose.position.y = tf.transform.translation.y
        odom.pose.pose.position.z = tf.transform.translation.z

        odom.pose.pose.orientation = tf.transform.rotation

        self.pub_odom.publish(odom)


def main():
    rclpy.init()
    node = TfToOdometry()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
