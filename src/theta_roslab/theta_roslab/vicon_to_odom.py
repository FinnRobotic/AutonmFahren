import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Joy

from tf2_ros import TransformBroadcaster



class ViconToOdomPub(Node):


    def __init__(self):
        super().__init__('vicon_to_odom_node')


        self.pub_odom = self.create_publisher(
            Odometry,
            '/odom',
            10
        )
        self.odom = Odometry()
        self.odom.header.frame_id = 'odom'
        self.odom.child_frame_id = 'base_link'
        
        sub_twist = self.create_subscription(
            TwistStamped,
            '/Praktikum25_car3/twist',
            self.cb_twist,
            10
        )

        sub_pose = self.create_subscription(
            PoseStamped,
            '/Praktikum25_car3/pose',
            self.cb_pose,
            10
        )


        self.sub_Joy = self.create_subscription(
            Joy,
            '/joy',
            self.cb_joy,
            10
        )
        


        self.tf_broadcaster = TransformBroadcaster(self)


        self.odom_timer = self.create_timer( 0.01, self.cb_timer)
        self.tf_world_base = TransformStamped()
        self.tf_world_base.header.frame_id = 'world'
        self.tf_world_base.child_frame_id = 'base_link'

        
        self.odom_base_link = TransformStamped()
        self.odom_base_link.header.frame_id = 'odom'
        self.odom_base_link.child_frame_id = 'base_link'


        self.tf_world_map = TransformStamped()
        self.tf_world_map.header.frame_id = 'world'
        self.tf_world_map.child_frame_id = 'map'
        self.tf_world_map.header.stamp = self.get_clock().now().to_msg()

        
        self.tf_world_map.transform.rotation.w = 1.0
        self.tf_world_map.transform.translation.x = 0.0
        self.tf_world_map.transform.translation.y = 0.0
        self.tf_world_map.transform.translation.z = 0.0

    def cb_timer(self):
        self.pub_odom.publish(self.odom)
        self.tf_broadcaster.sendTransform(self.tf_world_base)
        self.tf_broadcaster.sendTransform(self.tf_world_map)
        #self.tf_broadcaster.sendTransform(self.odom_base_link)


    def cb_pose(self, msg: PoseStamped):


        self.odom.pose.pose = msg.pose
        self.odom.header.stamp = msg.header.stamp
        self.tf_world_base.header.stamp = msg.header.stamp
        self.tf_world_map.header.stamp = msg.header.stamp
        self.odom_base_link.header.stamp = msg.header.stamp

        self.tf_world_base.transform.rotation = msg.pose.orientation
        self.tf_world_base.transform.translation.x = msg.pose.position.x
        self.tf_world_base.transform.translation.y = msg.pose.position.y
        self.tf_world_base.transform.translation.z = msg.pose.position.z

        self.odom_base_link.transform.rotation = msg.pose.orientation
        self.odom_base_link.transform.translation.x = msg.pose.position.x
        self.odom_base_link.transform.translation.y = msg.pose.position.y
        self.odom_base_link.transform.translation.z = msg.pose.position.z


    def cb_twist(self, msg: TwistStamped):
        self.odom.twist.twist = msg.twist
        self.odom.header.stamp = msg.header.stamp

    def cb_joy(self ,msg: Joy):

        button =msg.buttons[5]
        if button == 1:
            self.tf_world_map.header.stamp = self.get_clock().now().to_msg()

            self.tf_world_map.transform.translation.x = self.tf_world_base.transform.translation.x
            self.tf_world_map.transform.translation.y = self.tf_world_base.transform.translation.y
            self.tf_world_map.transform.translation.z = self.tf_world_base.transform.translation.z

            self.tf_world_map.transform.rotation.x = self.tf_world_base.transform.rotation.x
            self.tf_world_map.transform.rotation.y = self.tf_world_base.transform.rotation.y
            self.tf_world_map.transform.rotation.z = self.tf_world_base.transform.rotation.z
            self.tf_world_map.transform.rotation.w = self.tf_world_base.transform.rotation.w

            t = self.tf_world_map.transform.translation
            r = self.tf_world_map.transform.rotation

            self.get_logger().info(
                f"New Map-Pose:\n"
                f"  Translation:\n"
                f"    x: {t.x:.3f}, y: {t.y:.3f}, z: {t.z:.3f}\n"
                f"  Rotation (quaternion):\n"
                f"    x: {r.x:.3f}, y: {r.y:.3f}, z: {r.z:.3f}, w: {r.w:.3f}"
            )

    





def main(args=None):
    rclpy.init(args=args)
    node = ViconToOdomPub()


    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()