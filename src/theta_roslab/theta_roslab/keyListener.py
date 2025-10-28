import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from ackermann_msgs.msg import AckermannDriveStamped
class keyListener(Node):
    def __init__(self):
        super().__init__('key_listener')

        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive', 
            10)
        
        self.key_subscriber = self.create_subscription(
            String,
            '/key',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        key = msg.data
        print(key)
        forward = 0.0
        left = 0.0
        #TODO: FILL IN THE CORRECT CONDITIONS
        if key == "UP": 
            forward = 1.0
        elif key == "DOWN":
            forward = -1.0
        elif key == "LEFT": 
            left = 0.5
        elif key == "RIGHT": 
            left = -0.5

        pub_msg = AckermannDriveStamped()
        pub_msg.drive.speed = forward
        pub_msg.drive.steering_angle = left
        self.drive_publisher.publish(pub_msg) #TODO

def main(args=None):
    rclpy.init(args=args)
    node = keyListener() #TODO
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()