import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String




#WALL FOLLOW PARAMS
ANGLE_RANGE = 270 # Hokuyo 10LX has 270 degrees scan
DESIRED_DISTANCE_RIGHT = 1.5 # meters
DESIRED_DISTANCE_LEFT = 0.55 # tune parameters if necessary
MIN_VELOCITY = 0.0
MAX_VELOCITY = 3.0
MIN_DISTANCE = 0.1
MAX_DISTANCE = 30.0
LOOKAHEADDIST = 0.30
THETA = 50

I_MAX = 0.5  
I_MIN = -0.5


class PubSubNode(Node):


    def __init__(self):
        super().__init__('pub_sub')


        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        




        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )



        #PID CONTROL PARAMS
        self.kp = 0.15
        self.kd = 0.1
        self.ki = 0.075
        self.servo_offset = 0.0
        self.prev_error = 0.0
        self.error = 0.0
        self.integral = 0.0
        self.last_time = None

        self.desired_dist = DESIRED_DISTANCE_RIGHT
        self.speed = 0.0

        



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
        if key == "LEFT": 
            self.desired_dist += 0.1
            if self.desired_dist > MAX_DISTANCE:
                self.desired_dist = MAX_DISTANCE
        elif key == "RIGHT":
            self.desired_dist -= 0.1
            if self.desired_dist < MIN_DISTANCE:
                self.desired_dist = MIN_DISTANCE
        elif key == "UP": 
            self.speed += 0.1
            if self.speed > MAX_VELOCITY:
                self.speed = MAX_VELOCITY
        elif key == "DOWN": 
            self.speed -= 0.1
            if self.speed < MIN_VELOCITY:
                self.speed = MIN_VELOCITY

        print(f'Speed: {self.speed}')
        print(f'Range: {self.desired_dist}')
        



    def getRange(self, scan: LaserScan, angle):
        # data: single message from topic /scan
        # angle: between -135 to 135 degrees, where 0 degrees is directly to the front
        # Outputs length in meters to object with angle in lidar scan field of view
        
        angle_incr = scan.angle_increment
        idx = int(np.radians(angle + 135) / angle_incr) - 1

        if np.isfinite(scan.ranges[idx]):
            return scan.ranges[idx]
        
        elif np.isfinite(scan.ranges[idx+1]):
            return scan.ranges[idx+1]
        elif np.isfinite(scan.ranges[idx-1]):
            return scan.ranges[idx-1]
        
        else:
            return -1

    

    def pid_control(self, error, velocity):
 
        angle = 0.0
        drive_msg = AckermannDriveStamped()

        P = error * self.kp

        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9  
        dt = t - self.last_time if self.last_time is not None else 0.0
        self.last_time = t
        
        D = 0
        if dt != 0:
            D = self.kd * (error - self.prev_error) / dt


        if dt > 0.0:
            self.integral += error * dt

            
            self.integral = max(min(self.integral, I_MAX), I_MIN)

        I = self.ki * self.integral

        angle = P + I + D + self.servo_offset

        drive_msg.drive.steering_angle = -angle
        drive_msg.drive.speed = velocity
        self.drive_publisher.publish(drive_msg)


    def followRight(self, scan: LaserScan):

        b = self.getRange(scan, -90)
        a = self.getRange(scan, -90 + THETA)


        alpha = np.arctan((a * np.cos(np.radians(THETA)) - b)/ (a * np.sin(np.radians(THETA))))

        D_t = b * np.cos(alpha) + LOOKAHEADDIST * np.sin(alpha)

        return D_t - self.desired_dist
    

    def lidar_callback(self, scan_msg: LaserScan):

        
        error = self.followRight(scan_msg)

        #send error to pid_control
        self.pid_control(error, self.speed)



def main(args=None):
    rclpy.init(args=args)
    node = PubSubNode()


    rclpy.spin(node)
    PubSubNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()