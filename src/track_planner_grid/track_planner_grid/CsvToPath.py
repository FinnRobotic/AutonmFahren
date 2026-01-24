import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv


CSV_PATH = "trajectory.csv"


class CsvToPath(Node):

    def __init__(self):
        super().__init__("csv_to_path")

        self.publisher = self.create_publisher(Path, "/trajectory_path", 10)
        self.timer = self.create_timer(1.0, self.publish_path)

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.load_csv()
        self.get_logger().info("CSV Path publisher started")


    def load_csv(self):
        with open(CSV_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.pose.position.x = float(row["x"])
                pose.pose.position.y = float(row["y"])
                pose.pose.orientation.w = 1.0
                self.path_msg.poses.append(pose)


    def publish_path(self):
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.path_msg)


def main():
    rclpy.init()
    node = CsvToPath()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
