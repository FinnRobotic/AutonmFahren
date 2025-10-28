
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import curses
stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(1)
stdscr.refresh()

class KeyboardControlTalker(Node):
    
    def __init__(self):

        super().__init__('KeboardControlTalker')

        self.pub_key = self.create_publisher(
            String,
            '/key',
            10
        )
    
    def listenToKeys(self):
        key =''
        while key != ord('q'):
            key = stdscr.getch()
            stdscr.refresh()
            msg = ""
            if key == curses.KEY_UP:
                msg = "UP" 
            elif key == curses.KEY_DOWN:
                msg = "DOWN" 
            elif key == curses.KEY_LEFT:
                msg = "LEFT" 
            elif key == curses.KEY_RIGHT:
                msg = "RIGHT" 
            print(msg)

            richtung = String()
            richtung.data = msg
            self.pub_key.publish(richtung)
        curses.endwin()

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlTalker()
    node.listenToKeys()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()






    
