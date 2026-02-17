# oakd_demo/oakd_subscriber.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class OakSubscriber(Node):
    def __init__(self):
        super().__init__('oakd_subscriber')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/oakd/rgb', self.rgb_cb, 10)
        self.create_subscription(Image, '/oakd/depth', self.depth_cb, 10)

    def rgb_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("RGB", img)
        cv2.waitKey(1)

    def depth_cb(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, "mono8")
        cv2.imshow("Depth", depth)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = OakSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
