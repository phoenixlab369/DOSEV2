# oakd_demo/oakd_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai
import cv2

class OakPublisher(Node):
    def __init__(self):
        super().__init__('oakd_publisher')
        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, '/oakd/rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/oakd/depth', 10)
        
        # Create device and pipeline (v3 API)
        self.device = dai.Device()
        self.pipeline = dai.Pipeline(self.device)
        
        # Get available cameras
        cameras = self.device.getConnectedCameras()
        self.get_logger().info(f'Connected cameras: {cameras}')
        
        # Create RGB camera (CAM_A is typically RGB)
        if dai.CameraBoardSocket.CAM_A in cameras:
            self.cam_rgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            self.q_rgb = self.cam_rgb.requestOutput((1920, 1080)).createOutputQueue()
        else:
            self.get_logger().warn('RGB camera not found')
            self.q_rgb = None
        
        # Create stereo depth with auto camera creation
        if dai.CameraBoardSocket.CAM_B in cameras and dai.CameraBoardSocket.CAM_C in cameras:
            self.stereo = self.pipeline.create(dai.node.StereoDepth).build(
                autoCreateCameras=True,
                size=(640, 400)
            )
            self.q_depth = self.stereo.depth.createOutputQueue()  # No parentheses!
        else:
            self.get_logger().warn('Stereo cameras not found')
            self.q_depth = None
        
        # Start pipeline
        self.pipeline.start()
        
        self.create_timer(0.03, self.publish_frames)  # ~30 FPS
        self.get_logger().info('OAK-D Publisher started')

    def publish_frames(self):
        try:
            # Get and publish RGB frame
            if self.q_rgb is not None:
                rgb_frame = self.q_rgb.get()
                if rgb_frame is not None:
                    rgb_cv = rgb_frame.getCvFrame()
                    self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb_cv, "bgr8"))
            
            # Get and publish depth frame
            if self.q_depth is not None:
                depth_frame = self.q_depth.get()
                if depth_frame is not None:
                    depth_cv = depth_frame.getCvFrame()
                    # Normalize depth for visualization
                    depth_norm = cv2.normalize(depth_cv, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                    self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_norm, "mono8"))
                    
        except Exception as e:
            self.get_logger().error(f'Error publishing frames: {str(e)}')

    def destroy_node(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        super().destroy_node()

def main():
    rclpy.init()
    node = OakPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()