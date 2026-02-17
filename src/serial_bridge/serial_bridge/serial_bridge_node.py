#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
import threading

class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__('serial_bridge_node')

        # ================= SERIAL CONFIG =================
        self.serial_port = '/dev/ttyTHS1'
        self.baudrate = 115200
        self.timeout = 0.1

        try:
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)
            self.get_logger().info(f"Connected to ESP32 on {self.serial_port}")
        except serial.SerialException as e:
            self.get_logger().error(f"Serial connection failed: {e}")
            raise e

        # ================= ROS SUB =================
        self.subscription = self.create_subscription(
            String,
            '/robot_cmd',
            self.cmd_callback,
            10
        )

        # ================= WATCHDOG =================
        self.last_cmd_time = time.time()
        self.cmd_timeout = 1.0
        self.last_cmd_sent = None
        #self.create_timer(0.1, self.watchdog)  # check every 100ms

        # ================= READ THREAD =================
        self.read_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
        self.read_thread.start()

        self.get_logger().info("Serial Bridge Node started for ESP32 full motor control")

    # ====================================================
    def cmd_callback(self, msg: String):
        """
        Expects full ESP32 commands: LF 100, RF 100, STOP, SPEED, PWMF 4000, etc.
        """
        cmd = msg.data.strip()
        if not cmd:
            return

        try:
            self.ser.write(f"{cmd}\n".encode())
            self.last_cmd_time = time.time()
            self.last_cmd_sent = cmd
            self.get_logger().info(f"Sent → ESP32: {cmd}")
        except serial.SerialException as e:
            self.get_logger().error(f"Serial write failed: {e}")

    # ====================================================
    def watchdog(self):
        # Send STOP only once after timeout
        if time.time() - self.last_cmd_time > self.cmd_timeout:
            if self.last_cmd_sent != 'STOP':
                try:
                    self.ser.write(b"STOP\n")
                    self.last_cmd_sent = 'STOP'
                    self.get_logger().info("Watchdog sent STOP")
                except serial.SerialException:
                    pass

    # ====================================================
    def read_serial_loop(self):
        """
        Continuously read from ESP32 and print responses.
        Useful for SPEED, STATUS, or error messages.
        """
        while True:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if line:
                        self.get_logger().info(f"ESP32 → {line}")
                else:
                    time.sleep(0.01)
            except serial.SerialException:
                pass

# ========================================================
def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Stopping robot and closing serial")
        try:
            node.ser.write(b"STOP\n")
            node.ser.close()
        except:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
