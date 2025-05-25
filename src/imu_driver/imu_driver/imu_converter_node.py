import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time
import signal




class IMUConverterNode(Node):
    def __init__(self):
        super().__init__('imu_converter_complementary_rpy')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/imu/data',
            self.imu_callback,
            10
        )

        #for calibration
        signal.signal(signal.SIGINT, self.handle_sigquit)

        self.start = time.time()
        self.calibrating = True
        self.slope = 0.0
        self.calibration_duration = 5.0  # secs
        self.yaw_at_5s = None


        # pub roll, pitch, yaw as Float32MultiArray (deg)
        self.rpy_deg_publisher = self.create_publisher(Float32MultiArray, '/imu/data_converted', 10)

        # Complementary filter params
        self.alpha = 0.92

        # Est angles (rads)
        self.roll_estimate_rad = 0.0
        self.pitch_estimate_rad = 0.0
        self.yaw_estimate_rad = 0.0

        self.last_time_ns = None

        self.get_logger().info("IMU Converter Node (Complementary RPY) Initialized")

    def handle_sigquit(self, signum, frame):
        # This for reseting it / recalibration 
        time.sleep(1)
        self.get_logger().info("SIG received — resetting IMU state instead of shutting down.")
        self.roll_estimate_rad = 0.0
        self.pitch_estimate_rad = 0.0
        self.yaw_estimate_rad = 0.0
        self.start = time.time()
        self.calibrating = True
        self.slope = 0.0
        self.last_time_ns = None

    def imu_callback(self, msg: Float32MultiArray):
        current_time_ros = self.get_clock().now()
        current_time_ns = current_time_ros.nanoseconds

        if self.last_time_ns is None:
            self.last_time_ns = current_time_ns
            self.get_logger().info("First IMU message received, initializing timestamp.")
            return

        dt = (current_time_ns - self.last_time_ns) / 1e9
        self.last_time_ns = current_time_ns

        if dt <= 0:
            self.get_logger().warn(f"dt is zero or negative ({dt:.4f}s), skipping update. Check IMU data rate/timestamps.")
            return

        if len(msg.data) < 6:
            self.get_logger().error(f"Received msg.data with insufficient length: {len(msg.data)}. Expected at least 6 elements.")
            return

        ax_m_s2 = msg.data[0]
        ay_m_s2 = msg.data[1]
        az_m_s2 = msg.data[2]

        gx_rad_s = msg.data[3]
        gy_rad_s = msg.data[4]
        gz_rad_s = msg.data[5]

        roll_acc_rad = np.arctan2(ay_m_s2, az_m_s2)

        pitch_acc_rad = np.arctan2(-ax_m_s2, np.sqrt(ay_m_s2**2 + az_m_s2**2))


        self.roll_estimate_rad = self.alpha * (self.roll_estimate_rad + gx_rad_s * dt) + \
                                 (1.0 - self.alpha) * roll_acc_rad
        self.pitch_estimate_rad = self.alpha * (self.pitch_estimate_rad + gy_rad_s * dt) + \
                                  (1.0 - self.alpha) * pitch_acc_rad


        self.yaw_estimate_rad = self.yaw_estimate_rad + gz_rad_s * dt

        self.roll_estimate_rad = np.arctan2(np.sin(self.roll_estimate_rad), np.cos(self.roll_estimate_rad))
        self.pitch_estimate_rad = np.arctan2(np.sin(self.pitch_estimate_rad), np.cos(self.pitch_estimate_rad))
        self.yaw_estimate_rad = np.arctan2(np.sin(self.yaw_estimate_rad), np.cos(self.yaw_estimate_rad))

        roll_deg = float(np.degrees(self.roll_estimate_rad))
        pitch_deg = float(np.degrees(self.pitch_estimate_rad))
        yaw_deg = float(np.degrees(self.yaw_estimate_rad))


        elapsed = time.time() - self.start

        if self.calibrating:
            if elapsed >= self.calibration_duration:
                self.slope = yaw_deg / self.calibration_duration #5s fro now
                self.calibrating = False
                self.get_logger().info(f"Calibration complete. Yaw slope = {np.degrees(self.slope):.4f} deg/s")
        else:
            # drift correction after calibration
            yaw_deg -= float(self.slope * (time.time() - self.start))


        #backup incase the above not working for some reason
        #np(0.5) val below is like 29 degrees
        #yaw_deg -= float(0.507959541322 * (time.time() -self.start))

        rpy_msg = Float32MultiArray()
        rpy_msg.data = [roll_deg, pitch_deg, yaw_deg]
        self.rpy_deg_publisher.publish(rpy_msg)




        #self.get_logger().info(f"RPY (deg): Roll={roll_deg:.2f}, Pitch={pitch_deg:.2f}, Yaw={yaw_deg:.2f} | dt={dt:.4f}s")
        #print(f" {time.time() - self.start}, {yaw_deg}")


def main(args=None):
    rclpy.init(args=args)
    node = IMUConverterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()