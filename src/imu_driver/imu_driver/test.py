import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np

class IMUConverterNode(Node):
    def __init__(self):
        super().__init__('imu_converter_complementary_rpy')

        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.rpy_deg_publisher = self.create_publisher(Float32MultiArray, '/imu/data_converted', 10)

        self.alpha = 0.98

        self.roll_estimate_rad = 0.0
        self.pitch_estimate_rad = 0.0
        self.yaw_estimate_rad = 0.0

        self.last_time_ns = None

        self.get_logger().info("IMU Converter Node (Complementary RPY) Initialized")

    def imu_callback(self, msg: Imu):
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

        gx_rad_s = np.radians(msg.angular_velocity.x)
        gy_rad_s = np.radians(msg.angular_velocity.y)
        gz_rad_s = np.radians(msg.angular_velocity.z)

        ax_m_s2 = msg.linear_acceleration.x
        ay_m_s2 = msg.linear_acceleration.y
        az_m_s2 = msg.linear_acceleration.z

        roll_acc_rad = np.arctan2(ay_m_s2, az_m_s2)
        pitch_acc_rad = np.arctan2(-ax_m_s2, np.sqrt(ay_m_s2**2 + az_m_s2**2))

        self.roll_estimate_rad = self.alpha * (self.roll_estimate_rad + gx_rad_s * dt) + \
                                 (1.0 - self.alpha) * roll_acc_rad

        self.pitch_estimate_rad = self.alpha * (self.pitch_estimate_rad + gy_rad_s * dt) + \
                                  (1.0 - self.alpha) * pitch_acc_rad

        self.yaw_estimate_rad = self.yaw_estimate_rad + gz_rad_s * dt

        self.yaw_estimate_rad = np.arctan2(np.sin(self.yaw_estimate_rad), np.cos(self.yaw_estimate_rad))

        roll_deg = np.degrees(self.roll_estimate_rad)
        pitch_deg = np.degrees(self.pitch_estimate_rad)
        yaw_deg = np.degrees(self.yaw_estimate_rad)

        rpy_msg = Float32MultiArray()
        rpy_msg.data = [roll_deg, pitch_deg, yaw_deg]
        self.rpy_deg_publisher.publish(rpy_msg)

        self.get_logger().info(f"RPY (deg): Roll={roll_deg:.2f}, Pitch={pitch_deg:.2f}, Yaw={yaw_deg:.2f}")


        self.get_logger().debug(
            f"RPY [deg]: Roll={roll_deg:.2f}, Pitch={pitch_deg:.2f}, Yaw={yaw_deg:.2f} | dt={dt:.4f}s"
        )

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