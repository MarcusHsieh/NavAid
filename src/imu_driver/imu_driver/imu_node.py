#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import board
import busio
import adafruit_icm20x

class IMUPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')

        # 1) setup publisher on /imu/data
        self.pub = self.create_publisher(Float32MultiArray, 'imu/data', 10)

        # 2) init I2C + sensor
        i2c = busio.I2C(board.SCL, board.SDA)
        try:
            self.sensor = adafruit_icm20x.ICM20948(i2c)
        except Exception as e:
            self.get_logger().error(f'Failed to init ICM-20948: {e}')
            raise

        self.get_logger().info('ICM-20948 initialized, publishing on /imu/data @10 Hz')
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        # read accel (m/s^2), gyro (rad/s), mag (uT)
        ax, ay, az = self.sensor.acceleration
        gx, gy, gz = self.sensor.gyro
        mx, my, mz = self.sensor.magnetic

        msg = Float32MultiArray()
        msg.data = [ax, ay, az, gx, gy, gz, mx, my, mz]

        


        self.pub.publish(msg)

        self.get_logger().debug(
            f'Published /imu/data: '
            f'acc=[{ax:.2f},{ay:.2f},{az:.2f}]  '
            f'gyro=[{gx:.2f},{gy:.2f},{gz:.2f}]  '
            f'mag=[{mx:.1f},{my:.1f},{mz:.1f}]'
        )

def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
