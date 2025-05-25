#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3Stamped 
from std_msgs.msg import Float32MultiArray
import math
import numpy as np
import struct
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as GeoPoint
import tf_transformations

from navaid_msgs.msg import LedArrayState

class NavaidProcessorNode(Node):
    def __init__(self):
        super().__init__('navaid_processor')

        # --- Configuration --- (Same as before)
        self.NUM_LEDS = 20
        self.DEGREES_PER_LED = 360.0 / self.NUM_LEDS
        self.LED_HALF_ANGLE_DEG = self.DEGREES_PER_LED / 2.0
        self.PROXIMITY_ZONES = [
            {'distance': 0.20, 'led_value': 3, 'color_rgb': [1.0, 0.0, 0.0]}, # Red = closest
            {'distance': 0.50, 'led_value': 2, 'color_rgb': [1.0, 0.5, 0.0]}, # Orange = medium
            {'distance': 1.0, 'led_value': 1, 'color_rgb': [1.0, 1.0, 0.0]}, # Yellow = furthest
            {'distance': float('inf'), 'led_value': 0, 'color_rgb': [0.0, 1.0, 0.0]} # Green = clear
        ]
        self.PROXIMITY_ZONES.sort(key=lambda x: x['distance'])
        self.lidar_angular_offset_radians = 0.0
        self.current_device_yaw_radians = 0.0

        # Subscriptions
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_subscription = self.create_subscription(Float32MultiArray, '/imu/data_converted', self.imu_callback, 10)

        # Publishers
        self.led_publisher = self.create_publisher(LedArrayState, '/led_commands', 10)
        self.colored_pc_publisher = self.create_publisher(PointCloud2, '/navaid/colored_scan', 10)
        self.orientation_marker_publisher = self.create_publisher(Marker, '/navaid/device_orientation_marker', 10)


        self.get_logger().info("NavAid Processor Node Started.")

    def imu_callback(self, msg: Float32MultiArray):
        # self.get_logger().info("IMU callback entered!")
        if len(msg.data) >= 3:
            yaw_degrees = msg.data[2]
            self.current_device_yaw_radians = math.radians(yaw_degrees)
            # self.get_logger().info(f"Got yaw_degrees: {yaw_degrees:.2f}")
            # self.get_logger().info(f"Attempting to publish marker for yaw_rad: {self.current_device_yaw_radians:.2f}") 
            self.publish_orientation_marker(self.current_device_yaw_radians)
        else:
            self.get_logger().warn(f"RPY Float32MultiArray too short: {len(msg.data)}")

    def publish_orientation_marker(self, yaw_radians):
        # self.get_logger().info(f"Publishing marker with yaw_rad: {yaw_radians:.2f}")
        marker = Marker()
        marker.header.frame_id = "base_link" # Marker is relative to base_link
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "navaid_orientation"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.1
        
        q = tf_transformations.quaternion_from_euler(0, 0, 0)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale.x = 0.5
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        
        self.orientation_marker_publisher.publish(marker)
        # self.get_logger().info("Marker published.") 

    def normalize_angle_radians(self, angle_rad):
        while angle_rad > math.pi: angle_rad -= 2 * math.pi
        while angle_rad < -math.pi: angle_rad += 2 * math.pi
        return angle_rad

    def pack_rgb(self, r, g, b):
        r_uint8 = int(max(0, min(1, r)) * 255.0)
        g_uint8 = int(max(0, min(1, g)) * 255.0)
        b_uint8 = int(max(0, min(1, b)) * 255.0)
        rgb_int = (r_uint8 << 16) | (g_uint8 << 8) | b_uint8
        return struct.unpack('f', struct.pack('I', rgb_int))[0]

    def scan_callback(self, scan_msg: LaserScan):
        led_output_values = [0] * self.NUM_LEDS
        
        points_for_pc = [] # [x, y, z, rgb_packed_float]

        for i, range_m in enumerate(scan_msg.ranges):
            is_valid_range = (scan_msg.range_min < range_m < scan_msg.range_max and np.isfinite(range_m))
            
            current_led_value = 0
            current_color_rgb = self.PROXIMITY_ZONES[-1]['color_rgb']

            if is_valid_range:
                for zone in self.PROXIMITY_ZONES:
                    if range_m <= zone['distance']:
                        current_led_value = zone['led_value']
                        current_color_rgb = zone['color_rgb']
                        break
            
            # LIDAR to LED mapping logic
            if current_led_value > 0: # IFF triggers an LED
                lidar_point_angle_rad_lidar_frame = scan_msg.angle_min + i * scan_msg.angle_increment
                angle_relative_to_device_forward_rad = self.normalize_angle_radians(
                    lidar_point_angle_rad_lidar_frame + self.lidar_angular_offset_radians
                )
                angle_deg_device_frame = math.degrees(angle_relative_to_device_forward_rad)
                if angle_deg_device_frame < 0: angle_deg_device_frame += 360.0

                assigned_led_idx = -1
                for idx in range(self.NUM_LEDS):
                    if idx <= 9:
                        led_center_deg = (180.0 + idx * self.DEGREES_PER_LED) % 360.0
                    else:
                        led_center_deg = ((idx - 10) * self.DEGREES_PER_LED) % 360.0
                    
                    diff = abs(angle_deg_device_frame - led_center_deg)
                    angular_distance = min(diff, 360.0 - diff)

                    if angular_distance <= self.LED_HALF_ANGLE_DEG:
                        assigned_led_idx = idx
                        break
                
                if assigned_led_idx != -1:
                    if current_led_value > led_output_values[assigned_led_idx]:
                        led_output_values[assigned_led_idx] = current_led_value
            
            #  point -> PointCloud2
            if is_valid_range:
                angle_rad_for_xy = scan_msg.angle_min + i * scan_msg.angle_increment
                x = range_m * math.cos(angle_rad_for_xy)
                y = range_m * math.sin(angle_rad_for_xy)
                z = 0.0
                rgb_packed = self.pack_rgb(current_color_rgb[0], current_color_rgb[1], current_color_rgb[2])
                points_for_pc.append([x, y, z, rgb_packed])

        led_cmd_msg = LedArrayState()
        # led_cmd_msg.led_values = [np.uint8(val) for val in led_output_values]
        led_cmd_msg.led_values = [int(val) for val in led_output_values]
        self.led_publisher.publish(led_cmd_msg)

        if points_for_pc:
            header = Header(stamp=scan_msg.header.stamp, frame_id=scan_msg.header.frame_id)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            pc2_msg = PointCloud2(
                header=header,
                height=1,
                width=len(points_for_pc),
                is_dense=True,
                is_bigendian=False,
                fields=fields,
                point_step=16,
                row_step=16 * len(points_for_pc),
                data=np.array(points_for_pc, dtype=np.float32).tobytes()
            )
            self.colored_pc_publisher.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NavaidProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()