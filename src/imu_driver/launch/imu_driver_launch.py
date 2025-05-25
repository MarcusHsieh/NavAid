from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='imu_driver',
            executable='imu_node',
            name='imu_node'
        ),
        Node(
            package='imu_driver',
            executable='imu_converter',
            name='imu_converter'
        ),
        Node(
            package='imu_driver',
            executable='imu_subscriber',
            name='imu_subscriber'
        ),
    ])