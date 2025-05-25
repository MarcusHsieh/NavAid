#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from navaid_msgs.msg import LedArrayState

import Jetson.GPIO as GPIO

class LedControllerNode(Node):
    def __init__(self):
        super().__init__('led_controller')

        self.LED_PINS_BCM = [11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 
                             24, 26, 29, 31, 32, 33, 35, 36, 37, 38]

        self.LED_PINS = self.LED_PINS_BCM

        self.setup_gpio()

        self.subscription = self.create_subscription(
            LedArrayState,
            '/led_commands',
            self.led_command_callback,
            10)
        self.get_logger().info("LED Controller Node Started. Listening to /led_commands.")
        self.get_logger().info(f"Using GPIO pins (BOARD mode assumed): {self.LED_PINS}")


    def setup_gpio(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False) 

        for pin in self.LED_PINS:
            try:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            except Exception as e:
                self.get_logger().error(f"Failed to setup GPIO pin {pin}: {e}")
        self.get_logger().info("GPIO pins setup complete.")

    def led_command_callback(self, msg: LedArrayState):
        if len(msg.led_values) != len(self.LED_PINS):
            self.get_logger().warn(f"Received LED command with incorrect length: {len(msg.led_values)}. Expected {len(self.LED_PINS)}.")
            return

        # self.get_logger().info(f"Received LED values: {list(msg.led_values)}")
        for i, led_on_state_value in enumerate(msg.led_values):
            pin_to_control = self.LED_PINS[i]
            try:
                if led_on_state_value > 0:
                    GPIO.output(pin_to_control, GPIO.HIGH)
                else:
                    GPIO.output(pin_to_control, GPIO.LOW)
            except Exception as e:
                 self.get_logger().error(f"Error controlling GPIO pin {pin_to_control}: {e}")


    def destroy_node(self):
        self.get_logger().info("LED Controller cleaning up GPIO.")
        try:
            GPIO.cleanup()
        except Exception as e:
            self.get_logger().error(f"Error during GPIO cleanup: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LedControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("LED Controller shutting down.")
    finally:
        if rclpy.ok():
          node.destroy_node() 
        if rclpy.ok():
          rclpy.shutdown()

if __name__ == '__main__':
    main()