import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import threading
import time
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# shared orientation values
roll, pitch, yaw = 0.0, 0.0, 0.0
rpy_lock = threading.Lock()  # <-- lock to protect access

class IMUNode(Node):
    def __init__(self):
        super().__init__('imu_visualizer')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/imu/data_converted',
            self.update_rpy,
            10
        )

    def update_rpy(self, msg_array: Float32MultiArray):
        global roll, pitch, yaw
        with rpy_lock:
            roll = msg_array.data[0]
            pitch = msg_array.data[1]
            yaw = msg_array.data[2]

def imu_thread_func():
    rclpy.init()
    node = IMUNode()
    rclpy.spin(node)
    rclpy.shutdown()

def draw_axes():
    glBegin(GL_LINES)
    # X (Roll) - Red
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(2, 0, 0)
    # Y (Pitch) - Green
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 2, 0)
    # Z (Yaw) - Blue
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 2)
    glEnd()

def draw_cube():
    glBegin(GL_QUADS)
    glColor3f(0.0, 0.7, 0.3)
    # Front
    glVertex3f( 1, 1,-1)
    glVertex3f(-1, 1,-1)
    glVertex3f(-1,-1,-1)
    glVertex3f( 1,-1,-1)
    # Back
    glVertex3f( 1, 1,1)
    glVertex3f(-1, 1,1)
    glVertex3f(-1,-1,1)
    glVertex3f( 1,-1,1)
    # Left
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1,-1)
    glVertex3f(-1,-1,-1)
    glVertex3f(-1,-1, 1)
    # Right
    glVertex3f( 1, 1, 1)
    glVertex3f( 1, 1,-1)
    glVertex3f( 1,-1,-1)
    glVertex3f( 1,-1, 1)
    # Top
    glVertex3f( 1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1,-1)
    glVertex3f( 1, 1,-1)
    # Bottom
    glVertex3f( 1,-1, 1)
    glVertex3f(-1,-1, 1)
    glVertex3f(-1,-1,-1)
    glVertex3f( 1,-1,-1)
    glEnd()

def main():
    # Start IMU thread
    thread = threading.Thread(target=imu_thread_func, daemon=True)
    thread.start()

    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    #glEnable(GL_DEPTH_TEST)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -7)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        with rpy_lock:
            local_roll = roll
            local_pitch = pitch
            local_yaw = yaw

        glPushMatrix()
        print(local_roll, local_pitch, local_yaw)
        glRotatef(local_roll, 1, 0, 0)
        glRotatef(local_pitch, 0, 1, 0)
        glRotatef(local_yaw, 0, 0, 1)
        draw_cube()
        draw_axes()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == '__main__':
    main()
