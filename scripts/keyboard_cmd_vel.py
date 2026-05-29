#!/usr/bin/env python3

import sys
import time
import tty
import termios
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP = """
ROS2 LeKiwi Keyboard Teleop -> /cmd_vel

Keys:
  w : forward
  s : backward
  a : left strafe
  d : right strafe
  z : rotate left
  x : rotate right

  r : speed up
  f : speed down
  space : stop
  q : quit

Safety:
  Command is published continuously.
  If no key is pressed for timeout duration, velocity becomes zero.
"""


class KeyboardCmdVel(Node):
    def __init__(self):
        super().__init__("keyboard_cmd_vel")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("key_timeout_s", 0.35)

        self.declare_parameter("xy_slow", 0.05)
        self.declare_parameter("xy_medium", 0.10)
        self.declare_parameter("xy_fast", 0.20)

        self.declare_parameter("theta_slow", 0.25)
        self.declare_parameter("theta_medium", 0.50)
        self.declare_parameter("theta_fast", 0.90)

        self.topic = self.get_parameter("cmd_vel_topic").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.key_timeout_s = float(self.get_parameter("key_timeout_s").value)

        self.xy_levels = [
            float(self.get_parameter("xy_slow").value),
            float(self.get_parameter("xy_medium").value),
            float(self.get_parameter("xy_fast").value),
        ]

        self.theta_levels = [
            float(self.get_parameter("theta_slow").value),
            float(self.get_parameter("theta_medium").value),
            float(self.get_parameter("theta_fast").value),
        ]

        self.speed_index = 0

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.last_key_time = 0.0
        self.running = True

        self.pub = self.create_publisher(Twist, self.topic, 10)

        period = 1.0 / self.publish_hz
        self.timer = self.create_timer(period, self.loop)

        self.old_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        print(HELP)
        self.print_speed()

    def restore_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_termios)

    def get_key_nonblocking(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.read(1)
        return None

    def current_xy(self):
        return self.xy_levels[self.speed_index]

    def current_theta(self):
        return self.theta_levels[self.speed_index]

    def print_speed(self):
        print(
            f"[speed level {self.speed_index + 1}/3] "
            f"xy={self.current_xy():.2f} m/s, "
            f"theta={self.current_theta():.2f} rad/s"
        )

    def set_cmd_from_key(self, key):
        xy = self.current_xy()
        th = self.current_theta()

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        if key == "w":
            self.vx = xy
        elif key == "s":
            self.vx = -xy
        elif key == "a":
            self.vy = xy
        elif key == "d":
            self.vy = -xy
        elif key == "z":
            self.wz = th
        elif key == "x":
            self.wz = -th
        elif key == " ":
            self.stop()
        elif key == "r":
            self.speed_index = min(self.speed_index + 1, len(self.xy_levels) - 1)
            self.print_speed()
            self.stop()
        elif key == "f":
            self.speed_index = max(self.speed_index - 1, 0)
            self.print_speed()
            self.stop()
        elif key == "q":
            self.stop()
            self.running = False
        else:
            return

        self.last_key_time = time.monotonic()

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = float(self.vx)
        msg.linear.y = float(self.vy)
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(self.wz)

        self.pub.publish(msg)

    def loop(self):
        key = self.get_key_nonblocking()
        if key is not None:
            self.set_cmd_from_key(key)

        now = time.monotonic()
        if now - self.last_key_time > self.key_timeout_s:
            self.stop()

        self.publish_cmd()

        if not self.running:
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = KeyboardCmdVel()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        for _ in range(5):
            node.publish_cmd()
            time.sleep(0.02)

        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()
        print("\nkeyboard_cmd_vel stopped.")


if __name__ == "__main__":
    main()
