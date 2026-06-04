#!/usr/bin/env python3

import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


HELP = """
Keyboard LeKiwi drive node

keys:
  w : forward
  s : backward
  a : left strafe
  d : right strafe
  z : rotate left
  x : rotate right
  space : stop
  r : speed up
  f : speed down
  q : quit

This publishes geometry_msgs/Twist to /dashboard/cmd_vel by default.
"""


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_twist(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> Twist:
    msg = Twist()
    msg.linear.x = float(x)
    msg.linear.y = float(y)
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = float(yaw)
    return msg


class KeyboardDriveNode(Node):
    def __init__(self, args):
        super().__init__("keyboard_drive_node")

        self.topic = args.topic
        self.max_x = float(args.max_x)
        self.max_y = float(args.max_y)
        self.max_yaw = float(args.max_yaw)
        self.step_x = float(args.step_x)
        self.step_y = float(args.step_y)
        self.step_yaw = float(args.step_yaw)
        self.hold_sec = float(args.hold_sec)
        self.publish_hz = float(args.publish_hz)

        self.scale = 1.0
        self.last_cmd = make_twist()
        self.last_key_time = 0.0

        self.pub = self.create_publisher(Twist, self.topic, 10)

        self.get_logger().info(
            f"keyboard_drive_node started: topic={self.topic}, "
            f"max_x={self.max_x}, max_y={self.max_y}, max_yaw={self.max_yaw}"
        )

    def set_cmd(self, x: float, y: float, yaw: float):
        x = clamp(x * self.scale, -self.max_x, self.max_x)
        y = clamp(y * self.scale, -self.max_y, self.max_y)
        yaw = clamp(yaw * self.scale, -self.max_yaw, self.max_yaw)

        self.last_cmd = make_twist(x, y, yaw)
        self.last_key_time = time.monotonic()
        self.pub.publish(self.last_cmd)

    def stop(self):
        self.last_cmd = make_twist()
        self.last_key_time = 0.0
        self.pub.publish(self.last_cmd)

    def publish_current_or_stop(self):
        now = time.monotonic()

        if self.last_key_time <= 0.0:
            self.pub.publish(make_twist())
            return

        if now - self.last_key_time > self.hold_sec:
            self.stop()
            return

        self.pub.publish(self.last_cmd)

    def handle_key(self, key: str) -> bool:
        if key == "q":
            self.stop()
            return False

        if key == " ":
            self.stop()
            self.get_logger().info("stop")
            return True

        if key == "r":
            self.scale = clamp(self.scale + 0.2, 0.2, 2.0)
            self.get_logger().info(f"speed scale={self.scale:.1f}")
            return True

        if key == "f":
            self.scale = clamp(self.scale - 0.2, 0.2, 2.0)
            self.get_logger().info(f"speed scale={self.scale:.1f}")
            return True

        if key == "w":
            self.set_cmd(self.step_x, 0.0, 0.0)
        elif key == "s":
            self.set_cmd(-self.step_x, 0.0, 0.0)
        elif key == "a":
            self.set_cmd(0.0, self.step_y, 0.0)
        elif key == "d":
            self.set_cmd(0.0, -self.step_y, 0.0)
        elif key == "z":
            self.set_cmd(0.0, 0.0, self.step_yaw)
        elif key == "x":
            self.set_cmd(0.0, 0.0, -self.step_yaw)

        return True


def read_key_nonblocking(timeout: float = 0.02) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return None
    return sys.stdin.read(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/dashboard/cmd_vel")

    parser.add_argument("--max-x", type=float, default=0.10)
    parser.add_argument("--max-y", type=float, default=0.10)
    parser.add_argument("--max-yaw", type=float, default=0.5236)

    parser.add_argument("--step-x", type=float, default=0.06)
    parser.add_argument("--step-y", type=float, default=0.06)
    parser.add_argument("--step-yaw", type=float, default=0.25)

    parser.add_argument("--hold-sec", type=float, default=0.25)
    parser.add_argument("--publish-hz", type=float, default=20.0)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = KeyboardDriveNode(args)

    old_term = termios.tcgetattr(sys.stdin)

    print(HELP)

    try:
        tty.setcbreak(sys.stdin.fileno())

        period = 1.0 / max(args.publish_hz, 1.0)
        next_pub = time.monotonic()

        running = True
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.0)

            key = read_key_nonblocking(timeout=0.01)
            if key is not None:
                running = node.handle_key(key)

            now = time.monotonic()
            if now >= next_pub:
                node.publish_current_or_stop()
                next_pub = now + period

            time.sleep(0.002)

    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        time.sleep(0.05)
        node.stop()

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
