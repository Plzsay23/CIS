#!/usr/bin/env python3

import argparse
import time
from enum import Enum

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from std_msgs.msg import String


class State(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    ALIGNED = "ALIGNED"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def zero_twist() -> Twist:
    return Twist()


class EggApproachNode(Node):
    def __init__(self, args):
        super().__init__("egg_approach_node")

        self.args = args

        self.state = State.IDLE
        self.last_egg_x = None
        self.last_egg_y = None
        self.last_egg_time = 0.0
        self.aligned_since = None

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)

        self.create_subscription(
            PointStamped,
            args.egg_topic,
            self.on_egg,
            10,
        )

        self.timer = self.create_timer(1.0 / args.control_hz, self.on_timer)

        self.publish_stop()

        self.get_logger().info(
            "egg_approach_node started: "
            f"egg_topic={args.egg_topic}, cmd_topic={args.cmd_topic}, "
            f"target_x={args.target_x:.3f}, target_y={args.target_y:.3f}"
        )

    def now(self) -> float:
        return time.monotonic()

    def set_state(self, state: State):
        if self.state == state:
            return
        self.state = state
        self.aligned_since = None
        self.get_logger().warn(f"STATE -> {state.value}")
        self.publish_status(f"state={state.value}")

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_stop(self):
        self.cmd_pub.publish(zero_twist())

    def on_egg(self, msg: PointStamped):
        x = float(msg.point.x)
        y = float(msg.point.y)

        if x < self.args.min_x or x > self.args.max_x:
            self.get_logger().warn(f"ignore egg x out of range: x={x:.3f}")
            return

        if abs(y) > self.args.max_abs_y:
            self.get_logger().warn(f"ignore egg y out of range: y={y:.3f}")
            return

        self.last_egg_x = x
        self.last_egg_y = y
        self.last_egg_time = self.now()

        if self.state == State.IDLE:
            self.set_state(State.APPROACH)

    def has_fresh_egg(self) -> bool:
        if self.last_egg_x is None or self.last_egg_y is None:
            return False
        return self.now() - self.last_egg_time <= self.args.detection_timeout

    def is_aligned(self, x: float, y: float) -> bool:
        err_x = x - self.args.target_x
        err_y = y - self.args.target_y
        return abs(err_x) <= self.args.x_tolerance and abs(err_y) <= self.args.y_tolerance

    def make_cmd(self, x: float, y: float) -> Twist:
        err_x = x - self.args.target_x
        err_y = y - self.args.target_y

        vx = self.args.kx * err_x
        vy = self.args.ky * err_y
        wz = self.args.kyaw * err_y

        if self.args.invert_x:
            vx *= -1.0
        if self.args.invert_y:
            vy *= -1.0
        if self.args.invert_yaw:
            wz *= -1.0

        cmd = Twist()
        cmd.linear.x = clamp(vx, -self.args.max_vx, self.args.max_vx)
        cmd.linear.y = clamp(vy, -self.args.max_vy, self.args.max_vy)
        cmd.angular.z = clamp(wz, -self.args.max_wz, self.args.max_wz)
        return cmd

    def on_timer(self):
        if self.state == State.IDLE:
            return

        if not self.has_fresh_egg():
            self.publish_stop()
            self.set_state(State.IDLE)
            self.publish_status("lost_egg_stop")
            return

        x = self.last_egg_x
        y = self.last_egg_y

        if x is None or y is None:
            self.publish_stop()
            self.set_state(State.IDLE)
            return

        if self.is_aligned(x, y):
            self.publish_stop()

            if self.aligned_since is None:
                self.aligned_since = self.now()
                self.get_logger().info(f"inside tolerance: x={x:.3f}, y={y:.3f}")

            stable = self.now() - self.aligned_since
            self.publish_status(
                f"ALIGNING_STABLE x={x:.3f} y={y:.3f} "
                f"stable={stable:.2f}/{self.args.stable_time:.2f}"
            )

            if stable >= self.args.stable_time:
                self.publish_stop()
                self.set_state(State.ALIGNED)
                self.publish_status(f"ALIGNED x={x:.3f} y={y:.3f}")
            return

        if self.state == State.ALIGNED:
            self.publish_stop()
            return

        self.aligned_since = None
        cmd = self.make_cmd(x, y)
        self.cmd_pub.publish(cmd)

        self.publish_status(
            f"APPROACH x={x:.3f} y={y:.3f} "
            f"target_x={self.args.target_x:.3f} target_y={self.args.target_y:.3f} "
            f"cmd_x={cmd.linear.x:.3f} cmd_y={cmd.linear.y:.3f} cmd_wz={cmd.angular.z:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--egg-topic", default="/egg_detection")
    parser.add_argument("--cmd-topic", default="/auto/cmd_vel")
    parser.add_argument("--status-topic", default="/egg_approach/status")

    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--detection-timeout", type=float, default=0.7)

    parser.add_argument("--target-x", type=float, default=0.28)
    parser.add_argument("--target-y", type=float, default=0.0)

    parser.add_argument("--x-tolerance", type=float, default=0.04)
    parser.add_argument("--y-tolerance", type=float, default=0.04)
    parser.add_argument("--stable-time", type=float, default=0.6)

    parser.add_argument("--kx", type=float, default=0.45)
    parser.add_argument("--ky", type=float, default=0.45)
    parser.add_argument("--kyaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=0.055)
    parser.add_argument("--max-vy", type=float, default=0.055)
    parser.add_argument("--max-wz", type=float, default=0.25)

    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")

    parser.add_argument("--min-x", type=float, default=0.10)
    parser.add_argument("--max-x", type=float, default=1.50)
    parser.add_argument("--max-abs-y", type=float, default=0.80)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = EggApproachNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
