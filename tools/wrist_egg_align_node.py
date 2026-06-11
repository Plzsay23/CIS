#!/usr/bin/env python3

import argparse
import json
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from std_msgs.msg import String


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class WristEggAlignNode(Node):
    def __init__(self, args):
        super().__init__("wrist_egg_align_node")
        self.args = args

        self.sub = self.create_subscription(
            PointStamped,
            args.input_topic,
            self.on_pixel,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)

        self.last_msg: Optional[PointStamped] = None
        self.last_time: Optional[float] = None
        self.aligned_since: Optional[float] = None
        self.aligned_latched = False

        self.timer = self.create_timer(1.0 / max(args.rate_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f"wrist egg align started: input={args.input_topic}, cmd={args.cmd_topic}, "
            f"target_x={args.target_x}, target_area={args.target_area}"
        )

    def on_pixel(self, msg: PointStamped):
        self.last_msg = msg
        self.last_time = time.monotonic()

    def publish_status(self, data: dict):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def on_timer(self):
        now = time.monotonic()

        if self.aligned_latched and self.args.latch_aligned:
            self.publish_stop()
            self.publish_status({
                "state": "ALIGNED",
                "latched": True,
                "stamp": time.time(),
            })
            return

        if self.last_msg is None or self.last_time is None:
            self.publish_stop()
            self.publish_status({
                "state": "WAITING_DETECTION",
                "stamp": time.time(),
            })
            return

        if now - self.last_time > self.args.detection_timeout:
            self.publish_stop()
            self.aligned_since = None
            self.publish_status({
                "state": "LOST",
                "age_sec": round(now - self.last_time, 3),
                "stamp": time.time(),
            })
            return

        x = float(self.last_msg.point.x)
        y = float(self.last_msg.point.y)
        area = float(self.last_msg.point.z)

        err_x = x - self.args.target_x
        x_ok = abs(err_x) <= self.args.x_tolerance

        err_y = None
        y_ok = True
        if self.args.target_y is not None:
            err_y = y - self.args.target_y
            y_ok = abs(err_y) <= self.args.y_tolerance

        err_area = None
        area_ok = True
        if self.args.target_area is not None:
            err_area = self.args.target_area - area
            area_ok = abs(err_area) <= self.args.area_tolerance

        all_ok = x_ok and y_ok and area_ok

        if all_ok:
            if self.aligned_since is None:
                self.aligned_since = now

            self.publish_stop()

            stable = now - self.aligned_since
            state = "STABLE_WAIT"
            if stable >= self.args.stable_time:
                state = "ALIGNED"
                self.aligned_latched = True

            self.publish_status({
                "state": state,
                "x": round(x, 5),
                "y": round(y, 5),
                "area": round(area, 6),
                "err_x": round(err_x, 5),
                "err_y": None if err_y is None else round(err_y, 5),
                "err_area": None if err_area is None else round(err_area, 6),
                "stable_sec": round(stable, 3),
                "stamp": time.time(),
            })
            return

        self.aligned_since = None

        cmd = Twist()

        # image x_norm: 오른쪽이 +
        # ROS 관례상 linear.y는 왼쪽이 +인 경우가 많으므로 기본 strafe_sign=-1 추천.
        if not x_ok:
            cmd.linear.y = clamp(
                self.args.strafe_sign * self.args.kp_x * err_x,
                -self.args.max_strafe,
                self.args.max_strafe,
            )

        # area가 목표보다 작으면 멀다는 뜻. forward_sign=+1이면 전진.
        if self.args.target_area is not None and not area_ok:
            cmd.linear.x = clamp(
                self.args.forward_sign * self.args.kp_area * err_area,
                -self.args.max_forward,
                self.args.max_forward,
            )

        # 너무 작은 속도는 바퀴가 안 움직일 수 있어서 최소 속도 부여
        if abs(cmd.linear.x) > 1e-6 and abs(cmd.linear.x) < self.args.min_forward:
            cmd.linear.x = self.args.min_forward if cmd.linear.x > 0 else -self.args.min_forward

        if abs(cmd.linear.y) > 1e-6 and abs(cmd.linear.y) < self.args.min_strafe:
            cmd.linear.y = self.args.min_strafe if cmd.linear.y > 0 else -self.args.min_strafe

        self.cmd_pub.publish(cmd)

        self.publish_status({
            "state": "ALIGNING",
            "x": round(x, 5),
            "y": round(y, 5),
            "area": round(area, 6),
            "err_x": round(err_x, 5),
            "err_y": None if err_y is None else round(err_y, 5),
            "err_area": None if err_area is None else round(err_area, 6),
            "cmd_x": round(float(cmd.linear.x), 5),
            "cmd_y": round(float(cmd.linear.y), 5),
            "x_ok": x_ok,
            "y_ok": y_ok,
            "area_ok": area_ok,
            "stamp": time.time(),
        })


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-topic", default="/egg/wrist_pixel")
    parser.add_argument("--cmd-topic", default="/auto/cmd_vel")
    parser.add_argument("--status-topic", default="/egg/wrist_align_status")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--detection-timeout", type=float, default=0.4)

    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-area", type=float, default=None)

    parser.add_argument("--x-tolerance", type=float, default=0.08)
    parser.add_argument("--y-tolerance", type=float, default=0.10)
    parser.add_argument("--area-tolerance", type=float, default=0.015)

    parser.add_argument("--stable-time", type=float, default=0.6)
    parser.add_argument("--latch-aligned", action="store_true", default=True)

    parser.add_argument("--kp-x", type=float, default=0.05)
    parser.add_argument("--kp-area", type=float, default=0.80)

    parser.add_argument("--max-forward", type=float, default=0.035)
    parser.add_argument("--max-strafe", type=float, default=0.035)
    parser.add_argument("--min-forward", type=float, default=0.012)
    parser.add_argument("--min-strafe", type=float, default=0.012)

    parser.add_argument("--forward-sign", type=float, default=1.0)
    parser.add_argument("--strafe-sign", type=float, default=-1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = WristEggAlignNode(args)

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
