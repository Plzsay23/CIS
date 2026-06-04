#!/usr/bin/env python3

import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


@dataclass
class CmdSource:
    name: str
    timeout_sec: float
    last_msg: Optional[Twist] = None
    last_time: float = 0.0


def zero_twist() -> Twist:
    return Twist()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_twist(msg: Twist, max_x: float, max_y: float, max_yaw: float) -> Twist:
    out = Twist()
    out.linear.x = clamp(float(msg.linear.x), -max_x, max_x)
    out.linear.y = clamp(float(msg.linear.y), -max_y, max_y)
    out.linear.z = 0.0
    out.angular.x = 0.0
    out.angular.y = 0.0
    out.angular.z = clamp(float(msg.angular.z), -max_yaw, max_yaw)
    return out


class CmdVelMuxNode(Node):
    def __init__(self):
        super().__init__("cmd_vel_mux_node")

        self.declare_parameter("dashboard_timeout_sec", 0.7)
        self.declare_parameter("auto_timeout_sec", 0.7)
        self.declare_parameter("act_timeout_sec", 0.7)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.declare_parameter("max_linear_x", 0.10)
        self.declare_parameter("max_linear_y", 0.10)
        self.declare_parameter("max_angular_z", 0.5236)

        self.dashboard_timeout_sec = float(self.get_parameter("dashboard_timeout_sec").value)
        self.auto_timeout_sec = float(self.get_parameter("auto_timeout_sec").value)
        self.act_timeout_sec = float(self.get_parameter("act_timeout_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.max_linear_x = float(self.get_parameter("max_linear_x").value)
        self.max_linear_y = float(self.get_parameter("max_linear_y").value)
        self.max_angular_z = float(self.get_parameter("max_angular_z").value)

        self.sources = {
            "dashboard": CmdSource("dashboard", self.dashboard_timeout_sec),
            "auto": CmdSource("auto", self.auto_timeout_sec),
            "act": CmdSource("act", self.act_timeout_sec),
        }

        self.emergency_stop = False
        self.act_enabled = False

        self.pub = self.create_publisher(Twist, "/safe_cmd_vel", 10)

        self.create_subscription(Twist, "/dashboard/cmd_vel", self.on_dashboard_cmd, 10)
        self.create_subscription(Twist, "/auto/cmd_vel", self.on_auto_cmd, 10)
        self.create_subscription(Twist, "/act/cmd_vel", self.on_act_cmd, 10)
        self.create_subscription(Bool, "/emergency_stop", self.on_emergency_stop, 10)
        self.create_subscription(Bool, "/act/enabled", self.on_act_enabled, 10)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.last_selected_source = "none"

        self.get_logger().info("cmd_vel_mux_node_act_guard started")
        self.get_logger().info("priority: emergency_stop > act_hold > dashboard > auto > act")
        self.get_logger().info("output: /safe_cmd_vel")

    def now_sec(self) -> float:
        return time.monotonic()

    def update_source(self, key: str, msg: Twist):
        src = self.sources[key]
        src.last_msg = clamp_twist(
            msg,
            self.max_linear_x,
            self.max_linear_y,
            self.max_angular_z,
        )
        src.last_time = self.now_sec()

    def on_dashboard_cmd(self, msg: Twist):
        self.update_source("dashboard", msg)

    def on_auto_cmd(self, msg: Twist):
        self.update_source("auto", msg)

    def on_act_cmd(self, msg: Twist):
        self.update_source("act", msg)

    def on_emergency_stop(self, msg: Bool):
        self.emergency_stop = bool(msg.data)
        if self.emergency_stop:
            self.pub.publish(zero_twist())
            self.get_logger().warn("EMERGENCY STOP active")
        else:
            self.get_logger().warn("EMERGENCY STOP released")

    def on_act_enabled(self, msg: Bool):
        self.act_enabled = bool(msg.data)
        if self.act_enabled:
            self.pub.publish(zero_twist())
            self.get_logger().warn("ACT enabled: holding base at zero velocity")
        else:
            self.get_logger().warn("ACT disabled: base command mux released")

    def is_active(self, key: str) -> bool:
        src = self.sources[key]
        if src.last_msg is None:
            return False

        age = self.now_sec() - src.last_time
        return age <= src.timeout_sec

    def select_cmd(self) -> tuple[str, Twist]:
        if self.emergency_stop:
            return "emergency_stop", zero_twist()

        # During ACT manipulation the base must be physically stationary.
        # Dashboard/manual/auto commands are ignored until /act/enabled becomes false.
        if self.act_enabled:
            return "act_hold", zero_twist()

        if self.is_active("dashboard"):
            msg = self.sources["dashboard"].last_msg
            if msg is not None:
                return "dashboard", msg

        if self.is_active("auto"):
            msg = self.sources["auto"].last_msg
            if msg is not None:
                return "auto", msg

        if self.is_active("act"):
            msg = self.sources["act"].last_msg
            if msg is not None:
                return "act", msg

        return "none", zero_twist()

    def on_timer(self):
        selected, cmd = self.select_cmd()

        if selected != self.last_selected_source:
            self.get_logger().info(f"selected cmd source: {selected}")
            self.last_selected_source = selected

        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = CmdVelMuxNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(zero_twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()