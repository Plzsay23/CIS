#!/usr/bin/env python3

import argparse
import time
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Bool, String


class State(str, Enum):
    IDLE = "IDLE"
    PREHOME = "PREHOME"
    ALIGN = "ALIGN"
    SETTLE = "SETTLE"
    ACT = "ACT"
    COOLDOWN = "COOLDOWN"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def zero_twist() -> Twist:
    return Twist()


class EggActMissionManager(Node):
    def __init__(self, args):
        super().__init__("egg_act_mission_manager")

        self.args = args

        self.egg_sub = self.create_subscription(
            PointStamped,
            args.egg_topic,
            self.on_egg_detection,
            10,
        )

        self.auto_cmd_pub = self.create_publisher(Twist, args.auto_cmd_topic, 10)
        self.act_enabled_pub = self.create_publisher(Bool, args.act_enabled_topic, 10)
        self.arm_cmd_pub = self.create_publisher(String, args.arm_cmd_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)

        self.state = State.IDLE
        self.state_start_time = time.monotonic()

        self.last_egg = None
        self.last_egg_time = 0.0

        self.stable_start_time = None
        self.last_mission_end_time = 0.0
        self.act_last_publish_time = 0.0

        self.timer = self.create_timer(1.0 / args.control_hz, self.on_timer)

        self._last_act_enabled = None
        self.publish_act_enabled(False)
        self.publish_stop()

        self.get_logger().info(
            "Egg ACT mission manager started. "
            f"egg_topic={args.egg_topic}, auto_cmd_topic={args.auto_cmd_topic}, "
            f"target_distance={args.target_distance:.3f}, target_y={args.target_y:.3f}, "
            f"x_tol={args.x_tolerance:.3f}, y_tol={args.y_tolerance:.3f}"
        )

    def now(self) -> float:
        return time.monotonic()

    def set_state(self, state: State):
        if state == self.state:
            return

        self.state = state
        self.state_start_time = self.now()
        self.stable_start_time = None

        self.publish_status(f"state={state.value}")
        self.get_logger().warn(f"MISSION STATE -> {state.value}")

    def state_elapsed(self) -> float:
        return self.now() - self.state_start_time

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_stop(self):
        self.auto_cmd_pub.publish(zero_twist())

    def publish_act_enabled(self, enabled: bool, force: bool = False):
        enabled = bool(enabled)

        if not force and getattr(self, "_last_act_enabled", None) == enabled:
            return

        self._last_act_enabled = enabled

        msg = Bool()
        msg.data = enabled
        self.act_enabled_pub.publish(msg)
        
    def publish_arm_home(self):
        msg = String()
        msg.data = "arm_home"
        self.arm_cmd_pub.publish(msg)
        self.get_logger().warn("Published arm_home before ACT alignment")

    def on_egg_detection(self, msg: PointStamped):
        x = float(msg.point.x)
        y = float(msg.point.y)

        if x < self.args.min_valid_x or x > self.args.max_valid_x:
            return
        if abs(y) > self.args.max_valid_abs_y:
            return

        self.last_egg = (x, y)
        self.last_egg_time = self.now()

        if self.state == State.IDLE:
            return

            self.get_logger().warn(
                f"Egg detected. Starting PRE_ACT sequence. egg_x={x:.3f}, egg_y={y:.3f}"
            )

            self.publish_act_enabled(False)
            self.publish_stop()

            if self.args.home_before_align:
                self.publish_arm_home()
                self.set_state(State.PREHOME)
            else:
                self.set_state(State.ALIGN)

    def has_fresh_egg(self) -> bool:
        return (
            self.last_egg is not None
            and self.now() - self.last_egg_time <= self.args.egg_timeout
        )

    def make_align_cmd(self, egg_x: float, egg_y: float) -> Twist:
        err_x = egg_x - self.args.target_distance
        err_y = egg_y - self.args.target_y

        cmd = Twist()

        vx = self.args.kx * err_x
        vy = self.args.ky * err_y

        if self.args.invert_x:
            vx *= -1.0
        if self.args.invert_y:
            vy *= -1.0

        # lateral 이동이 불안하면 yaw도 약간 사용 가능.
        wz = self.args.kyaw * err_y
        if self.args.invert_yaw:
            wz *= -1.0

        cmd.linear.x = clamp(vx, -self.args.max_vx, self.args.max_vx)
        cmd.linear.y = clamp(vy, -self.args.max_vy, self.args.max_vy)
        cmd.angular.z = clamp(wz, -self.args.max_wz, self.args.max_wz)

        return cmd

    def is_aligned(self, egg_x: float, egg_y: float) -> bool:
        err_x = egg_x - self.args.target_distance
        err_y = egg_y - self.args.target_y

        return (
            abs(err_x) <= self.args.x_tolerance
            and abs(err_y) <= self.args.y_tolerance
        )

    def on_timer(self):
        if self.state == State.IDLE:
            self.publish_act_enabled(False)
            return

        if self.state == State.PREHOME:
            self.publish_stop()
            self.publish_act_enabled(False)

            if self.state_elapsed() >= self.args.prehome_wait:
                self.set_state(State.ALIGN)
            return

        if self.state == State.ALIGN:
            self.publish_act_enabled(False)

            if not self.has_fresh_egg():
                self.publish_stop()
                self.publish_status("ALIGN waiting_for_fresh_egg")
                return

            egg_x, egg_y = self.last_egg

            if self.is_aligned(egg_x, egg_y):
                self.publish_stop()

                if self.stable_start_time is None:
                    self.stable_start_time = self.now()
                    self.get_logger().info(
                        f"Alignment inside tolerance. egg_x={egg_x:.3f}, egg_y={egg_y:.3f}"
                    )

                stable_time = self.now() - self.stable_start_time
                self.publish_status(
                    f"ALIGN stable {stable_time:.2f}/{self.args.align_stable_time:.2f}"
                )

                if stable_time >= self.args.align_stable_time:
                    self.set_state(State.SETTLE)

                return

            self.stable_start_time = None
            cmd = self.make_align_cmd(egg_x, egg_y)
            self.auto_cmd_pub.publish(cmd)

            self.publish_status(
                "ALIGN "
                f"egg_x={egg_x:.3f} egg_y={egg_y:.3f} "
                f"target_x={self.args.target_distance:.3f} target_y={self.args.target_y:.3f} "
                f"cmd_x={cmd.linear.x:.3f} cmd_y={cmd.linear.y:.3f} cmd_wz={cmd.angular.z:.3f}"
            )
            return

        if self.state == State.SETTLE:
            self.publish_stop()
            self.publish_act_enabled(False)

            if self.state_elapsed() >= self.args.settle_time:
                self.get_logger().warn("Pre-ACT alignment complete. Enabling ACT.")
                self.set_state(State.ACT)
            return

        if self.state == State.ACT:
            self.publish_stop()

            # 주기적으로 true를 다시 발행해서 late subscriber 문제를 줄인다.
            if self.now() - self.act_last_publish_time >= 0.2:
                self.publish_act_enabled(True, force=True)
                self.act_last_publish_time = self.now()

            self.publish_status(f"ACT running {self.state_elapsed():.2f}/{self.args.act_run_time:.2f}")

            if self.state_elapsed() >= self.args.act_run_time:
                self.get_logger().warn("ACT run time reached. Disabling ACT.")
                self.publish_act_enabled(False)
                self.publish_stop()

                if self.args.home_after_act:
                    self.publish_arm_home()

                self.last_mission_end_time = self.now()
                self.set_state(State.COOLDOWN)
            return

        if self.state == State.COOLDOWN:
            self.publish_act_enabled(False)
            self.publish_stop()

            if self.state_elapsed() >= self.args.cooldown_time:
                self.set_state(State.IDLE)
            return


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--egg-topic", default="/egg_detection")
    parser.add_argument("--auto-cmd-topic", default="/auto/cmd_vel")
    parser.add_argument("--act-enabled-topic", default="/act/enabled")
    parser.add_argument("--arm-cmd-topic", default="/dashboard/arm_cmd")
    parser.add_argument("--status-topic", default="/egg_act/status")

    # 기존 start script 호환용. 내부에서는 ALIGN state가 항상 쓰인다.
    parser.add_argument("--enable-simple-align", action="store_true")

    parser.add_argument("--control-hz", type=float, default=20.0)

    # 데이터셋 시작 위치 기준.
    # egg_detection의 point.x가 base_link 기준 전방 거리라고 가정.
    parser.add_argument("--target-distance", type=float, default=0.28)
    parser.add_argument("--target-y", type=float, default=0.0)

    parser.add_argument("--x-tolerance", type=float, default=0.035)
    parser.add_argument("--y-tolerance", type=float, default=0.035)
    parser.add_argument("--align-stable-time", type=float, default=0.8)

    # detection sanity filter
    parser.add_argument("--egg-timeout", type=float, default=0.5)
    parser.add_argument("--min-valid-x", type=float, default=0.10)
    parser.add_argument("--max-valid-x", type=float, default=1.50)
    parser.add_argument("--max-valid-abs-y", type=float, default=0.80)

    # base align control
    parser.add_argument("--kx", type=float, default=0.45)
    parser.add_argument("--ky", type=float, default=0.45)
    parser.add_argument("--kyaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=0.055)
    parser.add_argument("--max-vy", type=float, default=0.055)
    parser.add_argument("--max-wz", type=float, default=0.25)

    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")

    # phase timings
    parser.add_argument("--home-before-align", action="store_true", default=True)
    parser.add_argument("--no-home-before-align", dest="home_before_align", action="store_false")
    parser.add_argument("--prehome-wait", type=float, default=1.2)

    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--act-run-time", type=float, default=12.0)

    parser.add_argument("--home-after-act", action="store_true", default=True)
    parser.add_argument("--no-home-after-act", dest="home_after_act", action="store_false")
    parser.add_argument("--cooldown-time", type=float, default=2.0)
    parser.add_argument("--restart-cooldown", type=float, default=3.0)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = EggActMissionManager(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.publish_act_enabled(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()