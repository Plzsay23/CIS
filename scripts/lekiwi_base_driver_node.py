#!/usr/bin/env python3

import argparse
import math
import time
import traceback

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorNormMode


WHEEL_MOTORS = {
    "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
}


class LeKiwiBaseDriverNode(Node):
    def __init__(
        self,
        port: str,
        cmd_vel_topic: str,
        estop_topic: str,
        control_hz: float,
        cmd_timeout_s: float,
        max_xy: float,
        max_theta_rad: float,
        wheel_radius: float,
        base_radius: float,
        max_raw: int,
    ):
        super().__init__("lekiwi_base_driver")

        self.port = port
        self.cmd_timeout_s = cmd_timeout_s
        self.max_xy = max_xy
        self.max_theta_rad = max_theta_rad
        self.wheel_radius = wheel_radius
        self.base_radius = base_radius
        self.max_raw = max_raw

        self.motor_names = list(WHEEL_MOTORS.keys())
        self.bus = None

        self.last_cmd = Twist()
        self.last_cmd_time = time.monotonic()
        self.estop = False
        self.last_sent = None

        self.cmd_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.on_cmd_vel,
            10,
        )

        self.estop_sub = self.create_subscription(
            Bool,
            estop_topic,
            self.on_estop,
            10,
        )

        self.timer = self.create_timer(1.0 / control_hz, self.on_timer)

        self.connect_and_configure()

        self.get_logger().info("LeKiwi base driver started")
        self.get_logger().info(f"port={self.port}")
        self.get_logger().info(f"cmd_vel_topic={cmd_vel_topic}")
        self.get_logger().info(f"estop_topic={estop_topic}")
        self.get_logger().info(f"cmd_timeout_s={self.cmd_timeout_s}")
        self.get_logger().info(f"max_xy={self.max_xy} m/s")
        self.get_logger().info(f"max_theta={self.max_theta_rad} rad/s")
        self.get_logger().info(f"max_raw={self.max_raw}")

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        raw = int(round(degps * steps_per_deg))
        return max(min(raw, 0x7FFF), -0x8000)

    def body_to_wheel_raw(self, x: float, y: float, theta_rad_s: float) -> dict[str, int]:
        """
        ROS2 Twist 기준:
          x: linear.x, m/s, 전진 +
          y: linear.y, m/s, 좌측 +
          theta_rad_s: angular.z, rad/s, 반시계 +

        기존 LeKiwi _body_to_wheel_raw()는 theta를 deg/s로 받았으므로,
        여기서는 ROS2 angular.z(rad/s)를 내부에서 deg/s로 바꾼다.
        """
        theta_deg_s = math.degrees(theta_rad_s)
        theta_rad_for_matrix = theta_deg_s * (np.pi / 180.0)

        velocity_vector = np.array([x, y, theta_rad_for_matrix])

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), self.base_radius] for a in angles])

        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / self.wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]

        max_raw_computed = max(raw_floats)
        if max_raw_computed > self.max_raw:
            scale = self.max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        wheel_raw = [self.degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def connect_and_configure(self):
        self.get_logger().info("Connecting Feetech bus...")

        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=WHEEL_MOTORS,
        )
        self.bus.connect()

        for name, motor in WHEEL_MOTORS.items():
            model_number = self.bus.ping(name)
            self.get_logger().info(f"{name} id={motor.id} ping={model_number}")

        self.get_logger().info("Configuring wheel motors only...")

        self.bus.disable_torque(self.motor_names)

        for name in self.motor_names:
            self.bus.write(
                "Operating_Mode",
                name,
                OperatingMode.VELOCITY.value,
                normalize=False,
            )

        for name in self.motor_names:
            try:
                self.bus.write("Maximum_Acceleration", name, 254, normalize=False)
            except Exception as e:
                self.get_logger().warn(f"skip {name}.Maximum_Acceleration: {e}")

            try:
                self.bus.write("Acceleration", name, 254, normalize=False)
            except Exception as e:
                self.get_logger().warn(f"skip {name}.Acceleration: {e}")

        self.bus.enable_torque(self.motor_names)
        self.stop_all()

        self.get_logger().info("Wheel motors configured")

    def clamp_cmd(self, msg: Twist) -> tuple[float, float, float]:
        x = max(min(float(msg.linear.x), self.max_xy), -self.max_xy)
        y = max(min(float(msg.linear.y), self.max_xy), -self.max_xy)
        theta = max(min(float(msg.angular.z), self.max_theta_rad), -self.max_theta_rad)
        return x, y, theta

    def on_cmd_vel(self, msg: Twist):
        self.last_cmd = msg
        self.last_cmd_time = time.monotonic()

    def on_estop(self, msg: Bool):
        self.estop = bool(msg.data)
        if self.estop:
            self.get_logger().warn("E-STOP enabled. Stopping base.")
            self.stop_all()
        else:
            self.get_logger().info("E-STOP released.")

    def on_timer(self):
        if self.bus is None:
            return

        now = time.monotonic()

        if self.estop:
            self.stop_all_if_needed()
            return

        if now - self.last_cmd_time > self.cmd_timeout_s:
            self.stop_all_if_needed()
            return

        x, y, theta = self.clamp_cmd(self.last_cmd)
        wheel_cmd = self.body_to_wheel_raw(x, y, theta)

        if wheel_cmd != self.last_sent:
            self.bus.sync_write("Goal_Velocity", wheel_cmd, normalize=False)
            self.last_sent = dict(wheel_cmd)

    def stop_all_if_needed(self):
        zero = dict.fromkeys(self.motor_names, 0)
        if self.last_sent != zero:
            self.stop_all()

    def stop_all(self):
        if self.bus is None:
            return

        zero = dict.fromkeys(self.motor_names, 0)
        try:
            self.bus.sync_write("Goal_Velocity", zero, normalize=False, num_retry=5)
            self.last_sent = dict(zero)
        except Exception as e:
            self.get_logger().error(f"Failed to stop base: {e}")

    def shutdown(self):
        self.get_logger().info("Shutting down LeKiwi base driver...")

        try:
            self.stop_all()
        except Exception:
            traceback.print_exc()

        try:
            if self.bus is not None:
                self.bus.disable_torque(self.motor_names)
        except Exception:
            traceback.print_exc()

        try:
            if self.bus is not None:
                self.bus.disconnect(disable_torque=False)
        except Exception:
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--estop-topic", default="/base/estop")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--cmd-timeout-s", type=float, default=0.5)

    # 기존 LeKiwi 텔레옵 slow/medium/fast가 0.1/0.2/0.3 m/s, 30/60/90 deg/s 구조다.
    parser.add_argument("--max-xy", type=float, default=0.3)
    parser.add_argument("--max-theta-deg", type=float, default=90.0)

    parser.add_argument("--wheel-radius", type=float, default=0.05)
    parser.add_argument("--base-radius", type=float, default=0.125)
    parser.add_argument("--max-raw", type=int, default=3000)

    args = parser.parse_args()

    rclpy.init()

    node = None
    try:
        node = LeKiwiBaseDriverNode(
            port=args.port,
            cmd_vel_topic=args.cmd_vel_topic,
            estop_topic=args.estop_topic,
            control_hz=args.control_hz,
            cmd_timeout_s=args.cmd_timeout_s,
            max_xy=args.max_xy,
            max_theta_rad=math.radians(args.max_theta_deg),
            wheel_radius=args.wheel_radius,
            base_radius=args.base_radius,
            max_raw=args.max_raw,
        )
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()