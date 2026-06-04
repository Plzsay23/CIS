#!/usr/bin/env python3

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from tf2_ros import TransformBroadcaster

from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode


ARM_MOTORS = {
    "arm_shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    # Keep the gripper exactly on the LeKiwi/LeRobot normalized path.
    # Do not raw-jog this motor unless you are recalibrating it.
    "arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

WHEEL_MOTORS = {
    "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
}

ARM_RANGES = {
    "arm_shoulder_pan": (695, 3379),
    "arm_shoulder_lift": (841, 3237),
    "arm_elbow_flex": (928, 3076),
    "arm_wrist_flex": (980, 3258),
    "arm_wrist_roll": (0, 4095),
    "arm_gripper": (2046, 3100),
}

ARM_ID_TO_NAME = {
    1: "arm_shoulder_pan",
    2: "arm_shoulder_lift",
    3: "arm_elbow_flex",
    4: "arm_wrist_flex",
    5: "arm_wrist_roll",
    6: "arm_gripper",
}

# UI direction correction.
# User-facing 1, 3, 4 are inverted relative to motor positive direction.
INVERT_ARM_JOG_MOTOR_IDS = {1, 3, 4}

DEFAULT_HOME_RAW_TICKS = {
    "arm_shoulder_pan": 2056,
    "arm_shoulder_lift": 871,
    "arm_elbow_flex": 3035,
    "arm_wrist_flex": 2925,
    "arm_wrist_roll": 1030,
    # Gripper is intentionally not used by default during home return.
    "arm_gripper": 2046,
}


def yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(round(max(lo, min(hi, v))))


def load_motor_calibration_json(path_str: str):
    path = Path(path_str).expanduser()
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    calibration = {}
    for name, item in data.items():
        calibration[name] = MotorCalibration(
            id=int(item["id"]),
            drive_mode=int(item.get("drive_mode", 0)),
            homing_offset=int(item.get("homing_offset", 0)),
            range_min=int(item["range_min"]),
            range_max=int(item["range_max"]),
        )

    return calibration


def select_bus_motors(disable_gripper: bool) -> Dict[str, Motor]:
    arm = {
        name: motor
        for name, motor in ARM_MOTORS.items()
        if not (disable_gripper and name == "arm_gripper")
    }
    return {**arm, **WHEEL_MOTORS}


class LeKiwiBaseDriverOdom(Node):
    def __init__(self, args):
        super().__init__("lekiwi_base_driver_odom")

        self.port = args.port
        self.control_hz = float(args.control_hz)
        self.cmd_timeout_s = float(args.cmd_timeout_s)
        self.cmd_topic = args.cmd_topic

        self.max_xy = float(args.max_xy)
        self.max_theta = math.radians(float(args.max_theta_deg))

        self.wheel_radius = float(args.wheel_radius)
        self.base_radius = float(args.base_radius)
        self.max_raw = int(args.max_raw)

        self.odom_frame = args.odom_frame
        self.base_frame = args.base_frame

        self.publish_tf = not args.no_tf
        self.use_measured_wheel_velocity = bool(args.use_measured_wheel_velocity)

        self.disable_gripper = bool(args.disable_gripper)
        self.home_include_gripper = bool(args.home_include_gripper)

        self.arm_acceleration = int(args.arm_acceleration)
        self.arm_jog_ticks_per_s = float(args.arm_jog_ticks_per_s)
        self.arm_jog_timeout_s = float(args.arm_jog_timeout_s)
        self.arm_home_json = args.arm_home_json
        self.arm_home_return_seconds = float(args.arm_home_return_seconds)
        self.arm_home_return_fps = float(args.arm_home_return_fps)

        # These are UI-space values. With gripper_invert=True, the value written
        # to the follower bus is 100 - ui_value, matching teleoperate.py behavior.
        self.gripper_open_norm = float(args.gripper_open_norm)
        self.gripper_close_norm = float(args.gripper_close_norm)
        self.gripper_jog_norm_per_s = float(args.gripper_jog_norm_per_s)
        self.gripper_invert = bool(args.gripper_invert)
        self.gripper_motion_seconds = float(args.gripper_motion_seconds)

        self.calibration_json = args.calibration_json
        self.calibration = load_motor_calibration_json(self.calibration_json)

        self.active_motors = select_bus_motors(self.disable_gripper)
        self.active_arm_names = [
            name
            for name in ARM_MOTORS.keys()
            if name in self.active_motors
        ]
        self.active_wheel_names = list(WHEEL_MOTORS.keys())

        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=self.active_motors,
            calibration=self.calibration,
        )

        self.last_cmd_time: Optional[float] = None
        self.last_loop_time = time.monotonic()

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.estop = False

        self.arm_motion_busy = False
        self.arm_jog_direction: Dict[str, int] = {}
        self.arm_jog_last_time: Dict[str, float] = {}
        self.arm_position_targets: Dict[str, int] = {}
        self.gripper_ui_target: Optional[float] = None

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)

        self.cmd_sub = self.create_subscription(Twist, self.cmd_topic, self.on_cmd_vel, 10)
        self.estop_sub = self.create_subscription(Bool, "/base/estop", self.on_base_estop, 10)
        self.dashboard_estop_sub = self.create_subscription(Bool, "/emergency_stop", self.on_dashboard_estop, 10)
        self.arm_cmd_sub = self.create_subscription(String, "/dashboard/arm_cmd", self.on_arm_cmd, 10)

        self.connect_and_configure()

        period = 1.0 / self.control_hz
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            "LeKiwi base+arm dashboard driver started. "
            f"port={self.port}, cmd_topic={self.cmd_topic}, "
            f"disable_gripper={self.disable_gripper}, "
            f"gripper_invert={self.gripper_invert}, "
            f"active_arm_names={self.active_arm_names}"
        )

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        raw = int(round(degps * steps_per_deg))
        return int(clamp(raw, -3000, 3000))

    @staticmethod
    def raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        return float(raw_speed) / steps_per_deg

    def safe_write(self, register: str, motor_name: str, value, normalize: Optional[bool] = None, num_retry: int = 5) -> bool:
        try:
            if normalize is None:
                self.bus.write(register, motor_name, value, num_retry=num_retry)
            else:
                self.bus.write(register, motor_name, value, normalize=normalize, num_retry=num_retry)
            return True
        except TypeError:
            try:
                self.bus.write(register, motor_name, value, num_retry=num_retry)
                return True
            except Exception as e:
                self.get_logger().warn(f"write failed: {register} {motor_name}={value}: {e}")
                return False
        except Exception as e:
            self.get_logger().warn(f"write failed: {register} {motor_name}={value}: {e}")
            return False

    def safe_enable_torque_one(self, name: str) -> bool:
        ok = self.safe_write("Torque_Enable", name, 1, num_retry=5)

        # Lock write may fail on a noisy bus. It should not kill the node.
        try:
            self.bus.write("Lock", name, 1, num_retry=5)
        except Exception as e:
            self.get_logger().warn(f"Lock write skipped/failed for {name}: {e}")

        return ok

    def safe_disable_torque_one(self, name: str) -> bool:
        ok = True
        try:
            self.bus.write("Lock", name, 0, num_retry=3)
        except Exception as e:
            self.get_logger().warn(f"Lock unlock skipped/failed for {name}: {e}")

        try:
            self.bus.write("Torque_Enable", name, 0, num_retry=3)
        except Exception as e:
            self.get_logger().warn(f"Torque disable skipped/failed for {name}: {e}")
            ok = False

        return ok

    def body_to_wheel_raw(self, vx: float, vy: float, wz_rad: float) -> Dict[str, int]:
        velocity_vector = np.array([vx, vy, wz_rad], dtype=float)

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), self.base_radius] for a in angles])

        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / self.wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / math.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_abs = [abs(v) * steps_per_deg for v in wheel_degps]
        max_raw_computed = max(raw_abs) if raw_abs else 0.0

        if max_raw_computed > self.max_raw:
            scale = self.max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        wheel_raw = [self.degps_to_raw(v) for v in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def wheel_raw_to_body(self, wheel_raw: Dict[str, int]):
        wheel_degps = np.array(
            [
                self.raw_to_degps(wheel_raw["base_left_wheel"]),
                self.raw_to_degps(wheel_raw["base_back_wheel"]),
                self.raw_to_degps(wheel_raw["base_right_wheel"]),
            ],
            dtype=float,
        )

        wheel_radps = wheel_degps * (math.pi / 180.0)
        wheel_linear_speeds = wheel_radps * self.wheel_radius

        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), self.base_radius] for a in angles])

        vx, vy, wz = np.linalg.inv(m).dot(wheel_linear_speeds)
        return float(vx), float(vy), float(wz)

    def connect_and_configure(self):
        self.get_logger().info(f"Connecting Feetech bus: {self.port}")
        self.bus.connect()

        self.get_logger().info("Configuring active arm motors as POSITION and wheels as VELOCITY")

        for name in self.active_arm_names:
            self.safe_disable_torque_one(name)
            self.safe_write("Operating_Mode", name, OperatingMode.POSITION.value, num_retry=5)

            # Conservative enough for dashboard/manual control.
            self.safe_write("P_Coefficient", name, 16, num_retry=3)
            self.safe_write("I_Coefficient", name, 0, num_retry=3)
            self.safe_write("D_Coefficient", name, 32, num_retry=3)
            self.safe_write("Acceleration", name, self.arm_acceleration, num_retry=3)

        for name in self.active_wheel_names:
            self.safe_disable_torque_one(name)
            self.safe_write("Operating_Mode", name, OperatingMode.VELOCITY.value, num_retry=5)
            self.safe_write("Acceleration", name, 254, num_retry=3)

        for name in self.active_arm_names + self.active_wheel_names:
            self.safe_enable_torque_one(name)

        self.stop_motors()

        try:
            self.arm_position_targets.update(self.read_arm_positions_raw())
            self.get_logger().info(f"Initial arm raw ticks: {self.arm_position_targets}")
        except Exception as e:
            self.get_logger().warn(f"Initial arm position read failed: {e}")

        if "arm_gripper" in self.active_arm_names:
            try:
                self.gripper_ui_target = self.read_gripper_ui_norm()
                self.get_logger().info(f"Initial gripper UI norm: {self.gripper_ui_target:.2f}")
            except Exception as e:
                self.get_logger().warn(f"Initial gripper normalized read failed: {e}")

    def stop_motors(self):
        self.bus.sync_write(
            "Goal_Velocity",
            {
                "base_left_wheel": 0,
                "base_back_wheel": 0,
                "base_right_wheel": 0,
            },
        )

    def clear_base_cmd(self):
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.last_cmd_time = None

    def stop_base_for_arm(self):
        self.clear_base_cmd()
        self.stop_motors()

    def has_active_arm_jog(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()

        for _, t in list(self.arm_jog_last_time.items()):
            if now - t <= self.arm_jog_timeout_s:
                return True

        return False

    def on_cmd_vel(self, msg: Twist):
        if self.arm_motion_busy or self.has_active_arm_jog():
            return

        self.cmd_vx = clamp(float(msg.linear.x), -self.max_xy, self.max_xy)
        self.cmd_vy = clamp(float(msg.linear.y), -self.max_xy, self.max_xy)
        self.cmd_wz = clamp(float(msg.angular.z), -self.max_theta, self.max_theta)
        self.last_cmd_time = time.monotonic()

    def on_base_estop(self, msg: Bool):
        self.estop = bool(msg.data)
        if self.estop:
            self.get_logger().warn("/base/estop enabled. Stopping motors.")
            self.clear_base_cmd()
            self.stop_arm_jog_all()
            self.stop_motors()

    def on_dashboard_estop(self, msg: Bool):
        self.estop = bool(msg.data)
        if self.estop:
            self.get_logger().warn("/emergency_stop enabled. Stopping motors.")
            self.clear_base_cmd()
            self.stop_arm_jog_all()
            self.stop_motors()
        else:
            self.get_logger().warn("/emergency_stop released.")

    def get_active_cmd(self):
        now = time.monotonic()

        if self.estop or self.arm_motion_busy or self.has_active_arm_jog(now):
            return 0.0, 0.0, 0.0

        if self.last_cmd_time is None:
            return 0.0, 0.0, 0.0

        if now - self.last_cmd_time > self.cmd_timeout_s:
            return 0.0, 0.0, 0.0

        return self.cmd_vx, self.cmd_vy, self.cmd_wz

    def read_body_velocity(self, fallback_vx, fallback_vy, fallback_wz):
        if not self.use_measured_wheel_velocity:
            return fallback_vx, fallback_vy, fallback_wz

        try:
            raw = self.bus.sync_read("Present_Velocity", self.active_wheel_names)
            return self.wheel_raw_to_body(raw)
        except Exception as e:
            self.get_logger().warn(f"Present_Velocity read failed. Falling back to cmd odom: {e}")
            return fallback_vx, fallback_vy, fallback_wz

    def integrate_odom(self, vx_body, vy_body, wz, dt):
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body

        self.x += vx_world * dt
        self.y += vy_world * dt
        self.yaw += wz * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

    def publish_odom(self, vx_body, vy_body, wz):
        now = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.linear.y = vy_body
        odom.twist.twist.angular.z = wz

        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10
        odom.twist.covariance[0] = 0.10
        odom.twist.covariance[7] = 0.10
        odom.twist.covariance[35] = 0.20

        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame

            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw

            self.tf_broadcaster.sendTransform(tf)

    def clamp_arm_tick(self, name: str, tick: float) -> int:
        lo, hi = ARM_RANGES[name]
        return clamp_int(tick, lo, hi)

    def read_arm_positions_raw(self) -> Dict[str, int]:
        names = [name for name in self.active_arm_names if name != "arm_gripper"]
        if not names:
            return {}

        try:
            raw = self.bus.sync_read("Present_Position", names, normalize=False)
        except TypeError:
            raw = self.bus.sync_read("Present_Position", names)

        return {name: int(raw[name]) for name in names}

    def read_arm_position_raw(self, name: str) -> int:
        try:
            return int(self.bus.read("Present_Position", name, normalize=False, num_retry=3))
        except TypeError:
            return int(self.bus.read("Present_Position", name))

    def write_arm_positions_raw(self, targets: Dict[str, int]):
        clamped = {
            name: self.clamp_arm_tick(name, value)
            for name, value in targets.items()
            if name in self.active_arm_names and name != "arm_gripper"
        }

        if not clamped:
            return

        try:
            self.bus.sync_write("Goal_Position", clamped, normalize=False)
        except TypeError:
            self.bus.sync_write("Goal_Position", clamped)

        self.arm_position_targets.update(clamped)

    # ----------------------------------------------------------------------
    # Gripper: normalized LeKiwi/teleoperate.py-compatible path only.
    # No raw tick write, no raw jog.
    # ----------------------------------------------------------------------
    def follower_norm_from_ui_norm(self, ui_norm: float) -> float:
        ui_norm = float(clamp(ui_norm, 0.0, 100.0))
        if self.gripper_invert:
            return 100.0 - ui_norm
        return ui_norm

    def ui_norm_from_follower_norm(self, follower_norm: float) -> float:
        follower_norm = float(clamp(follower_norm, 0.0, 100.0))
        if self.gripper_invert:
            return 100.0 - follower_norm
        return follower_norm

    def read_gripper_follower_norm(self) -> float:
        if "arm_gripper" not in self.active_arm_names:
            raise RuntimeError("gripper is disabled")

        try:
            return float(self.bus.read("Present_Position", "arm_gripper", normalize=True, num_retry=3))
        except TypeError:
            # If this fallback is hit, the installed bus API may not support normalize=True
            # on read. Keep the error explicit rather than raw-writing the gripper.
            raise RuntimeError("This FeetechMotorsBus.read does not support normalize=True for gripper")

    def read_gripper_ui_norm(self) -> float:
        return self.ui_norm_from_follower_norm(self.read_gripper_follower_norm())

    def write_gripper_ui_norm(self, ui_norm: float, num_retry: int = 3):
        if "arm_gripper" not in self.active_arm_names:
            self.get_logger().warn("Ignoring gripper write because gripper is disabled.")
            return

        ui_norm = float(clamp(ui_norm, 0.0, 100.0))
        follower_norm = self.follower_norm_from_ui_norm(ui_norm)

        self.bus.write(
            "Goal_Position",
            "arm_gripper",
            follower_norm,
            normalize=True,
            num_retry=num_retry,
        )
        self.gripper_ui_target = ui_norm

    def move_gripper_ui_smooth(self, target_ui_norm: float):
        if "arm_gripper" not in self.active_arm_names:
            self.get_logger().warn("Ignoring gripper motion because gripper is disabled.")
            return

        self.stop_base_for_arm()
        self.stop_arm_jog_all()

        target_ui_norm = float(clamp(target_ui_norm, 0.0, 100.0))

        try:
            current_ui = self.read_gripper_ui_norm()
        except Exception as e:
            self.get_logger().warn(f"Gripper normalized read failed, using cached/default target: {e}")
            current_ui = self.gripper_ui_target
            if current_ui is None:
                current_ui = target_ui_norm

        duration = max(0.05, self.gripper_motion_seconds)
        fps = max(1.0, self.arm_home_return_fps)
        steps = max(1, int(round(duration * fps)))
        sleep_s = 1.0 / fps

        self.get_logger().info(
            f"Gripper UI smooth move: {current_ui:.2f} -> {target_ui_norm:.2f}, "
            f"invert={self.gripper_invert}, duration={duration:.2f}s, steps={steps}"
        )

        for i in range(1, steps + 1):
            alpha = i / steps
            ui_value = current_ui * (1.0 - alpha) + target_ui_norm * alpha
            self.write_gripper_ui_norm(ui_value, num_retry=2)
            time.sleep(sleep_s)

        self.write_gripper_ui_norm(target_ui_norm, num_retry=5)

    def move_gripper_open(self):
        self.get_logger().info(f"gripper_open UI norm -> {self.gripper_open_norm}")
        self.move_gripper_ui_smooth(self.gripper_open_norm)

    def move_gripper_close(self):
        self.get_logger().info(f"gripper_close UI norm -> {self.gripper_close_norm}")
        self.move_gripper_ui_smooth(self.gripper_close_norm)

    def start_gripper_jog(self, direction: int):
        if "arm_gripper" not in self.active_arm_names:
            self.get_logger().warn("Ignoring motor_6 jog because gripper is disabled.")
            return

        now = time.monotonic()
        self.stop_base_for_arm()

        if self.gripper_ui_target is None:
            try:
                self.gripper_ui_target = self.read_gripper_ui_norm()
            except Exception as e:
                self.get_logger().warn(f"Cannot read gripper norm; starting from 50.0: {e}")
                self.gripper_ui_target = 50.0

        self.arm_jog_direction["arm_gripper"] = 1 if direction > 0 else -1
        self.arm_jog_last_time["arm_gripper"] = now

    def update_gripper_jog(self, dt: float, direction: int):
        if self.gripper_ui_target is None:
            try:
                self.gripper_ui_target = self.read_gripper_ui_norm()
            except Exception:
                self.gripper_ui_target = 50.0

        next_ui = float(clamp(
            self.gripper_ui_target + direction * self.gripper_jog_norm_per_s * dt,
            0.0,
            100.0,
        ))

        self.write_gripper_ui_norm(next_ui, num_retry=1)

        if next_ui <= 0.0 or next_ui >= 100.0:
            self.arm_jog_direction.pop("arm_gripper", None)
            self.arm_jog_last_time.pop("arm_gripper", None)

    # ----------------------------------------------------------------------
    # Home return: raw deterministic return for 1~5 by default.
    # Gripper is excluded unless --home-include-gripper is explicitly passed.
    # ----------------------------------------------------------------------
    def load_home_ticks(self) -> Dict[str, int]:
        path = Path(self.arm_home_json).expanduser()

        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            self.get_logger().warn(f"Home json not found: {path}. Using built-in defaults.")
            data = DEFAULT_HOME_RAW_TICKS

        home = {}
        for name in self.active_arm_names:
            if name == "arm_gripper" and not self.home_include_gripper:
                continue
            if name == "arm_gripper":
                # Do not raw-home the gripper. It uses normalized open/close only.
                continue
            if name not in data:
                raise ValueError(f"Missing home tick for {name} in {path}")
            home[name] = self.clamp_arm_tick(name, int(data[name]))

        return home

    def stop_arm_jog_all(self):
        self.arm_jog_direction.clear()
        self.arm_jog_last_time.clear()

    def stop_arm_jog_motor(self, motor_id: int):
        if motor_id not in ARM_ID_TO_NAME:
            return

        name = ARM_ID_TO_NAME[motor_id]
        self.arm_jog_direction.pop(name, None)
        self.arm_jog_last_time.pop(name, None)
        self.get_logger().info(f"Stopped arm jog motor {motor_id} ({name})")

    def start_arm_jog(self, motor_id: int, direction: int):
        if motor_id not in ARM_ID_TO_NAME:
            self.get_logger().warn(f"Invalid arm motor id: {motor_id}")
            return

        name = ARM_ID_TO_NAME[motor_id]

        if name == "arm_gripper":
            # Gripper jog is normalized, not raw.
            self.start_gripper_jog(direction)
            return

        if name not in self.active_arm_names:
            self.get_logger().warn(f"Ignoring inactive arm motor command: {motor_id} ({name})")
            return

        now = time.monotonic()
        self.stop_base_for_arm()

        if name not in self.arm_position_targets:
            self.arm_position_targets[name] = self.clamp_arm_tick(
                name,
                self.read_arm_position_raw(name),
            )

        corrected_direction = 1 if direction > 0 else -1
        if motor_id in INVERT_ARM_JOG_MOTOR_IDS:
            corrected_direction *= -1

        self.arm_jog_direction[name] = corrected_direction
        self.arm_jog_last_time[name] = now

    def update_arm_jog(self, dt: float):
        if self.arm_motion_busy or self.estop:
            return

        now = time.monotonic()
        targets = {}

        for name in list(self.arm_jog_direction.keys()):
            last = self.arm_jog_last_time.get(name, 0.0)

            if now - last > self.arm_jog_timeout_s:
                self.arm_jog_direction.pop(name, None)
                self.arm_jog_last_time.pop(name, None)
                continue

            direction = self.arm_jog_direction[name]

            if name == "arm_gripper":
                self.update_gripper_jog(dt, direction)
                continue

            if name not in self.arm_position_targets:
                self.arm_position_targets[name] = self.clamp_arm_tick(
                    name,
                    self.read_arm_position_raw(name),
                )

            current_target = self.arm_position_targets[name]
            next_target = self.clamp_arm_tick(
                name,
                current_target + direction * self.arm_jog_ticks_per_s * dt,
            )

            self.arm_position_targets[name] = next_target
            targets[name] = next_target

            lo, hi = ARM_RANGES[name]
            if next_target <= lo or next_target >= hi:
                self.arm_jog_direction.pop(name, None)
                self.arm_jog_last_time.pop(name, None)

        if targets:
            self.write_arm_positions_raw(targets)

    def move_arm_targets_smooth(self, final_targets: Dict[str, int], seconds: Optional[float] = None):
        if seconds is None:
            seconds = self.arm_home_return_seconds

        self.stop_base_for_arm()
        self.stop_arm_jog_all()

        names = [
            name
            for name in final_targets
            if name in self.active_arm_names and name != "arm_gripper"
        ]
        if not names:
            return

        current_all = self.read_arm_positions_raw()
        current = {name: current_all[name] for name in names}
        final = {
            name: self.clamp_arm_tick(name, final_targets[name])
            for name in names
        }

        duration = max(0.05, float(seconds))
        fps = max(1.0, float(self.arm_home_return_fps))
        steps = max(1, int(round(duration * fps)))
        sleep_s = 1.0 / fps

        self.get_logger().info(
            f"Smooth arm move: motors={names}, duration={duration:.2f}s, fps={fps:.1f}, steps={steps}, target={final}"
        )

        for i in range(1, steps + 1):
            alpha = i / steps
            target = {}

            for name in names:
                c = current[name]
                h = final[name]
                value = c * (1.0 - alpha) + h * alpha
                target[name] = self.clamp_arm_tick(name, value)

            self.write_arm_positions_raw(target)
            time.sleep(sleep_s)

        self.write_arm_positions_raw(final)

    def return_arm_home(self):
        home = self.load_home_ticks()
        self.get_logger().info(f"Returning arm home without raw gripper -> {home}")
        self.move_arm_targets_smooth(home, seconds=self.arm_home_return_seconds)
        self.get_logger().info("Arm home return complete.")

    def on_arm_cmd(self, msg: String):
        command = msg.data.strip()

        if not command:
            return

        if self.estop:
            self.get_logger().warn(f"Ignoring arm command during estop: {command}")
            return

        try:
            if command == "gripper_open":
                if self.disable_gripper:
                    self.get_logger().warn("Ignoring gripper_open because --disable-gripper is set.")
                    return
                self.arm_motion_busy = True
                self.move_gripper_open()

            elif command == "gripper_close":
                if self.disable_gripper:
                    self.get_logger().warn("Ignoring gripper_close because --disable-gripper is set.")
                    return
                self.arm_motion_busy = True
                self.move_gripper_close()

            elif command == "arm_home":
                self.arm_motion_busy = True
                self.return_arm_home()

            elif command.startswith("motor_"):
                parts = command.split("_")
                if len(parts) != 3:
                    self.get_logger().warn(f"Invalid motor command: {command}")
                    return

                motor_id = int(parts[1])
                op = parts[2]

                if op == "up":
                    self.start_arm_jog(motor_id, 1)
                elif op == "down":
                    self.start_arm_jog(motor_id, -1)
                elif op == "stop":
                    self.stop_arm_jog_motor(motor_id)
                else:
                    self.get_logger().warn(f"Invalid motor jog op: {command}")

            else:
                self.get_logger().warn(f"Unsupported arm command: {command}")

        except Exception as e:
            self.get_logger().error(f"Arm command failed: {command}: {e}")

        finally:
            if command in {"gripper_open", "gripper_close", "arm_home"}:
                self.stop_base_for_arm()
                self.arm_motion_busy = False

    def on_timer(self):
        now = time.monotonic()
        dt = now - self.last_loop_time
        self.last_loop_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 1.0 / self.control_hz

        vx_cmd, vy_cmd, wz_cmd = self.get_active_cmd()
        wheel_goal = self.body_to_wheel_raw(vx_cmd, vy_cmd, wz_cmd)

        try:
            self.bus.sync_write("Goal_Velocity", wheel_goal)
        except Exception as e:
            self.get_logger().error(f"Goal_Velocity write failed: {e}")
            return

        try:
            self.update_arm_jog(dt)
        except Exception as e:
            self.get_logger().error(f"Arm jog update failed: {e}")

        vx_odom, vy_odom, wz_odom = self.read_body_velocity(vx_cmd, vy_cmd, wz_cmd)
        self.integrate_odom(vx_odom, vy_odom, wz_odom, dt)
        self.publish_odom(vx_odom, vy_odom, wz_odom)

    def shutdown(self):
        self.get_logger().info("Stopping motors and disconnecting.")
        try:
            self.stop_motors()
        except Exception:
            pass

        try:
            self.bus.disconnect()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--cmd-timeout-s", type=float, default=0.7)
    parser.add_argument("--cmd-topic", default="/safe_cmd_vel")

    parser.add_argument("--max-xy", type=float, default=0.10)
    parser.add_argument("--max-theta-deg", type=float, default=30.0)

    parser.add_argument("--wheel-radius", type=float, default=0.05)
    parser.add_argument("--base-radius", type=float, default=0.125)
    parser.add_argument("--max-raw", type=int, default=3000)

    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")

    parser.add_argument("--no-tf", action="store_true")
    parser.add_argument("--use-measured-wheel-velocity", action="store_true")

    parser.add_argument("--calibration-json", default="/home/lerobot/CIS/config/lekiwi.json")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--home-include-gripper", action="store_true")

    parser.add_argument("--arm-acceleration", type=int, default=180)
    parser.add_argument("--arm-jog-ticks-per-s", type=float, default=700.0)
    parser.add_argument("--arm-jog-timeout-s", type=float, default=0.25)
    parser.add_argument("--arm-home-json", default="/home/lerobot/CIS/config/arm_home_raw_ticks.json")
    parser.add_argument("--arm-home-return-seconds", type=float, default=0.9)
    parser.add_argument("--arm-home-return-fps", type=float, default=40.0)

    # UI-space gripper values. Internally inverted by default to match teleoperate.py:
    # follower_gripper = 100 - ui_gripper
    parser.add_argument("--gripper-open-norm", type=float, default=0.0)
    parser.add_argument("--gripper-close-norm", type=float, default=100.0)
    parser.add_argument("--gripper-jog-norm-per-s", type=float, default=25.0)
    parser.add_argument("--gripper-motion-seconds", type=float, default=1.2)
    parser.add_argument("--no-gripper-invert", dest="gripper_invert", action="store_false")
    parser.set_defaults(gripper_invert=True)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = LeKiwiBaseDriverOdom(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()