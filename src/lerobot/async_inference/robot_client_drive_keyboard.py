#!/usr/bin/env python3

import logging
import json
import math
import os
import pickle  # nosec
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.feetech.feetech import OperatingMode
from lerobot.transport import services_pb2  # type: ignore
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import Action, Observation, RawObservation, visualize_action_queue_size
from .robot_client import RobotClient


WHEEL_MOTORS = {
    "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
}

WHEEL_NAMES = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]


def load_motor_calibration_json(path_str: str):
    path = os.path.expanduser(path_str)
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
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


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().flatten()[0].item())
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def norm_key(key: str) -> str:
    out = str(key).lower()
    for token in [
        "observation.",
        "action.",
        "state.",
        "motors.",
        "motor.",
        ".pos",
        "_pos",
        ".position",
        "_position",
        "position",
    ]:
        out = out.replace(token, "")
    out = out.replace(".", "_")
    out = out.replace("-", "_")
    return out


class DriveKeyboardRobotClient(RobotClient):
    def __init__(self, config: RobotClientConfig):
        self.bus_lock = threading.RLock()

        super().__init__(config)

        # ACT policy는 SO101 arm 6축으로 학습됨.
        # 이후 wheel motor 7/8/9를 같은 bus에 등록하면 self.robot.action_features가 9개로 늘 수 있다.
        # 따라서 ACT action tensor는 여기서 저장한 arm_action_features에만 매핑한다.
        self.arm_action_features = list(self.robot.action_features)
        self.arm_action_dim = len(self.arm_action_features)
        self.logger.info(f"[ACT] frozen arm action features: {self.arm_action_features}")

        self.policy_enabled = threading.Event()
        self.keyboard_shutdown = threading.Event()

        self.home_action: dict[str, float] | None = None
        self.home_lock = threading.Lock()

        self.wheel_bus = self._find_robot_bus()
        self.wheels_ready = False

        self.wheel_radius = env_float("LEKIWI_WHEEL_RADIUS", 0.05)
        self.base_radius = env_float("LEKIWI_BASE_RADIUS", 0.125)
        self.max_raw = env_int("LEKIWI_MAX_RAW", 700)

        self.linear_speed = env_float("LEKIWI_LINEAR_SPEED", 0.050)
        self.angular_speed = env_float("LEKIWI_ANGULAR_SPEED", 0.45)
        self.linear_step = env_float("LEKIWI_LINEAR_STEP", 0.010)
        self.angular_step = env_float("LEKIWI_ANGULAR_STEP", 0.10)
        self.min_linear_speed = env_float("LEKIWI_MIN_LINEAR_SPEED", 0.015)
        self.max_linear_speed = env_float("LEKIWI_MAX_LINEAR_SPEED", 0.120)
        self.min_angular_speed = env_float("LEKIWI_MIN_ANGULAR_SPEED", 0.15)
        self.max_angular_speed = env_float("LEKIWI_MAX_ANGULAR_SPEED", 1.20)
        self.command_hold_sec = env_float("LEKIWI_COMMAND_HOLD_SEC", 0.35)

        self.home_seconds = env_float("ACT_HOME_SECONDS", 0.8)
        self.home_steps = env_int("ACT_HOME_STEPS", 40)

        self.last_drive_cmd = (0.0, 0.0, 0.0)
        self.last_drive_cmd_time = 0.0

        self._configure_wheels()
        self.capture_home_from_current_pose()

    # ------------------------------------------------------------------
    # Bus / wheel setup
    # ------------------------------------------------------------------
    def _find_robot_bus(self):
        candidates = []

        for name in ["bus", "motors_bus", "follower_bus"]:
            if hasattr(self.robot, name):
                candidates.append((name, getattr(self.robot, name)))

        try:
            for name, value in vars(self.robot).items():
                candidates.append((name, value))
                if isinstance(value, dict):
                    for k, v in value.items():
                        candidates.append((f"{name}.{k}", v))
        except Exception:
            pass

        for name, obj in candidates:
            if obj is None:
                continue
            if hasattr(obj, "sync_write") and hasattr(obj, "write"):
                self.logger.info(f"[WHEEL] using robot bus object: {name}")
                return obj

        raise RuntimeError(
            "Could not find Feetech bus inside robot object. "
            "Need robot.bus or equivalent object with sync_write/write."
        )

    def _ensure_wheel_motors_registered(self):
        if not hasattr(self.wheel_bus, "motors"):
            raise RuntimeError("wheel_bus has no .motors dict; cannot register wheel motors")

        # SOFollower가 만든 bus는 기본적으로 arm 1~6만 알고 있다.
        # .motors만 update하면 sync_write 내부의 _id_to_model_dict 등이 갱신되지 않아 KeyError: 7이 난다.
        self.wheel_bus.motors.update(WHEEL_MOTORS)

        # wheel calibration도 기존 LeKiwi base driver와 같은 lekiwi.json에서 보강한다.
        calibration_path = os.environ.get("LEKIWI_CALIBRATION_JSON", "/home/lerobot/CIS/config/lekiwi.json")
        wheel_calibration = load_motor_calibration_json(calibration_path)

        for attr in ("calibration", "_calibration"):
            cal = getattr(self.wheel_bus, attr, None)
            if isinstance(cal, dict):
                for name in WHEEL_NAMES:
                    if name in wheel_calibration:
                        cal[name] = wheel_calibration[name]

        for name, motor in WHEEL_MOTORS.items():
            motor_id = int(motor.id)
            model = str(motor.model)

            # 모델 맵: 이번 에러의 직접 원인
            for attr in ("_id_to_model_dict", "id_to_model_dict"):
                d = getattr(self.wheel_bus, attr, None)
                if isinstance(d, dict):
                    d[motor_id] = model

            # 이름↔ID 맵. 설치 버전마다 이름이 다를 수 있어 후보를 모두 보강한다.
            for attr in ("_name_to_id_dict", "name_to_id_dict", "_motor_name_to_id", "motor_name_to_id"):
                d = getattr(self.wheel_bus, attr, None)
                if isinstance(d, dict):
                    d[name] = motor_id

            for attr in ("_id_to_name_dict", "id_to_name_dict", "_id_to_motor_name", "id_to_motor_name"):
                d = getattr(self.wheel_bus, attr, None)
                if isinstance(d, dict):
                    d[motor_id] = name

        # motor name list/set 캐시가 있으면 여기도 보강
        for attr in ("motor_names", "_motor_names"):
            names = getattr(self.wheel_bus, attr, None)
            if isinstance(names, list):
                for name in WHEEL_NAMES:
                    if name not in names:
                        names.append(name)
            elif isinstance(names, set):
                names.update(WHEEL_NAMES)

        self.logger.info(
            "[WHEEL] registered wheel motors into existing SOFollower bus: "
            f"{[(name, WHEEL_MOTORS[name].id) for name in WHEEL_NAMES]}"
        )

    def safe_write(self, register: str, motor_name: str, value, normalize=None, num_retry: int = 5) -> bool:
        last_error = None

        for _ in range(max(1, num_retry)):
            try:
                if normalize is None:
                    self.wheel_bus.write(register, motor_name, value, num_retry=1)
                else:
                    self.wheel_bus.write(register, motor_name, value, normalize=normalize, num_retry=1)
                return True
            except TypeError as e:
                last_error = e
                try:
                    self.wheel_bus.write(register, motor_name, value)
                    return True
                except Exception as e2:
                    last_error = e2
                    time.sleep(0.03)
            except Exception as e:
                last_error = e
                time.sleep(0.03)

        self.logger.warning(f"[WHEEL] write failed: {register} {motor_name}={value}: {last_error}")
        return False

    def _configure_wheels(self):
        with self.bus_lock:
            self._ensure_wheel_motors_registered()

            self.logger.info("[WHEEL] configuring wheel motors 7/8/9 as VELOCITY")

            for name in WHEEL_NAMES:
                self.safe_write("Torque_Enable", name, 0, num_retry=3)
                self.safe_write("Operating_Mode", name, OperatingMode.VELOCITY.value, num_retry=5)
                self.safe_write("Acceleration", name, 254, num_retry=3)

            for name in WHEEL_NAMES:
                self.safe_write("Torque_Enable", name, 1, num_retry=5)
                self.safe_write("Lock", name, 1, num_retry=3)

            stop_ok = self.stop_wheels_locked()
            self.wheels_ready = bool(stop_ok)

            if self.wheels_ready:
                self.logger.info("[WHEEL] ready")
            else:
                self.logger.error("[WHEEL] not ready. Wheel sync_write failed.")

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        raw = int(round(degps * steps_per_deg))
        return int(clamp(raw, -3000, 3000))

    def body_to_wheel_raw(self, vx: float, vy: float, wz_rad: float) -> dict[str, int]:
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

    def stop_wheels_locked(self):
        try:
            self.wheel_bus.sync_write(
                "Goal_Velocity",
                {
                    "base_left_wheel": 0,
                    "base_back_wheel": 0,
                    "base_right_wheel": 0,
                },
            )
            return True
        except Exception as e:
            self.logger.error(f"[WHEEL] stop sync_write failed: {e}")
            return False

    def stop_wheels(self):
        if not self.wheels_ready:
            return

        with self.bus_lock:
            try:
                self.stop_wheels_locked()
            except Exception as e:
                self.logger.error(f"[WHEEL] stop failed: {e}")

        self.last_drive_cmd = (0.0, 0.0, 0.0)

    def send_drive_cmd(self, vx: float, vy: float, wz: float):
        if self.policy_enabled.is_set():
            self.logger.info("[DRIVE] ACT is running. Press 'k' first.")
            return

        if not self.wheels_ready:
            self.logger.error("[DRIVE] wheels are not ready")
            return

        raw = self.body_to_wheel_raw(vx, vy, wz)

        with self.bus_lock:
            try:
                self.wheel_bus.sync_write("Goal_Velocity", raw)
            except Exception as e:
                self.logger.error(f"[DRIVE] Goal_Velocity failed: {e}")
                return

        self.last_drive_cmd = (vx, vy, wz)
        self.last_drive_cmd_time = time.time()

    def drive_tick(self):
        if self.policy_enabled.is_set():
            return

        vx, vy, wz = self.last_drive_cmd
        if abs(vx) + abs(vy) + abs(wz) <= 1e-9:
            return

        if time.time() - self.last_drive_cmd_time > self.command_hold_sec:
            self.stop_wheels()
            return

        self.send_drive_cmd(vx, vy, wz)

    # ------------------------------------------------------------------
    # Home pose
    # ------------------------------------------------------------------
    def _find_obs_value_for_action_key(self, obs: RawObservation, action_key: str):
        if action_key in obs:
            return obs[action_key]

        target = norm_key(action_key)

        for k, v in obs.items():
            if norm_key(k) == target:
                return v

        for k, v in obs.items():
            nk = norm_key(k)
            if nk.endswith(target) or target.endswith(nk):
                return v

        return None

    def _action_from_current_observation(self) -> dict[str, float] | None:
        try:
            with self.bus_lock:
                obs: RawObservation = self.robot.get_observation()
        except Exception as e:
            self.logger.error(f"Failed to read current observation for home pose: {e}")
            return None

        action = {}
        missing = []

        for action_key in self.arm_action_features:
            value = self._find_obs_value_for_action_key(obs, action_key)
            if value is None:
                missing.append(action_key)
                continue
            action[action_key] = to_float(value)

        if missing:
            self.logger.error(
                "Cannot build home action from current observation. "
                f"Missing keys: {missing}. Observation keys: {list(obs.keys())}"
            )
            return None

        return action

    def capture_home_from_current_pose(self):
        action = self._action_from_current_observation()
        if action is None:
            self.logger.error("[HOME] home pose was not captured")
            return

        with self.home_lock:
            self.home_action = action

        self.logger.info("[HOME] memorized current arm pose as start/home pose")

    def move_home(self):
        self.pause_policy()
        self.stop_wheels()

        with self.home_lock:
            if self.home_action is None:
                self.logger.error("[HOME] no home pose. Press 'm' at the start pose.")
                return
            target = dict(self.home_action)

        current = self._action_from_current_observation()
        if current is None:
            self.logger.error("[HOME] cannot read current pose")
            return

        steps = max(1, self.home_steps)
        dt = max(0.01, self.home_seconds / steps)

        self.logger.info(f"[HOME] moving to start pose: seconds={self.home_seconds}, steps={steps}")

        for i in range(1, steps + 1):
            if self.shutdown_event.is_set():
                break

            alpha = i / steps
            cmd = {}

            for key in self.arm_action_features:
                c = float(current[key])
                t = float(target[key])
                cmd[key] = c + (t - c) * alpha

            try:
                with self.bus_lock:
                    self.robot.send_action(cmd)
            except Exception as e:
                self.logger.error(f"[HOME] send_action failed: {e}")
                break

            time.sleep(dt)

        self.logger.info("[HOME] done")

    # ------------------------------------------------------------------
    # ACT pause/resume
    # ------------------------------------------------------------------
    def clear_action_queue(self):
        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_queue_size.clear()
        self.must_go.set()

    def pause_policy(self):
        self.policy_enabled.clear()
        self.clear_action_queue()
        self.stop_wheels()
        self.logger.info("[ACT] paused. Queue cleared. Drive keys enabled.")

    def resume_policy(self):
        self.stop_wheels()
        self.clear_action_queue()
        self.policy_enabled.set()
        self.logger.info("[ACT] started. Drive keys disabled.")

    # ------------------------------------------------------------------
    # ACT action mapping
    # ------------------------------------------------------------------
    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        flat = action_tensor.detach().cpu().flatten()
        n = int(flat.numel())

        if n != self.arm_action_dim:
            self.logger.warning(
                f"[ACT] action dim mismatch: tensor_dim={n}, arm_action_dim={self.arm_action_dim}. "
                "Using min dimension."
            )

        use_n = min(n, self.arm_action_dim)

        return {
            self.arm_action_features[i]: float(flat[i].item())
            for i in range(use_n)
        }

    # ------------------------------------------------------------------
    # Override robot bus accesses with lock
    # ------------------------------------------------------------------
    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        with self.bus_lock:
            return super().control_loop_action(verbose)

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        with self.bus_lock:
            return super().control_loop_observation(task, verbose)

    def receive_actions(self, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions = pickle.loads(actions_chunk.data)  # nosec

                if not self.policy_enabled.is_set():
                    continue

                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                self.must_go.set()

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")
                time.sleep(0.05)
            except Exception as e:
                self.logger.error(f"Error handling received actions: {e}")
                time.sleep(0.05)

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            if self.policy_enabled.is_set():
                if self.actions_available():
                    try:
                        _performed_action = self.control_loop_action(verbose)
                    except Empty:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error performing action: {e}")

                if self._ready_to_send_observation():
                    _captured_observation = self.control_loop_observation(task, verbose)
            else:
                time.sleep(0.02)

            elapsed = time.perf_counter() - control_loop_start
            time.sleep(max(0, self.config.environment_dt - elapsed))

        return _captured_observation, _performed_action

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def print_keyboard_menu(self):
        print()
        print("========== ROBOT CLIENT DRIVE KEYBOARD ==========")
        print("w/s : forward/back")
        print("a/d : strafe left/right")
        print("z/c : rotate left/right")
        print("space : wheel stop")
        print("r/f : speed up/down")
        print("")
        print("p : ACT START  - wheels stop, drive keys disabled")
        print("k : ACT STOP   - queue clear, drive keys enabled")
        print("h : return arm to memorized start pose")
        print("m : memorize current arm pose as start pose")
        print("q : quit client")
        print("? : show menu")
        print("================================================")
        print(f"[SPEED] linear={self.linear_speed:.3f}, angular={self.angular_speed:.3f}")
        print("[STATE] ACT initially: OFF")
        print()

    def keyboard_loop(self):
        self.print_keyboard_menu()

        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            while self.running and not self.keyboard_shutdown.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.03)

                if readable:
                    ch = sys.stdin.read(1)
                    keep_running = self.handle_key(ch)
                    if not keep_running:
                        break

                self.drive_tick()

        except Exception as e:
            self.logger.error(f"Keyboard loop error: {e}")

        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

    def handle_key(self, ch: str) -> bool:
        if ch == "w":
            self.send_drive_cmd(vx=self.linear_speed, vy=0.0, wz=0.0)
        elif ch == "s":
            self.send_drive_cmd(vx=-self.linear_speed, vy=0.0, wz=0.0)
        elif ch == "a":
            self.send_drive_cmd(vx=0.0, vy=self.linear_speed, wz=0.0)
        elif ch == "d":
            self.send_drive_cmd(vx=0.0, vy=-self.linear_speed, wz=0.0)
        elif ch == "z":
            self.send_drive_cmd(vx=0.0, vy=0.0, wz=self.angular_speed)
        elif ch == "c":
            self.send_drive_cmd(vx=0.0, vy=0.0, wz=-self.angular_speed)
        elif ch == " ":
            self.stop_wheels()
            print("[DRIVE] stop")

        elif ch == "r":
            self.linear_speed = min(self.max_linear_speed, self.linear_speed + self.linear_step)
            self.angular_speed = min(self.max_angular_speed, self.angular_speed + self.angular_step)
            print(f"[SPEED] linear={self.linear_speed:.3f}, angular={self.angular_speed:.3f}")
        elif ch == "f":
            self.linear_speed = max(self.min_linear_speed, self.linear_speed - self.linear_step)
            self.angular_speed = max(self.min_angular_speed, self.angular_speed - self.angular_step)
            print(f"[SPEED] linear={self.linear_speed:.3f}, angular={self.angular_speed:.3f}")

        elif ch == "p":
            self.resume_policy()
        elif ch == "k":
            self.pause_policy()
        elif ch == "h":
            self.move_home()
        elif ch == "m":
            self.capture_home_from_current_pose()
        elif ch == "q":
            self.logger.info("[KEYBOARD] quit requested")
            self.shutdown_event.set()
            return False
        elif ch == "?":
            self.print_keyboard_menu()
        else:
            print(f"[KEYBOARD] unknown key: {repr(ch)}")

        return True

    def stop(self):
        try:
            self.pause_policy()
            self.stop_wheels()
        except Exception:
            pass
        super().stop()


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = DriveKeyboardRobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        keyboard_thread = threading.Thread(target=client.keyboard_loop, daemon=True)

        action_receiver_thread.start()
        keyboard_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.keyboard_shutdown.set()
            client.stop()

            action_receiver_thread.join(timeout=2.0)
            keyboard_thread.join(timeout=2.0)

            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
