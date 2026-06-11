#!/usr/bin/env python3

import logging
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
import torch

from lerobot.transport import services_pb2  # type: ignore

from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    Observation,
    RawObservation,
    TimedObservation,
    visualize_action_queue_size,
)
from .robot_client import RobotClient


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


class KeyboardRobotClient(RobotClient):
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)

        self.policy_enabled = threading.Event()
        self.keyboard_shutdown = threading.Event()

        start_enabled = os.environ.get("ACT_START_ENABLED", "0").strip().lower()
        if start_enabled in ("1", "true", "yes", "on"):
            self.policy_enabled.set()

        self.home_action: dict[str, float] | None = None
        self.home_lock = threading.Lock()

        self.home_seconds = float(os.environ.get("ACT_HOME_SECONDS", "0.8"))
        self.home_steps = int(os.environ.get("ACT_HOME_STEPS", "40"))

        self.capture_home_from_current_pose()

    def print_keyboard_menu(self):
        print()
        print("========== ROBOT CLIENT KEYBOARD ==========")
        print("p : ACT START / resume policy actions")
        print("k : ACT STOP  / pause policy actions")
        print("h : return arm to memorized start pose")
        print("m : memorize current arm pose as start pose")
        print("q : quit client")
        print("? : show this menu")
        print("===========================================")
        print("[STATE] ACT initially:", "ON" if self.policy_enabled.is_set() else "OFF")
        print()

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
            obs: RawObservation = self.robot.get_observation()
        except Exception as e:
            self.logger.error(f"Failed to read current observation for home pose: {e}")
            return None

        action = {}
        missing = []

        for action_key in self.robot.action_features:
            value = self._find_obs_value_for_action_key(obs, action_key)
            if value is None:
                missing.append(action_key)
                continue
            action[action_key] = to_float(value)

        if missing:
            self.logger.error(
                "Cannot build home action from current observation. "
                f"Missing keys: {missing}. "
                f"Observation keys: {list(obs.keys())}"
            )
            return None

        return action

    def capture_home_from_current_pose(self):
        action = self._action_from_current_observation()
        if action is None:
            self.logger.error("Home pose was not captured. Key 'h' will not work yet.")
            return

        with self.home_lock:
            self.home_action = action

        self.logger.info("Memorized current arm pose as start/home pose")
        self.logger.info(f"Home action keys: {list(action.keys())}")

    def clear_action_queue(self):
        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_queue_size.clear()

        self.must_go.set()

    def pause_policy(self):
        self.policy_enabled.clear()
        self.clear_action_queue()
        self.logger.info("[KEYBOARD] ACT paused. Action queue cleared.")

    def resume_policy(self):
        self.clear_action_queue()
        self.policy_enabled.set()
        self.logger.info("[KEYBOARD] ACT resumed.")

    def move_home(self):
        self.pause_policy()

        with self.home_lock:
            if self.home_action is None:
                self.logger.error("No home pose memorized. Press 'm' at the correct pose first.")
                return
            target = dict(self.home_action)

        current = self._action_from_current_observation()
        if current is None:
            self.logger.error("Cannot read current pose. Home move aborted.")
            return

        steps = max(1, self.home_steps)
        dt = max(0.01, self.home_seconds / steps)

        self.logger.info(
            f"[KEYBOARD] Moving to memorized home pose: seconds={self.home_seconds}, steps={steps}"
        )

        for i in range(1, steps + 1):
            if self.shutdown_event.is_set():
                break

            alpha = i / steps
            cmd = {}

            for key in self.robot.action_features:
                c = float(current[key])
                t = float(target[key])
                cmd[key] = c + (t - c) * alpha

            try:
                self.robot.send_action(cmd)
            except Exception as e:
                self.logger.error(f"Failed during home motion: {e}")
                break

            time.sleep(dt)

        self.logger.info("[KEYBOARD] Home motion done.")

    def keyboard_loop(self):
        self.print_keyboard_menu()

        old_settings = None
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            while self.running and not self.keyboard_shutdown.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue

                ch = sys.stdin.read(1)

                if ch == "p":
                    self.resume_policy()
                elif ch == "k":
                    self.pause_policy()
                elif ch == "h":
                    self.move_home()
                elif ch == "m":
                    self.capture_home_from_current_pose()
                elif ch == "q":
                    self.logger.info("[KEYBOARD] Quit requested.")
                    self.shutdown_event.set()
                    break
                elif ch == "?":
                    self.print_keyboard_menu()
                else:
                    print(f"[KEYBOARD] unknown key: {repr(ch)}")

        except Exception as e:
            self.logger.error(f"Keyboard loop error: {e}")

        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

    def receive_actions(self, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")
                time.sleep(0.05)
                continue

            try:
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

            except Exception as e:
                self.logger.error(f"Error handling received actions: {e}")

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


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = KeyboardRobotClient(cfg)

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
