#!/usr/bin/env python3

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pprint import pformat
from urllib.parse import urlparse

import draccus

from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import visualize_action_queue_size
from .robot_client_drive_keyboard import DriveKeyboardRobotClient


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def make_api_handler(client: "DriveDashboardClient"):
    class DashboardApiHandler(BaseHTTPRequestHandler):
        server_version = "LeKiwiActDashboardAPI/1.0"

        def log_message(self, fmt, *args):
            client.logger.info("[API] " + fmt % args)

        def _headers(self, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")

        def _send_json(self, data, status=200):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._headers(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_OPTIONS(self):
            self._headers(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path

            if path == "/api/status":
                self._send_json(client.api_status())
                return

            if path == "/":
                self._send_json({
                    "ok": True,
                    "service": "LeKiwi ACT dashboard API",
                    "endpoints": [
                        "/api/status",
                        "/api/drive",
                        "/api/arm_command",
                        "/api/act/start",
                        "/api/act/stop",
                        "/api/emergency_stop",
                        "/api/emergency_stop/release",
                    ],
                })
                return

            self._send_json({"ok": False, "error": "not_found", "path": path}, status=404)

        def do_POST(self):
            path = urlparse(self.path).path

            try:
                payload = self._read_json()
            except Exception as e:
                self._send_json({"ok": False, "error": f"invalid_json: {e}"}, status=400)
                return

            try:
                if path == "/api/drive":
                    x = float(payload.get("x", 0.0))
                    y = float(payload.get("y", 0.0))
                    yaw = float(payload.get("yaw", 0.0))
                    client.api_drive(x, y, yaw)
                    self._send_json({"ok": True, "type": "drive", "x": x, "y": y, "yaw": yaw})
                    return

                if path == "/api/arm_command":
                    command = str(payload.get("command", "")).strip()
                    result = client.api_arm_command(command)
                    self._send_json(result)
                    return

                if path == "/api/act/start":
                    result = client.api_arm_command("act_start")
                    self._send_json(result)
                    return

                if path == "/api/act/stop":
                    result = client.api_arm_command("act_stop")
                    self._send_json(result)
                    return

                if path == "/api/emergency_stop":
                    client.api_set_estop(True)
                    self._send_json({"ok": True, "estop": True})
                    return

                if path == "/api/emergency_stop/release":
                    client.api_set_estop(False)
                    self._send_json({"ok": True, "estop": False})
                    return

                self._send_json({"ok": False, "error": "not_found", "path": path}, status=404)

            except Exception as e:
                client.logger.exception(f"[API] request failed: path={path}")
                self._send_json({"ok": False, "error": str(e)}, status=500)

    return DashboardApiHandler


class DriveDashboardClient(DriveKeyboardRobotClient):
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)

        self.api_estop = threading.Event()
        self.api_server = None
        self.api_thread = None

        self.api_host = os.environ.get("LEKIWI_API_HOST", "0.0.0.0")
        self.api_port = int(os.environ.get("LEKIWI_API_PORT", "8082"))

        self.start_api_server()

    def start_api_server(self):
        if self.api_server is not None:
            return

        handler = make_api_handler(self)
        self.api_server = ThreadingHTTPServer((self.api_host, self.api_port), handler)
        self.api_thread = threading.Thread(
            target=self.api_server.serve_forever,
            daemon=True,
            name="lekiwi_dashboard_api",
        )
        self.api_thread.start()

        self.logger.info(f"[API] dashboard control API listening on http://{self.api_host}:{self.api_port}")

    def stop_api_server(self):
        if self.api_server is None:
            return

        try:
            self.api_server.shutdown()
            self.api_server.server_close()
        except Exception:
            pass

        self.api_server = None
        self.api_thread = None
        self.logger.info("[API] stopped")

    def api_status(self):
        return {
            "ok": True,
            "act_enabled": bool(self.policy_enabled.is_set()),
            "estop": bool(self.api_estop.is_set()),
            "wheels_ready": bool(self.wheels_ready),
            "linear_speed": float(self.linear_speed),
            "angular_speed": float(self.angular_speed),
            "last_drive_cmd": {
                "x": float(self.last_drive_cmd[0]),
                "y": float(self.last_drive_cmd[1]),
                "yaw": float(self.last_drive_cmd[2]),
            },
            "server_time": time.time(),
        }

    def api_drive(self, x: float, y: float, yaw: float):
        if self.api_estop.is_set():
            self.stop_wheels()
            self.logger.info("[API] drive ignored: E-STOP active")
            return

        self.send_drive_cmd(float(x), float(y), float(yaw))

    def api_set_estop(self, active: bool):
        if active:
            self.api_estop.set()
            self.pause_policy()
            self.stop_wheels()
            self.logger.warning("[API] E-STOP active")
        else:
            self.api_estop.clear()
            self.stop_wheels()
            self.logger.info("[API] E-STOP released")

    def api_arm_command(self, command: str):
        command = command.strip()

        if command in ("act_start", "start_act", "policy_start", "p"):
            if self.api_estop.is_set():
                return {"ok": False, "command": command, "error": "E-STOP active"}
            self.resume_policy()
            return {"ok": True, "command": command, "act_enabled": True}

        if command in ("act_stop", "stop_act", "policy_stop", "k"):
            self.pause_policy()
            return {"ok": True, "command": command, "act_enabled": False}

        if command in ("arm_home", "home", "h"):
            threading.Thread(target=self.move_home, daemon=True).start()
            return {"ok": True, "command": command, "started": True}

        if command in ("arm_memorize", "memorize_home", "save_home", "m"):
            self.capture_home_from_current_pose()
            return {"ok": True, "command": command}

        if command in ("wheel_stop", "drive_stop", "stop"):
            self.stop_wheels()
            return {"ok": True, "command": command}

        return {"ok": False, "command": command, "error": "unknown command"}

    def send_drive_cmd(self, vx: float, vy: float, wz: float):
        if self.api_estop.is_set():
            self.stop_wheels()
            return
        return super().send_drive_cmd(vx, vy, wz)

    def resume_policy(self):
        if self.api_estop.is_set():
            self.logger.warning("[ACT] start blocked: E-STOP active")
            return
        return super().resume_policy()

    def stop(self):
        try:
            self.stop_api_server()
        except Exception:
            pass
        return super().stop()


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    client = DriveDashboardClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        keyboard_thread = None
        if env_bool("LEKIWI_ENABLE_LOCAL_KEYBOARD", True):
            keyboard_thread = threading.Thread(target=client.keyboard_loop, daemon=True)
            keyboard_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.keyboard_shutdown.set()
            client.stop()

            action_receiver_thread.join(timeout=2.0)
            if keyboard_thread is not None:
                keyboard_thread.join(timeout=2.0)

            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
