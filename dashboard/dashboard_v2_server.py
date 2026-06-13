#!/usr/bin/env python3
"""Dashboard v2: virtual 2D map pose + Arduino sensor bridge."""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import rclpy
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from geometry_msgs.msg import Twist
from pydantic import BaseModel
from rclpy.node import Node
from std_msgs.msg import Bool

from iot.arduino_read import find_serial_port, open_serial, parse_sensor_json


APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "dashboard_v2.html"
DEFAULT_MAP_YAML = Path(
    os.environ.get(
        "DASHBOARD_V2_MAP_YAML",
        "/home/lerobot/CIS/nav_maps/generated/lekiwi_map_v8.yaml",
    )
)
MAP_CACHE_DIR = APP_DIR / ".dashboard_v2_cache"

MAX_LINEAR_X = 0.10
MAX_LINEAR_Y = 0.10
MAX_ANGULAR_Z = 0.5236
SERVER_INPUT_TIMEOUT_SEC = 0.7

MAP_WIDTH_M = 52.0
MAP_HEIGHT_M = 24.0


def parse_simple_yaml(path: Path) -> dict:
    data = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            data[key.strip()] = [
                float(item.strip())
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        else:
            data[key.strip()] = value
    return data


def load_map_metadata() -> dict:
    yaml_path = DEFAULT_MAP_YAML
    data = parse_simple_yaml(yaml_path)
    image_path = Path(str(data["image"]))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    from PIL import Image

    with Image.open(image_path) as im:
        width_px, height_px = im.size

    resolution = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    return {
        "yaml_path": str(yaml_path),
        "image_path": str(image_path),
        "resolution": resolution,
        "origin": origin,
        "width_px": width_px,
        "height_px": height_px,
        "width_m": width_px * resolution,
        "height_m": height_px * resolution,
        "image_url": "/map_image.png",
    }


MAP_META = load_map_metadata()
MAP_WIDTH_M = float(MAP_META["width_m"])
MAP_HEIGHT_M = float(MAP_META["height_m"])


def ensure_map_png() -> Path:
    from PIL import Image

    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = MAP_CACHE_DIR / "dashboard_v2_map.png"
    src = Path(str(MAP_META["image_path"]))
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    with Image.open(src) as im:
        im.convert("RGB").save(out)
    return out


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_twist(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> Twist:
    msg = Twist()
    msg.linear.x = clamp(float(x), -MAX_LINEAR_X, MAX_LINEAR_X)
    msg.linear.y = clamp(float(y), -MAX_LINEAR_Y, MAX_LINEAR_Y)
    msg.angular.z = clamp(float(yaw), -MAX_ANGULAR_Z, MAX_ANGULAR_Z)
    return msg


def yaw_wrap(yaw: float) -> float:
    while yaw > math.pi:
        yaw -= math.tau
    while yaw < -math.pi:
        yaw += math.tau
    return yaw


class DriveRequest(BaseModel):
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class DashboardV2State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pose = {
            "x": MAP_WIDTH_M * 0.10,
            "y": MAP_HEIGHT_M * 0.50,
            "yaw": 0.0,
            "map_width_m": MAP_WIDTH_M,
            "map_height_m": MAP_HEIGHT_M,
            "updated_at": time.time(),
        }
        self.cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.last_cmd_time = 0.0
        self.sensor: Optional[dict] = None
        self.sensor_source = "fallback"

    def set_cmd(self, x: float, y: float, yaw: float) -> None:
        with self.lock:
            self.cmd = {
                "x": clamp(x, -MAX_LINEAR_X, MAX_LINEAR_X),
                "y": clamp(y, -MAX_LINEAR_Y, MAX_LINEAR_Y),
                "yaw": clamp(yaw, -MAX_ANGULAR_Z, MAX_ANGULAR_Z),
            }
            self.last_cmd_time = time.monotonic()

    def stop_cmd(self) -> None:
        self.set_cmd(0.0, 0.0, 0.0)

    def integrate_pose(self, dt: float) -> None:
        with self.lock:
            if self.last_cmd_time <= 0.0:
                return
            if time.monotonic() - self.last_cmd_time > SERVER_INPUT_TIMEOUT_SEC:
                self.cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0}

            yaw = float(self.pose["yaw"])
            vx = float(self.cmd["x"])
            vy = float(self.cmd["y"])
            wz = float(self.cmd["yaw"])

            # Body-frame LeKiwi command -> virtual map-frame displacement.
            dx = (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            dy = (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt

            self.pose["x"] = clamp(float(self.pose["x"]) + dx, 0.0, MAP_WIDTH_M)
            self.pose["y"] = clamp(float(self.pose["y"]) + dy, 0.0, MAP_HEIGHT_M)
            self.pose["yaw"] = yaw_wrap(yaw + wz * dt)
            self.pose["updated_at"] = time.time()

    def set_sensor(self, sensor: dict, source: str) -> None:
        with self.lock:
            self.sensor = sensor
            self.sensor_source = source

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "ok": True,
                "pose": dict(self.pose),
                "cmd": dict(self.cmd),
                "sensor": dict(self.sensor) if self.sensor else None,
                "sensor_source": self.sensor_source,
                "limits": {
                    "max_linear_x": MAX_LINEAR_X,
                    "max_linear_y": MAX_LINEAR_Y,
                    "max_angular_z": MAX_ANGULAR_Z,
                },
            }


class DashboardV2RosPublisher(Node):
    def __init__(self, state: DashboardV2State):
        super().__init__("dashboard_v2_cmd_vel_publisher")
        self.state = state
        self.cmd_pub = self.create_publisher(Twist, "/dashboard/cmd_vel", 10)
        self.estop_pub = self.create_publisher(Bool, "/emergency_stop", 10)

    def publish_drive(self, x: float, y: float, yaw: float) -> None:
        msg = make_twist(x, y, yaw)
        self.state.set_cmd(msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_pub.publish(msg)

    def publish_stop(self) -> None:
        self.state.stop_cmd()
        try:
            self.cmd_pub.publish(make_twist())
        except Exception:
            pass

    def publish_estop(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        try:
            self.estop_pub.publish(msg)
        except Exception:
            pass
        if active:
            self.publish_stop()


state = DashboardV2State()
app = FastAPI(title="CIS Dashboard V2")
ros_node: Optional[DashboardV2RosPublisher] = None
pose_task: Optional[asyncio.Task] = None
arduino_thread: Optional[threading.Thread] = None
arduino_stop = threading.Event()


def fallback_sensor() -> dict:
    t = time.time()
    return {
        "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_c": 23.0 + math.sin(t * 0.07) * 1.2,
        "humidity_percent": 62.0 + math.cos(t * 0.05) * 5.0,
        "co2_ppm_raw": 1450.0 + math.sin(t * 0.04) * 120.0,
        "co2_ppm_filtered": 1450.0 + math.sin(t * 0.04) * 120.0,
        "nh3_raw": 35.0 + math.cos(t * 0.06) * 8.0,
        "nh3_voltage": None,
        "overall_status": "NORMAL",
        "co2_status": "NORMAL",
        "temp_hum_status": "NORMAL",
        "nh3_status": "NORMAL",
    }


def arduino_reader_loop() -> None:
    while not arduino_stop.is_set():
        port = find_serial_port()
        if port is None:
            state.set_sensor(fallback_sensor(), "fallback")
            arduino_stop.wait(2.0)
            continue

        try:
            ser = open_serial(port, 9600, 2.0, 2.0)
        except Exception:
            state.set_sensor(fallback_sensor(), "fallback")
            arduino_stop.wait(2.0)
            continue

        try:
            while not arduino_stop.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                frame = parse_sensor_json(line)
                if frame is not None:
                    state.set_sensor(asdict(frame), port)
        except Exception:
            state.set_sensor(fallback_sensor(), "fallback")
        finally:
            try:
                ser.close()
            except Exception:
                pass


async def pose_loop() -> None:
    last = time.monotonic()
    while True:
        await asyncio.sleep(0.05)
        now = time.monotonic()
        state.integrate_pose(now - last)
        last = now


@app.on_event("startup")
async def startup_event() -> None:
    global ros_node, pose_task, arduino_thread
    if not rclpy.ok():
        rclpy.init(args=None)
    ros_node = DashboardV2RosPublisher(state)
    pose_task = asyncio.create_task(pose_loop())
    arduino_stop.clear()
    arduino_thread = threading.Thread(target=arduino_reader_loop, daemon=True)
    arduino_thread.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global ros_node, pose_task
    if pose_task is not None:
        pose_task.cancel()
        pose_task = None
    arduino_stop.set()
    if ros_node is not None:
        ros_node.publish_stop()
        ros_node.destroy_node()
        ros_node = None
    if rclpy.ok():
        rclpy.shutdown()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not HTML_PATH.exists():
        return HTMLResponse("<h1>dashboard_v2.html not found</h1>", status_code=404)
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    snap = state.snapshot()
    snap["ros_ready"] = ros_node is not None
    snap["html"] = str(HTML_PATH)
    snap["cmd_topic"] = "/dashboard/cmd_vel"
    snap["map"] = MAP_META
    return snap


@app.get("/api/state")
async def api_state() -> dict:
    snap = state.snapshot()
    snap["map"] = MAP_META
    return snap


@app.get("/api/map")
async def api_map() -> dict:
    return {"ok": True, "map": MAP_META}


@app.get("/map_image.png")
async def map_image() -> FileResponse:
    return FileResponse(ensure_map_png(), media_type="image/png")


@app.post("/api/drive")
async def drive(req: DriveRequest):
    if ros_node is None:
        return JSONResponse({"ok": False, "error": "ROS2 publisher is not ready"}, status_code=503)
    ros_node.publish_drive(req.x, req.y, req.yaw)
    return state.snapshot()


@app.post("/api/emergency_stop")
async def emergency_stop() -> dict:
    if ros_node is not None:
        ros_node.publish_estop(True)
    return {"ok": True, "emergency_stop": True}


@app.post("/api/emergency_stop/release")
async def emergency_stop_release() -> dict:
    if ros_node is not None:
        ros_node.publish_estop(False)
    return {"ok": True, "emergency_stop": False}


@app.websocket("/ws/drive")
async def drive_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if ros_node is not None:
                ros_node.publish_drive(
                    float(data.get("x", 0.0)),
                    float(data.get("y", 0.0)),
                    float(data.get("yaw", 0.0)),
                )
            await websocket.send_json(state.snapshot())
    except WebSocketDisconnect:
        if ros_node is not None:
            ros_node.publish_stop()


@app.websocket("/ws/state")
async def state_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            snap = state.snapshot()
            snap["map"] = MAP_META
            await websocket.send_json(snap)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        reload=False,
    )
