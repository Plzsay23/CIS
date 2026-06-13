#!/usr/bin/env python3
"""Robot dashboard: show the physical LeKiwi robot on the map using /odom."""

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pydantic import BaseModel
from rclpy.node import Node
from std_msgs.msg import Bool, String

try:
    from iot.arduino_read import find_serial_port, open_serial, parse_sensor_json
except Exception:  # Allows the dashboard to run even when the iot package is absent.
    find_serial_port = None
    open_serial = None
    parse_sensor_json = None


APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "robot_dashboard.html"

DEFAULT_MAP_YAML = Path(
    os.environ.get(
        "DASHBOARD_MAP_YAML",
        "/home/lerobot/CIS/nav_maps/generated/lekiwi_poultry_house.yaml",
    )
)
MAP_CACHE_DIR = APP_DIR / ".dashboard_cache"

POSE_TOPIC = os.environ.get("DASHBOARD_POSE_TOPIC", "/odom")
DASHBOARD_START_X_ENV = os.environ.get("DASHBOARD_START_X")
DASHBOARD_START_Y_ENV = os.environ.get("DASHBOARD_START_Y")
ODOM_X_SIGN = float(os.environ.get("DASHBOARD_ODOM_X_SIGN", "1.0"))
ODOM_Y_SIGN = float(os.environ.get("DASHBOARD_ODOM_Y_SIGN", "1.0"))
YAW_SIGN = float(os.environ.get("DASHBOARD_ODOM_YAW_SIGN", "1.0"))

MAX_LINEAR_X = float(os.environ.get("DASHBOARD_MAX_LINEAR_X", "0.10"))
MAX_LINEAR_Y = float(os.environ.get("DASHBOARD_MAX_LINEAR_Y", "0.10"))
MAX_ANGULAR_Z = float(os.environ.get("DASHBOARD_MAX_ANGULAR_Z", "0.5236"))
SERVER_INPUT_TIMEOUT_SEC = float(os.environ.get("DASHBOARD_INPUT_TIMEOUT_SEC", "0.7"))

FALLBACK_MAP_WIDTH_M = 52.0
FALLBACK_MAP_HEIGHT_M = 24.0

ALLOWED_ARM_COMMANDS = {
    "gripper_open",
    "gripper_close",
    "arm_home",
    "motor_1_up",
    "motor_1_down",
    "motor_1_stop",
    "motor_2_up",
    "motor_2_down",
    "motor_2_stop",
    "motor_3_up",
    "motor_3_down",
    "motor_3_stop",
    "motor_4_up",
    "motor_4_down",
    "motor_4_stop",
    "motor_5_up",
    "motor_5_down",
    "motor_5_stop",
    "motor_6_up",
    "motor_6_down",
    "motor_6_stop",
}


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
    yaml_path = DEFAULT_MAP_YAML.expanduser()

    if not yaml_path.exists():
        return {
            "ok": False,
            "yaml_path": str(yaml_path),
            "image_path": None,
            "resolution": 0.05,
            "origin": [0.0, 0.0, 0.0],
            "width_px": int(FALLBACK_MAP_WIDTH_M / 0.05),
            "height_px": int(FALLBACK_MAP_HEIGHT_M / 0.05),
            "width_m": FALLBACK_MAP_WIDTH_M,
            "height_m": FALLBACK_MAP_HEIGHT_M,
            "image_url": None,
            "warning": "map yaml not found; using fallback map size",
        }

    data = parse_simple_yaml(yaml_path)
    image_path = Path(str(data.get("image", "")))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    resolution = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])

    width_px = int(FALLBACK_MAP_WIDTH_M / resolution)
    height_px = int(FALLBACK_MAP_HEIGHT_M / resolution)
    image_url = None
    warning = None

    if image_path.exists():
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                width_px, height_px = im.size
            image_url = "/map_image.png"
        except Exception as exc:
            warning = f"failed to read map image: {exc}"
    else:
        warning = f"map image not found: {image_path}"

    return {
        "ok": True,
        "yaml_path": str(yaml_path),
        "image_path": str(image_path),
        "resolution": resolution,
        "origin": origin,
        "width_px": width_px,
        "height_px": height_px,
        "width_m": width_px * resolution,
        "height_m": height_px * resolution,
        "image_url": image_url,
        "warning": warning,
    }


MAP_META = load_map_metadata()
MAP_WIDTH_M = float(MAP_META["width_m"])
MAP_HEIGHT_M = float(MAP_META["height_m"])
DASHBOARD_START_X = float(DASHBOARD_START_X_ENV) if DASHBOARD_START_X_ENV is not None else MAP_WIDTH_M * 0.10
DASHBOARD_START_Y = float(DASHBOARD_START_Y_ENV) if DASHBOARD_START_Y_ENV is not None else MAP_HEIGHT_M * 0.50


def ensure_map_png() -> Path:
    image_path_value = MAP_META.get("image_path")
    if not image_path_value:
        raise FileNotFoundError("map image is not configured")

    src = Path(str(image_path_value))
    if not src.exists():
        raise FileNotFoundError(str(src))

    from PIL import Image

    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = MAP_CACHE_DIR / "dashboard_map.png"
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
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = clamp(float(yaw), -MAX_ANGULAR_Z, MAX_ANGULAR_Z)
    return msg


def yaw_wrap(yaw: float) -> float:
    return math.atan2(math.sin(yaw), math.cos(yaw))


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # Standard quaternion -> yaw conversion.
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class DriveRequest(BaseModel):
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class ArmCommandRequest(BaseModel):
    command: str


class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pose = {
            "x": clamp(DASHBOARD_START_X, 0.0, MAP_WIDTH_M),
            "y": clamp(DASHBOARD_START_Y, 0.0, MAP_HEIGHT_M),
            "yaw": 0.0,
            "map_width_m": MAP_WIDTH_M,
            "map_height_m": MAP_HEIGHT_M,
            "updated_at": time.time(),
        }
        self.cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.last_cmd_time = 0.0
        self.sensor: Optional[dict] = None
        self.sensor_source = "fallback"
        self.odom_origin: Optional[dict] = None
        self.pose_received = False
        self.last_pose_time = 0.0
        self.pose_source = POSE_TOPIC

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

    def set_odom_pose(self, odom_x: float, odom_y: float, odom_yaw: float) -> None:
        now = time.time()
        with self.lock:
            if self.odom_origin is None:
                self.odom_origin = {
                    "x": float(odom_x),
                    "y": float(odom_y),
                    "yaw": float(odom_yaw),
                    "received_at": now,
                    "dashboard_start_x": clamp(DASHBOARD_START_X, 0.0, MAP_WIDTH_M),
                    "dashboard_start_y": clamp(DASHBOARD_START_Y, 0.0, MAP_HEIGHT_M),
                }

            dx = (float(odom_x) - float(self.odom_origin["x"])) * ODOM_X_SIGN
            dy = (float(odom_y) - float(self.odom_origin["y"])) * ODOM_Y_SIGN

            self.pose["x"] = clamp(float(self.odom_origin["dashboard_start_x"]) + dx, 0.0, MAP_WIDTH_M)
            self.pose["y"] = clamp(float(self.odom_origin["dashboard_start_y"]) + dy, 0.0, MAP_HEIGHT_M)
            self.pose["yaw"] = yaw_wrap(float(odom_yaw) * YAW_SIGN)
            self.pose["updated_at"] = now
            self.pose_received = True
            self.last_pose_time = now

    def set_sensor(self, sensor: dict, source: str) -> None:
        with self.lock:
            self.sensor = sensor
            self.sensor_source = source

    def snapshot(self) -> dict:
        with self.lock:
            pose_age = None
            if self.last_pose_time > 0.0:
                pose_age = max(0.0, time.time() - self.last_pose_time)

            return {
                "ok": True,
                "pose_mode": "robot",
                "pose_source": self.pose_source,
                "pose_age_sec": pose_age,
                "pose_received": self.pose_received,
                "odom_origin": dict(self.odom_origin) if self.odom_origin else None,
                "pose": dict(self.pose),
                "cmd": dict(self.cmd),
                "sensor": dict(self.sensor) if self.sensor else None,
                "sensor_source": self.sensor_source,
                "limits": {
                    "max_linear_x": MAX_LINEAR_X,
                    "max_linear_y": MAX_LINEAR_Y,
                    "max_angular_z": MAX_ANGULAR_Z,
                    "server_input_timeout_sec": SERVER_INPUT_TIMEOUT_SEC,
                },
                "odom_transform": {
                    "start_x": DASHBOARD_START_X,
                    "start_y": DASHBOARD_START_Y,
                    "x_sign": ODOM_X_SIGN,
                    "y_sign": ODOM_Y_SIGN,
                    "yaw_sign": YAW_SIGN,
                },
            }


class DashboardRosBridge(Node):
    def __init__(self, state: DashboardState):
        super().__init__("robot_dashboard_ros_bridge")
        self.state = state
        self.cmd_pub = self.create_publisher(Twist, "/dashboard/cmd_vel", 10)
        self.estop_pub = self.create_publisher(Bool, "/emergency_stop", 10)
        self.arm_cmd_pub = self.create_publisher(String, "/dashboard/arm_cmd", 10)
        self.odom_sub = self.create_subscription(Odometry, POSE_TOPIC, self.on_odom, 20)
        self.get_logger().info(f"robot dashboard started: pose_topic={POSE_TOPIC}, cmd_topic=/dashboard/cmd_vel")

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        self.state.set_odom_pose(float(p.x), float(p.y), yaw)

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

    def publish_arm_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.arm_cmd_pub.publish(msg)


state = DashboardState()
app = FastAPI(title="CIS Robot Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ros_node: Optional[DashboardRosBridge] = None
watchdog_task: Optional[asyncio.Task] = None
arduino_thread: Optional[threading.Thread] = None
arduino_stop = threading.Event()
ros_spin_thread: Optional[threading.Thread] = None
ros_spin_stop = threading.Event()


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
    if find_serial_port is None or open_serial is None or parse_sensor_json is None:
        while not arduino_stop.is_set():
            state.set_sensor(fallback_sensor(), "fallback_no_iot_module")
            arduino_stop.wait(2.0)
        return

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


def ros_spin_loop() -> None:
    global ros_node
    while not ros_spin_stop.is_set():
        if ros_node is None or not rclpy.ok():
            time.sleep(0.05)
            continue
        try:
            rclpy.spin_once(ros_node, timeout_sec=0.1)
        except Exception:
            time.sleep(0.05)


async def dashboard_watchdog_loop() -> None:
    global ros_node
    while True:
        await asyncio.sleep(0.05)
        if ros_node is None:
            continue
        if state.last_cmd_time <= 0.0:
            continue
        if time.monotonic() - state.last_cmd_time > SERVER_INPUT_TIMEOUT_SEC:
            ros_node.publish_stop()
            state.last_cmd_time = 0.0


@app.on_event("startup")
async def startup_event() -> None:
    global ros_node, watchdog_task, arduino_thread, ros_spin_thread
    if not rclpy.ok():
        rclpy.init(args=None)

    ros_node = DashboardRosBridge(state)

    ros_spin_stop.clear()
    ros_spin_thread = threading.Thread(target=ros_spin_loop, daemon=True)
    ros_spin_thread.start()

    watchdog_task = asyncio.create_task(dashboard_watchdog_loop())

    arduino_stop.clear()
    arduino_thread = threading.Thread(target=arduino_reader_loop, daemon=True)
    arduino_thread.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global ros_node, watchdog_task, ros_spin_thread

    if watchdog_task is not None:
        watchdog_task.cancel()
        watchdog_task = None

    arduino_stop.set()
    ros_spin_stop.set()

    if ros_node is not None:
        try:
            ros_node.publish_stop()
        except Exception:
            pass
        try:
            ros_node.destroy_node()
        except Exception:
            pass
        ros_node = None

    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass

    ros_spin_thread = None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not HTML_PATH.exists():
        return HTMLResponse("<h1>robot_dashboard.html not found</h1>", status_code=404)
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    snap = state.snapshot()
    snap["ros_ready"] = ros_node is not None
    snap["html"] = str(HTML_PATH)
    snap["cmd_topic"] = "/dashboard/cmd_vel"
    snap["estop_topic"] = "/emergency_stop"
    snap["arm_cmd_topic"] = "/dashboard/arm_cmd"
    snap["allowed_arm_commands"] = sorted(ALLOWED_ARM_COMMANDS)
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
    try:
        return FileResponse(ensure_map_png(), media_type="image/png")
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)


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


@app.post("/api/arm_command")
async def arm_command(req: ArmCommandRequest):
    if ros_node is None:
        return JSONResponse({"ok": False, "error": "ROS2 publisher is not ready"}, status_code=503)

    command = req.command.strip()
    if command not in ALLOWED_ARM_COMMANDS:
        return JSONResponse(
            {
                "ok": False,
                "error": "unsupported arm command",
                "command": command,
                "allowed": sorted(ALLOWED_ARM_COMMANDS),
            },
            status_code=400,
        )

    ros_node.publish_arm_command(command)
    return {"ok": True, "topic": "/dashboard/arm_cmd", "command": command}


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
    except Exception:
        if ros_node is not None:
            ros_node.publish_stop()
        raise


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

    port = int(os.environ.get("DASHBOARD_PORT", "8082"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
    )
