#!/usr/bin/env python3
"""Bridge Arduino environmental sensor frames to a Tomcat-compatible WebSocket."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any


DEFAULT_LAPTOP_IP = os.environ.get("CIS_LAPTOP_IP", "100.90.245.102")
DEFAULT_TOMCAT_PORT = os.environ.get("CIS_TOMCAT_PORT", "8081")
DEFAULT_PROJECT_NAME = os.environ.get("CIS_TOMCAT_PROJECT", "CIS_Server")
DEFAULT_WS_URL = os.environ.get(
    "CIS_TOMCAT_WS_URL",
    f"ws://{DEFAULT_LAPTOP_IP}:{DEFAULT_TOMCAT_PORT}/{DEFAULT_PROJECT_NAME}/eggstream",
)
DEFAULT_SERIAL_PORT = os.environ.get("CIS_ARDUINO_PORT", "/dev/arduino")
DEFAULT_BAUD_RATE = int(os.environ.get("CIS_ARDUINO_BAUD", "9600"))

SAMPLE_SENSOR_LINE = json.dumps(
    {
        "co2_ppm": 812,
        "temperature_c": 26.4,
        "humidity_percent": 61.2,
        "nh3_raw": 42,
        "nh3_voltage": 0.21,
    }
)


def parse_number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def payload_from_sensor_dict(data: dict[str, Any], side: str = "left") -> dict[str, Any]:
    co2 = int(parse_number(data.get("co2_ppm", data.get("co2")), 0.0))
    temp = parse_number(data.get("temperature_c", data.get("temp")), 0.0)
    humid = parse_number(data.get("humidity_percent", data.get("humid")), 0.0)
    nh3 = parse_number(data.get("nh3_voltage", data.get("nh3")), 0.0)

    return {
        "type": "robot_update",
        "co2": co2,
        "temp": round(temp, 3),
        "humid": round(humid, 3),
        "nh3": round(nh3, 6),
        "side": side,
        "received_at": time.time(),
    }


def parse_sensor_line(line: str, side: str = "left") -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None

    if line.startswith("{") and line.endswith("}"):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return payload_from_sensor_dict(data, side=side)
        return None

    csv_fields = [item.strip() for item in line.split(",")]
    if len(csv_fields) == 4:
        try:
            return payload_from_sensor_dict(
                {
                    "co2": int(float(csv_fields[0])),
                    "temp": float(csv_fields[1]),
                    "humid": float(csv_fields[2]),
                    "nh3": float(csv_fields[3]),
                },
                side=side,
            )
        except ValueError:
            return None

    patterns = {
        "co2": r"CO2:\s*([-+]?\d+(?:\.\d+)?)",
        "temp": r"Temp:\s*([-+]?\d+(?:\.\d+)?)",
        "humid": r"Humidity:\s*([-+]?\d+(?:\.\d+)?)",
        "nh3": r"NH3:\s*([-+]?\d+(?:\.\d+)?)",
    }
    matched = {}
    for key, pattern in patterns.items():
        hit = re.search(pattern, line, re.IGNORECASE)
        if hit:
            matched[key] = hit.group(1)

    if {"co2", "temp", "humid"}.issubset(matched):
        return payload_from_sensor_dict(matched, side=side)

    return None


def import_serial_module():
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for Arduino streaming") from exc
    return serial


def import_websocket_module():
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("websocket-client is required for WebSocket streaming") from exc
    return websocket


def iter_serial_lines(port: str, baud_rate: int, timeout: float):
    serial = import_serial_module()
    with serial.Serial(port, baud_rate, timeout=timeout) as ser:
        ser.flush()
        while True:
            raw = ser.readline()
            if not raw:
                continue
            yield raw.decode("utf-8", errors="replace").strip()


def stream(args: argparse.Namespace) -> int:
    if args.sample:
        payload = parse_sensor_line(SAMPLE_SENSOR_LINE, side=args.side)
        if payload is None:
            print("sample payload parse failed", file=sys.stderr)
            return 1
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False))
            return 0
        lines = [SAMPLE_SENSOR_LINE]
    else:
        lines = iter_serial_lines(args.serial_port, args.baud_rate, args.read_timeout_s)

    if args.dry_run:
        count = 0
        for line in lines:
            payload = parse_sensor_line(line, side=args.side)
            if payload is not None:
                print(json.dumps(payload, ensure_ascii=False))
                count += 1
                if args.once:
                    break
        return 0 if count else 1

    websocket = import_websocket_module()

    while True:
        try:
            print(f"[CIS] connecting websocket: {args.ws_url}")
            ws = websocket.create_connection(args.ws_url, timeout=args.websocket_timeout_s)
            print(f"[CIS] streaming Arduino {args.serial_port} -> {args.ws_url}")

            for line in lines:
                payload = parse_sensor_line(line, side=args.side)
                if payload is None:
                    continue
                ws.send(json.dumps(payload, ensure_ascii=False))
                print(
                    "[sent] "
                    f"CO2={payload['co2']}ppm "
                    f"T={payload['temp']}C "
                    f"H={payload['humid']}% "
                    f"NH3={payload['nh3']}"
                )
                if args.once:
                    ws.close()
                    return 0

        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(f"[CIS] bridge error: {exc}", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(args.reconnect_delay_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--read-timeout-s", type=float, default=1.0)
    parser.add_argument("--websocket-timeout-s", type=float, default=5.0)
    parser.add_argument("--reconnect-delay-s", type=float, default=5.0)
    parser.add_argument("--side", default="left")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Use a built-in sample sensor frame.")
    return parser


def main() -> int:
    return stream(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
