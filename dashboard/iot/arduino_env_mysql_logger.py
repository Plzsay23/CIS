#!/usr/bin/env python3
"""
Arduino Serial -> MySQL logger for LeKiwi barn environment monitoring.

Expected Arduino line format, recommended:
  {"temperature_c":23.7,"humidity_percent":64.2,"co2_ppm":1820,"nh3_ppm":7.4}

Also accepted:
  TEMP:23.7,HUM:64.2,CO2:1820,NH3:7.4
  temperature=23.7, humidity=64.2, co2=1820, nh3=7.4

Partial lines are accepted, but one complete JSON/CSV line per measurement is recommended.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pymysql
import serial


@dataclass
class EnvMeasurement:
    robot_id: str
    measured_at: str

    map_x: float | None
    map_y: float | None
    yaw: float | None

    temperature_c: float | None
    humidity_percent: float | None
    co2_ppm: float | None
    nh3_ppm: float | None

    ventilation_status: str
    source: str
    raw_line: str


STOP = False


def handle_stop_signal(signum: int, frame: Any) -> None:
    global STOP
    STOP = True


def now_mysql_datetime_ms() -> str:
    # Local time. For a local barn dashboard this is usually easier to read.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None

    # Remove common units and separators.
    text = (
        text.replace("ppm", "")
        .replace("%", "")
        .replace("℃", "")
        .replace("°C", "")
        .replace("C", "")
        .replace(",", "")
        .strip()
    )

    try:
        return float(text)
    except ValueError:
        return None


def first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return to_float(match.group(0))


def normalize_keys(data: dict[str, Any]) -> dict[str, float | None]:
    """
    Normalize many possible key names into:
      temperature_c, humidity_percent, co2_ppm, nh3_ppm
    """
    aliases = {
        "temperature_c": [
            "temperature_c",
            "temperature",
            "temp",
            "TEMP",
            "Temperature",
            "T",
        ],
        "humidity_percent": [
            "humidity_percent",
            "humidity",
            "hum",
            "HUM",
            "Humidity",
            "H",
        ],
        "co2_ppm": [
            "co2_ppm",
            "co2",
            "CO2",
            "co2ppm",
            "CO2ppm",
        ],
        "nh3_ppm": [
            "nh3_ppm",
            "nh3",
            "NH3",
            "nh3ppm",
            "NH3ppm",
            "ammonia",
            "Ammonia",
        ],
    }

    out: dict[str, float | None] = {
        "temperature_c": None,
        "humidity_percent": None,
        "co2_ppm": None,
        "nh3_ppm": None,
    }

    for target_key, key_candidates in aliases.items():
        for key in key_candidates:
            if key in data:
                out[target_key] = to_float(data[key])
                break

    return out


def parse_json_line(line: str) -> dict[str, float | None] | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    return normalize_keys(data)


def parse_key_value_line(line: str) -> dict[str, float | None] | None:
    """
    Accepts:
      TEMP:23.7,HUM:64.2,CO2:1820,NH3:7.4
      temperature=23.7 humidity=64.2 co2=1820 nh3=7.4
      CO2: 500 ppm
    """
    pairs = re.findall(
        r"([A-Za-z0-9_]+)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        line,
    )

    if pairs:
        raw: dict[str, Any] = {}
        for key, value in pairs:
            raw[key] = value
        return normalize_keys(raw)

    # Fallback for lines such as "CO2: 500 ppm" or "NH3 7.2 ppm"
    lower = line.lower()

    out: dict[str, float | None] = {
        "temperature_c": None,
        "humidity_percent": None,
        "co2_ppm": None,
        "nh3_ppm": None,
    }

    if "co2" in lower:
        out["co2_ppm"] = first_number(line)

    elif "nh3" in lower or "ammonia" in lower:
        out["nh3_ppm"] = first_number(line)

    elif "temp" in lower or "temperature" in lower:
        out["temperature_c"] = first_number(line)

    elif "hum" in lower or "humidity" in lower:
        out["humidity_percent"] = first_number(line)

    if any(v is not None for v in out.values()):
        return out

    return None


def parse_arduino_line(line: str) -> dict[str, float | None] | None:
    line = line.strip()

    if not line:
        return None

    parsed = parse_json_line(line)
    if parsed is not None:
        return parsed

    parsed = parse_key_value_line(line)
    if parsed is not None:
        return parsed

    return None


def ventilation_status(
    temperature_c: float | None,
    humidity_percent: float | None,
    co2_ppm: float | None,
    nh3_ppm: float | None,
) -> str:
    """
    Barn air quality rule.

    CO2:
      2500 ppm: recommended upper value for good indoor air quality
      3000 ppm: threshold limit

    NH3:
      10 ppm: recommended limit for good indoor air quality
      20 ppm: short period exposure limit
      25 ppm: should not exceed

    Temperature/humidity use the dashboard demo rules:
      temp >= 31 or humidity >= 80 => 불량
      temp >= 28 or humidity >= 75 => 주의
      temp >= 25 or humidity >= 70 => 보통
    """
    if (
        (temperature_c is not None and temperature_c >= 31)
        or (humidity_percent is not None and humidity_percent >= 80)
        or (co2_ppm is not None and co2_ppm >= 3000)
        or (nh3_ppm is not None and nh3_ppm >= 25)
    ):
        return "불량"

    if (
        (temperature_c is not None and temperature_c >= 28)
        or (humidity_percent is not None and humidity_percent >= 75)
        or (co2_ppm is not None and co2_ppm >= 2500)
        or (nh3_ppm is not None and nh3_ppm >= 20)
    ):
        return "주의"

    if (
        (temperature_c is not None and temperature_c >= 25)
        or (humidity_percent is not None and humidity_percent >= 70)
        or (nh3_ppm is not None and nh3_ppm >= 10)
    ):
        return "보통"

    if (
        temperature_c is not None
        and humidity_percent is not None
        and co2_ppm is not None
        and nh3_ppm is not None
        and 20 <= temperature_c < 25
        and 50 <= humidity_percent < 70
        and co2_ppm < 2500
        and nh3_ppm < 10
    ):
        return "양호"

    return "우수"


def connect_mysql(args: argparse.Namespace):
    return pymysql.connect(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
        read_timeout=5,
        write_timeout=5,
    )


def ensure_schema(conn) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS sensor_measurements (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,

        robot_id VARCHAR(64) NOT NULL,
        measured_at DATETIME(3) NOT NULL,

        map_x DOUBLE NULL,
        map_y DOUBLE NULL,
        yaw DOUBLE NULL,

        temperature_c DOUBLE NULL,
        humidity_percent DOUBLE NULL,
        co2_ppm DOUBLE NULL,
        nh3_ppm DOUBLE NULL,

        ventilation_status VARCHAR(32) NOT NULL,
        source VARCHAR(64) NOT NULL DEFAULT 'arduino',

        raw_line TEXT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_measured_at (measured_at),
        INDEX idx_robot_time (robot_id, measured_at),
        INDEX idx_position (map_x, map_y),
        INDEX idx_status (ventilation_status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    with conn.cursor() as cur:
        cur.execute(sql)


def insert_measurement(conn, m: EnvMeasurement) -> int:
    sql = """
    INSERT INTO sensor_measurements (
        robot_id,
        measured_at,

        map_x,
        map_y,
        yaw,

        temperature_c,
        humidity_percent,
        co2_ppm,
        nh3_ppm,

        ventilation_status,
        source,
        raw_line
    )
    VALUES (
        %(robot_id)s,
        %(measured_at)s,

        %(map_x)s,
        %(map_y)s,
        %(yaw)s,

        %(temperature_c)s,
        %(humidity_percent)s,
        %(co2_ppm)s,
        %(nh3_ppm)s,

        %(ventilation_status)s,
        %(source)s,
        %(raw_line)s
    );
    """

    params = {
        "robot_id": m.robot_id,
        "measured_at": m.measured_at,

        "map_x": m.map_x,
        "map_y": m.map_y,
        "yaw": m.yaw,

        "temperature_c": m.temperature_c,
        "humidity_percent": m.humidity_percent,
        "co2_ppm": m.co2_ppm,
        "nh3_ppm": m.nh3_ppm,

        "ventilation_status": m.ventilation_status,
        "source": m.source,
        "raw_line": m.raw_line,
    }

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.lastrowid)


def make_measurement(
    args: argparse.Namespace,
    parsed: dict[str, float | None],
    raw_line: str,
) -> EnvMeasurement:
    temp = parsed.get("temperature_c")
    hum = parsed.get("humidity_percent")
    co2 = parsed.get("co2_ppm")
    nh3 = parsed.get("nh3_ppm")

    status = ventilation_status(temp, hum, co2, nh3)

    return EnvMeasurement(
        robot_id=args.robot_id,
        measured_at=now_mysql_datetime_ms(),

        # 지금은 위치 연동 전이므로 CLI 기본값 또는 NULL.
        # 나중에 ROS2 /tf, /odom, /map pose와 결합하면 여기를 실제 위치로 채우면 됨.
        map_x=args.map_x,
        map_y=args.map_y,
        yaw=args.yaw,

        temperature_c=temp,
        humidity_percent=hum,
        co2_ppm=co2,
        nh3_ppm=nh3,

        ventilation_status=status,
        source=args.source,
        raw_line=raw_line,
    )


def print_measurement(row_id: int | None, m: EnvMeasurement) -> None:
    prefix = f"[DB id={row_id}]" if row_id is not None else "[DRY-RUN]"

    print(
        f"{prefix} "
        f"time={m.measured_at} "
        f"robot={m.robot_id} "
        f"temp={fmt(m.temperature_c)}C "
        f"hum={fmt(m.humidity_percent)}% "
        f"co2={fmt(m.co2_ppm)}ppm "
        f"nh3={fmt(m.nh3_ppm)}ppm "
        f"status={m.ventilation_status} "
        f"x={fmt(m.map_x)} y={fmt(m.map_y)} yaw={fmt(m.yaw)}",
        flush=True,
    )


def fmt(v: float | None) -> str:
    if v is None:
        return "NULL"
    return f"{v:.2f}"


def open_serial(args: argparse.Namespace) -> serial.Serial:
    ser = serial.Serial(
        port=args.serial_port,
        baudrate=args.baud,
        timeout=args.serial_timeout,
    )

    # Arduino reset delay on serial open.
    time.sleep(args.arduino_reset_wait_s)
    ser.reset_input_buffer()

    return ser


def run(args: argparse.Namespace) -> None:
    conn = connect_mysql(args)
    ensure_schema(conn)

    if args.init_only:
        print("[OK] DB schema initialized.")
        conn.close()
        return

    print("[OK] MySQL connected.")
    print(f"[OK] Listening Arduino serial: {args.serial_port} @ {args.baud}")
    print("[INFO] Press Ctrl+C to stop.")

    if args.dry_run:
        print("[INFO] dry-run mode: DB insert disabled.")

    ser = open_serial(args)

    last_db_error_at = 0.0

    while not STOP:
        try:
            raw_bytes = ser.readline()
        except serial.SerialException as e:
            print(f"[SERIAL ERROR] {e}", file=sys.stderr, flush=True)
            time.sleep(1.0)
            continue

        if not raw_bytes:
            continue

        raw_line = raw_bytes.decode("utf-8", errors="ignore").strip()

        if not raw_line:
            continue

        parsed = parse_arduino_line(raw_line)

        if parsed is None:
            print(f"[SKIP] cannot parse: {raw_line}", flush=True)
            continue

        if not any(v is not None for v in parsed.values()):
            print(f"[SKIP] no numeric value: {raw_line}", flush=True)
            continue

        m = make_measurement(args, parsed, raw_line)

        if args.dry_run:
            print_measurement(None, m)
            continue

        try:
            row_id = insert_measurement(conn, m)
            print_measurement(row_id, m)

        except Exception as e:
            now = time.time()

            # Avoid spamming error messages too quickly.
            if now - last_db_error_at > 2.0:
                print(f"[DB ERROR] {e}", file=sys.stderr, flush=True)
                print("[DB] reconnecting...", file=sys.stderr, flush=True)
                last_db_error_at = now

            try:
                conn.close()
            except Exception:
                pass

            time.sleep(1.0)
            conn = connect_mysql(args)
            ensure_schema(conn)

    try:
        ser.close()
    except Exception:
        pass

    try:
        conn.close()
    except Exception:
        pass

    print("\n[OK] stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Arduino environmental sensor values from serial and save them to MySQL."
    )

    parser.add_argument(
        "--serial-port",
        default=os.getenv("ARDUINO_SERIAL_PORT", "/dev/ttyACM0"),
        help="Arduino serial port. Default: /dev/ttyACM0",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=int(os.getenv("ARDUINO_BAUD", "9600")),
        help="Arduino baudrate. Default: 9600",
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=float(os.getenv("ARDUINO_SERIAL_TIMEOUT", "2.0")),
        help="Serial readline timeout seconds. Default: 2.0",
    )
    parser.add_argument(
        "--arduino-reset-wait-s",
        type=float,
        default=float(os.getenv("ARDUINO_RESET_WAIT_S", "2.0")),
        help="Wait after opening serial because Arduino often resets. Default: 2.0",
    )

    parser.add_argument(
        "--db-host",
        default=os.getenv("MYSQL_HOST", "localhost"),
        help="MySQL host. Default: localhost",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("MYSQL_PORT", "3306")),
        help="MySQL port. Default: 3306",
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("MYSQL_USER", "lekiwi"),
        help="MySQL user. Default: lekiwi",
    )
    parser.add_argument(
        "--db-password",
        default=os.getenv("MYSQL_PASSWORD", "lekiwi1234!"),
        help="MySQL password. Default comes from MYSQL_PASSWORD or test password.",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("MYSQL_DATABASE", "lekiwi_iot"),
        help="MySQL database. Default: lekiwi_iot",
    )

    parser.add_argument(
        "--robot-id",
        default=os.getenv("ROBOT_ID", "lekiwi_left"),
        help="Robot id to store with each measurement.",
    )
    parser.add_argument(
        "--source",
        default=os.getenv("SENSOR_SOURCE", "arduino"),
        help="Measurement source label.",
    )

    # Position placeholders.
    # Later, replace these with ROS2 pose lookup from /tf, /odom, /amcl_pose, etc.
    parser.add_argument("--map-x", type=float, default=None)
    parser.add_argument("--map-y", type=float, default=None)
    parser.add_argument("--yaw", type=float, default=None)

    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only create DB schema and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print serial values without DB insert.",
    )

    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    args = parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()