#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import serial
from serial.tools import list_ports


@dataclass
class SensorFrame:
    raw_line: str
    received_at: str

    uptime_ms: int | None

    co2_ppm_raw: float | None
    co2_error: int | None
    co2_ppm_filtered: float | None
    co2_suspect: bool
    co2_status: str

    temperature_c: float | None
    humidity_percent: float | None
    dht_error: bool
    temp_hum_status: str

    nh3_raw: float | None
    nh3_voltage: float | None
    nh3_status: str

    overall_status: str


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    v = to_float(value)
    if v is None:
        return None
    return int(v)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def sanitize_co2(co2_ppm: float | None) -> tuple[float | None, bool]:
    """
    CO2 값 필터링.

    350 미만:
      일반 환경에서 비정상 가능성이 큼.

    10000 초과:
      입김/밀폐/센서 이상/보정 이상 가능성이 커서 suspect 처리.
      단, co2_ppm_raw에는 원본값을 유지함.
    """
    if co2_ppm is None:
        return None, False

    if co2_ppm < 350 or co2_ppm > 10000:
        return None, True

    return co2_ppm, False


def co2_status(co2_ppm_filtered: float | None, co2_error: int | None, co2_suspect: bool) -> str:
    if co2_error is not None:
        return "ERROR"

    if co2_suspect:
        return "SUSPECT"

    if co2_ppm_filtered is None:
        return "UNKNOWN"

    if co2_ppm_filtered >= 3000:
        return "DANGER"

    if co2_ppm_filtered >= 2500:
        return "WARNING"

    return "NORMAL"


def temp_hum_status(temperature_c: float | None, humidity_percent: float | None, dht_error: bool) -> str:
    if dht_error:
        return "ERROR"

    if temperature_c is None or humidity_percent is None:
        return "UNKNOWN"

    if temperature_c >= 31 or humidity_percent >= 80:
        return "DANGER"

    if temperature_c >= 28 or humidity_percent >= 75:
        return "WARNING"

    if temperature_c >= 25 or humidity_percent >= 70:
        return "NOTICE"

    return "NORMAL"


def nh3_status_from_raw(nh3_raw: float | None) -> str:
    """
    SEN0567은 현재 raw ADC 기반으로 판단.
    nh3_raw는 ppm이 아니다.

    현재 테스트 기준:
      A0-GND 10k pulldown 사용.
      깨끗한 환경 baseline을 잡고 threshold는 현장에서 조정해야 함.
    """
    if nh3_raw is None:
        return "UNKNOWN"

    if nh3_raw >= 150:
        return "STRONG_WARNING"

    if nh3_raw >= 100:
        return "WARNING"

    if nh3_raw >= 50:
        return "NOTICE"

    return "NORMAL"


def overall_status(co2_s: str, th_s: str, nh3_s: str) -> str:
    statuses = {co2_s, th_s, nh3_s}

    if "ERROR" in statuses:
        return "ERROR"

    if "DANGER" in statuses or "STRONG_WARNING" in statuses:
        return "DANGER"

    if "WARNING" in statuses:
        return "WARNING"

    if "SUSPECT" in statuses:
        return "SUSPECT"

    if "NOTICE" in statuses:
        return "NOTICE"

    if statuses == {"NORMAL"}:
        return "NORMAL"

    return "UNKNOWN"


def parse_sensor_json(line: str) -> SensorFrame | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    uptime_ms = to_int(data.get("uptime_ms"))

    co2_raw = to_float(data.get("co2_ppm"))
    co2_err = to_int(data.get("co2_error"))

    temperature_c = to_float(data.get("temperature_c"))
    humidity_percent = to_float(data.get("humidity_percent"))
    dht_error = to_bool(data.get("dht_error"))

    nh3_raw = to_float(data.get("nh3_raw"))
    nh3_voltage = to_float(data.get("nh3_voltage"))

    co2_filtered, co2_suspect = sanitize_co2(co2_raw)
    co2_s = co2_status(co2_filtered, co2_err, co2_suspect)
    th_s = temp_hum_status(temperature_c, humidity_percent, dht_error)
    nh3_s = nh3_status_from_raw(nh3_raw)
    all_s = overall_status(co2_s, th_s, nh3_s)

    return SensorFrame(
        raw_line=line,
        received_at=now_text(),

        uptime_ms=uptime_ms,

        co2_ppm_raw=co2_raw,
        co2_error=co2_err,
        co2_ppm_filtered=co2_filtered,
        co2_suspect=co2_suspect,
        co2_status=co2_s,

        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        dht_error=dht_error,
        temp_hum_status=th_s,

        nh3_raw=nh3_raw,
        nh3_voltage=nh3_voltage,
        nh3_status=nh3_s,

        overall_status=all_s,
    )


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NULL"
    return f"{value:.{digits}f}"


def fmt_intlike(value: float | None) -> str:
    if value is None:
        return "NULL"
    return str(int(value))


def print_frame(frame: SensorFrame) -> None:
    print(
        f"[{frame.received_at}] "
        f"overall={frame.overall_status:<8} | "
        f"CO2={fmt_intlike(frame.co2_ppm_filtered):>5} ppm "
        f"(raw={fmt_intlike(frame.co2_ppm_raw):>5}, err={frame.co2_error}, "
        f"suspect={str(frame.co2_suspect).lower()}, status={frame.co2_status}) | "
        f"T={fmt_num(frame.temperature_c, 1):>5} C "
        f"H={fmt_num(frame.humidity_percent, 1):>5} % "
        f"(dht_error={str(frame.dht_error).lower()}, status={frame.temp_hum_status}) | "
        f"NH3_RAW={fmt_intlike(frame.nh3_raw):>4} "
        f"NH3_V={fmt_num(frame.nh3_voltage, 3):>5} V "
        f"(status={frame.nh3_status})",
        flush=True,
    )


def find_serial_port() -> str | None:
    ports = list(list_ports.comports())

    if not ports:
        return None

    keywords = [
        "arduino",
        "ch340",
        "ch341",
        "usb serial",
        "usb2.0-serial",
        "ttyacm",
        "ttyusb",
    ]

    for port in ports:
        text = f"{port.device} {port.description} {port.manufacturer}".lower()
        if any(keyword in text for keyword in keywords):
            return port.device

    return ports[0].device


def open_serial(port: str, baud: int, timeout: float, reset_wait_s: float) -> serial.Serial:
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=timeout,
    )

    # Arduino는 시리얼 연결 시 리셋되는 경우가 많음.
    time.sleep(reset_wait_s)
    ser.reset_input_buffer()

    return ser


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read Arduino environmental sensor JSON from serial."
    )

    parser.add_argument(
        "--port",
        default=None,
        help="Serial port. Example: /dev/ttyACM0 or /dev/ttyUSB0. If omitted, auto-detect.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Serial baudrate. Default: 9600.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Serial readline timeout seconds. Default: 2.0.",
    )
    parser.add_argument(
        "--reset-wait-s",
        type=float,
        default=2.0,
        help="Wait seconds after opening serial. Default: 2.0.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON lines only.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print malformed or non-JSON lines.",
    )

    args = parser.parse_args()

    port = args.port or find_serial_port()

    if port is None:
        print("ERROR: Arduino serial port not found.", file=sys.stderr)
        print("Check with: ls /dev/ttyACM* /dev/ttyUSB*", file=sys.stderr)
        return 1

    print(f"[OK] Opening serial: {port} @ {args.baud}")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        ser = open_serial(port, args.baud, args.timeout, args.reset_wait_s)

    except serial.SerialException as e:
        print(f"ERROR: Failed to open serial port: {e}", file=sys.stderr)
        return 1

    try:
        while True:
            try:
                raw_bytes = ser.readline()

            except serial.SerialException as e:
                print(f"[SERIAL ERROR] {e}", file=sys.stderr, flush=True)
                time.sleep(1.0)
                continue

            if not raw_bytes:
                continue

            line = raw_bytes.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            if args.raw:
                print(line, flush=True)
                continue

            frame = parse_sensor_json(line)

            if frame is None:
                if args.show_skipped:
                    print(f"[SKIP] {line}", flush=True)
                continue

            print_frame(frame)

    except KeyboardInterrupt:
        print("\n[OK] stopped.")
        return 0

    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())