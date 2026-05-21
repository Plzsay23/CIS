#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime

import serial
from serial.tools import list_ports


def find_arduino_port() -> str | None:
    ports = list(list_ports.comports())

    if not ports:
        return None

    # Arduino/CH340/USB Serial 계열 우선 탐색
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
        if any(k in text for k in keywords):
            return port.device

    # 못 찾으면 첫 번째 포트 반환
    return ports[0].device


def format_value(value, suffix: str = "", digits: int | None = None) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}{suffix}"

    return f"{value}{suffix}"


def print_sensor_row(data: dict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    co2 = data.get("co2_ppm")
    co2_status = data.get("co2_status")
    temp = data.get("temperature_c")
    hum = data.get("humidity_percent")
    nh3_raw = data.get("nh3_raw")
    nh3_voltage = data.get("nh3_voltage")

    co2_text = format_value(co2, " ppm")
    temp_text = format_value(temp, " °C", digits=1)
    hum_text = format_value(hum, " %", digits=1)
    nh3_raw_text = format_value(nh3_raw)
    nh3_voltage_text = format_value(nh3_voltage, " V", digits=3)

    print(
        f"[{now}] "
        f"CO2={co2_text:<9} "
        f"CO2_STATUS={co2_status:<7} "
        f"TEMP={temp_text:<8} "
        f"HUM={hum_text:<8} "
        f"NH3_RAW={nh3_raw_text:<5} "
        f"NH3_VOLT={nh3_voltage_text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read Arduino sensor JSON lines from serial."
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port, e.g. /dev/ttyACM0 or /dev/ttyUSB0. If omitted, auto-detect.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Serial baudrate. Default: 9600",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw serial lines instead of formatted sensor values.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Serial read timeout in seconds.",
    )
    args = parser.parse_args()

    port = args.port or find_arduino_port()

    if port is None:
        print("ERROR: Arduino serial port not found.", file=sys.stderr)
        print("Check: ls /dev/ttyACM* /dev/ttyUSB*", file=sys.stderr)
        return 1

    print(f"Opening serial port: {port} @ {args.baud} baud")

    try:
        with serial.Serial(port, args.baud, timeout=args.timeout) as ser:
            # Arduino는 시리얼 연결 시 리셋되는 경우가 많아서 잠깐 대기
            time.sleep(2.0)

            print("Reading sensor data. Press Ctrl+C to stop.")

            while True:
                raw_line = ser.readline()

                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()

                if not line:
                    continue

                if args.raw:
                    print(line)
                    continue

                if not line.startswith("{"):
                    print(f"[INFO] {line}")
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] Invalid JSON line: {line}")
                    continue

                # event 라인은 상태 메시지로 출력
                if "event" in data:
                    print(f"[EVENT] {data}")
                    continue

                print_sensor_row(data)

    except KeyboardInterrupt:
        print("\nStopped.")
        return 0

    except serial.SerialException as e:
        print(f"ERROR: Serial error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())