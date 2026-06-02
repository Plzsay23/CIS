#!/usr/bin/env python3
import json
import re
import time

import serial
import websocket


# ====================================================================
# C.I.S Edge-Hub network and hardware settings
# ====================================================================
# The laptop and Jetson must be connected to the same Wi-Fi router.
LAPTOP_IP = "192.168.200.162"
TOMCAT_PORT = "8080"
PROJECT_NAME = "CIS_Server"

WS_URL = f"ws://{LAPTOP_IP}:{TOMCAT_PORT}/{PROJECT_NAME}/eggstream"

# Arduino Uno / QinHeng Electronics serial port.
SERIAL_PORT = "/dev/ttyACM1"
BAUD_RATE = 9600

RECONNECT_DELAY_S = 5
SERIAL_LOOP_DELAY_S = 0.1


def on_message(ws, message):
    # Server control messages or echo messages can be handled here later.
    pass


def on_error(ws, error):
    print(f"[Network error]: {error}")
    print("Tip: check Windows Firewall and confirm Eclipse Tomcat is running.")


def on_close(ws, close_status_code, close_msg):
    print(f"[WebSocket closed] code={close_status_code}, msg={close_msg}")
    print(f"Reconnecting in {RECONNECT_DELAY_S} seconds...")


def parse_sensor_line(line):
    """Parse Arduino CSV or the human-readable MH-Z19B/DHT11 output."""
    data_fields = line.split(",")

    if len(data_fields) == 4:
        return {
            "type": "robot_update",
            "co2": int(data_fields[0]),
            "temp": float(data_fields[1]),
            "humid": float(data_fields[2]),
            "nh3": float(data_fields[3]),
            "side": "left",
        }

    co2_match = re.search(r"CO2:\s*([-+]?\d+)", line, re.IGNORECASE)
    temp_match = re.search(r"Temp:\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
    humid_match = re.search(
        r"Humidity:\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE
    )
    nh3_match = re.search(r"NH3:\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)

    if co2_match and temp_match and humid_match:
        return {
            "type": "robot_update",
            "co2": int(co2_match.group(1)),
            "temp": float(temp_match.group(1)),
            "humid": float(humid_match.group(1)),
            "nh3": float(nh3_match.group(1)) if nh3_match else 0.0,
            "side": "left",
        }

    return None


def stream_serial_to_websocket(ws):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            ser.flush()
            print(f"Arduino serial opened: {SERIAL_PORT} @ {BAUD_RATE}")
            print("Streaming sensor data to laptop websocket...\n")

            while ws.keep_running:
                if ser.in_waiting <= 0:
                    time.sleep(SERIAL_LOOP_DELAY_S)
                    continue

                line = ser.readline().decode("utf-8", errors="replace").strip()

                if not line:
                    continue

                try:
                    payload = parse_sensor_line(line)
                except ValueError:
                    print(f"[Data skipped] Invalid numeric sensor line: {line!r}")
                    continue

                if payload is None:
                    print(f"[Data skipped] Expected 4 CSV fields, got: {line!r}")
                    continue

                ws.send(json.dumps(payload))
                print(
                    "[Sent] "
                    f"CO2={payload['co2']}ppm | "
                    f"Temp={payload['temp']}C | "
                    f"Humid={payload['humid']}% | "
                    f"NH3={payload['nh3']}"
                )

                time.sleep(SERIAL_LOOP_DELAY_S)

    except serial.SerialException as e:
        print(f"[Hardware error] Arduino not available on {SERIAL_PORT}: {e}")
        ws.close()


def on_open(ws):
    print(f"WebSocket connected: {WS_URL}")
    stream_serial_to_websocket(ws)


def connect_and_run():
    while True:
        print(f"[C.I.S remote bridge] Connecting to {WS_URL}...")
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()
        time.sleep(RECONNECT_DELAY_S)


if __name__ == "__main__":
    connect_and_run()
