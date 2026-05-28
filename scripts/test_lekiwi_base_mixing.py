#!/usr/bin/env python3

import argparse
import time
import traceback

import numpy as np

from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorNormMode


WHEEL_MOTORS = {
    "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
}


def degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    raw = int(round(degps * steps_per_deg))
    return max(min(raw, 0x7FFF), -0x8000)


def body_to_wheel_raw(
    x: float,
    y: float,
    theta: float,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
    max_raw: int = 3000,
) -> dict[str, int]:
    """
    실제 src/lerobot/robots/lekiwi/lekiwi.py 의 _body_to_wheel_raw()와 같은 계산식.
    x: m/s, forward +
    y: m/s, left +
    theta: deg/s, ccw +
    """
    theta_rad = theta * (np.pi / 180.0)
    velocity_vector = np.array([x, y, theta_rad])

    angles = np.radians(np.array([240, 0, 120]) - 90)
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius
    wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

    steps_per_deg = 4096.0 / 360.0
    raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]

    max_raw_computed = max(raw_floats)
    if max_raw_computed > max_raw:
        scale = max_raw / max_raw_computed
        wheel_degps = wheel_degps * scale

    wheel_raw = [degps_to_raw(deg) for deg in wheel_degps]

    return {
        "base_left_wheel": wheel_raw[0],
        "base_back_wheel": wheel_raw[1],
        "base_right_wheel": wheel_raw[2],
    }


def stop_all(bus):
    bus.sync_write("Goal_Velocity", dict.fromkeys(WHEEL_MOTORS.keys(), 0), normalize=False)


def configure_base(bus):
    names = list(WHEEL_MOTORS.keys())

    print("[STEP] Disable torque")
    bus.disable_torque(names)

    print("[STEP] Set velocity mode")
    for name in names:
        bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value, normalize=False)

    print("[STEP] Set acceleration")
    for name in names:
        try:
            bus.write("Maximum_Acceleration", name, 254, normalize=False)
        except Exception as e:
            print(f"[SKIP] {name}.Maximum_Acceleration: {e}")

        try:
            bus.write("Acceleration", name, 254, normalize=False)
        except Exception as e:
            print(f"[SKIP] {name}.Acceleration: {e}")

    print("[STEP] Enable torque")
    bus.enable_torque(names)

    stop_all(bus)


def run_motion(bus, label, x, y, theta, duration):
    print("\n========================================")
    print(f"[TEST] {label}")
    print(f"body cmd: x={x:.3f} m/s, y={y:.3f} m/s, theta={theta:.1f} deg/s")

    wheel_cmd = body_to_wheel_raw(x, y, theta)
    print(f"wheel raw: {wheel_cmd}")
    print("========================================")

    bus.sync_write("Goal_Velocity", wheel_cmd, normalize=False)
    time.sleep(duration)
    stop_all(bus)
    time.sleep(0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--speed-level", choices=["slow", "medium", "fast"], default="slow")
    parser.add_argument("--duration", type=float, default=1.2)
    args = parser.parse_args()

    speed_table = {
        "slow": {"xy": 0.1, "theta": 30.0},
        "medium": {"xy": 0.2, "theta": 60.0},
        "fast": {"xy": 0.3, "theta": 90.0},
    }

    xy = speed_table[args.speed_level]["xy"]
    th = speed_table[args.speed_level]["theta"]

    bus = FeetechMotorsBus(port=args.port, motors=WHEEL_MOTORS)

    try:
        print("========================================")
        print("LeKiwi actual mixing test")
        print(f"port={args.port}")
        print(f"speed_level={args.speed_level}, xy={xy}, theta={th}")
        print("========================================")

        bus.connect()

        print("[STEP] Ping")
        for name, m in WHEEL_MOTORS.items():
            print(f"{name} id={m.id} ping={bus.ping(name)}")

        configure_base(bus)

        run_motion(bus, "forward: w", x=xy, y=0.0, theta=0.0, duration=args.duration)
        run_motion(bus, "backward: s", x=-xy, y=0.0, theta=0.0, duration=args.duration)
        run_motion(bus, "left strafe: a", x=0.0, y=xy, theta=0.0, duration=args.duration)
        run_motion(bus, "right strafe: d", x=0.0, y=-xy, theta=0.0, duration=args.duration)
        run_motion(bus, "rotate left: z", x=0.0, y=0.0, theta=th, duration=args.duration)
        run_motion(bus, "rotate right: x", x=0.0, y=0.0, theta=-th, duration=args.duration)

        print("\n[DONE] mixing test finished")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")

    except Exception:
        traceback.print_exc()

    finally:
        try:
            stop_all(bus)
        except Exception:
            pass

        try:
            bus.disable_torque(list(WHEEL_MOTORS.keys()))
        except Exception:
            pass

        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()