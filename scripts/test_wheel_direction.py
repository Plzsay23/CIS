#!/usr/bin/env python3

import argparse
import time
import traceback

from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import Motor, MotorNormMode


WHEEL_MOTORS = {
    "left_wheel": Motor(
        id=7,
        model="sts3215",
        norm_mode=MotorNormMode.RANGE_M100_100,
    ),
    "rear_wheel": Motor(
        id=8,
        model="sts3215",
        norm_mode=MotorNormMode.RANGE_M100_100,
    ),
    "right_wheel": Motor(
        id=9,
        model="sts3215",
        norm_mode=MotorNormMode.RANGE_M100_100,
    ),
}


def stop_all(bus):
    bus.sync_write(
        "Goal_Velocity",
        {
            "left_wheel": 0,
            "rear_wheel": 0,
            "right_wheel": 0,
        },
        normalize=False,
    )


def set_velocity_mode(bus):
    for name in WHEEL_MOTORS:
        bus.write(
            "Operating_Mode",
            name,
            OperatingMode.VELOCITY.value,
            normalize=False,
        )


def run_one(bus, target_name, velocity, duration):
    values = {
        "left_wheel": 0,
        "rear_wheel": 0,
        "right_wheel": 0,
    }
    values[target_name] = velocity

    print()
    print("========================================")
    print(f"[TEST] {target_name} velocity = {velocity}")
    print("Observe physical rotation direction.")
    print("========================================")

    bus.sync_write("Goal_Velocity", values, normalize=False)
    time.sleep(duration)
    stop_all(bus)
    time.sleep(0.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--speed", type=int, default=120)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors=WHEEL_MOTORS,
    )

    try:
        print("[STEP] Connect")
        bus.connect(handshake=True)

        print("[STEP] Enable torque only for wheel motors")
        bus.enable_torque(list(WHEEL_MOTORS.keys()))

        print("[STEP] Set velocity mode")
        set_velocity_mode(bus)

        print("[STEP] Stop all")
        stop_all(bus)
        time.sleep(0.7)

        for name in WHEEL_MOTORS:
            run_one(bus, name, +args.speed, args.duration)
            run_one(bus, name, -args.speed, args.duration)

        print()
        print("[DONE] Direction test finished.")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")

    except Exception:
        print("\n[ERROR]")
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