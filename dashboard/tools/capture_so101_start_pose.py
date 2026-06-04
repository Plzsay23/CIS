#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


ARM_MOTORS = {
    "arm_shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def read_average_position(bus, motor_names, samples: int, dt: float):
    rows = []

    for _ in range(samples):
        pos = bus.sync_read("Present_Position", motor_names, normalize=False)
        rows.append(pos)
        time.sleep(dt)

    avg = {}

    for name in motor_names:
        avg[name] = sum(float(row[name]) for row in rows) / len(rows)

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--out", default="dashboard/config/so101_start_pose.json")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--disable-torque", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    motor_names = list(ARM_MOTORS.keys())

    bus = FeetechMotorsBus(
        port=args.port,
        motors=ARM_MOTORS,
    )

    print(f"[INFO] connect: {args.port}")
    bus.connect()

    try:
        if args.disable_torque:
            print("[INFO] disable torque")
            bus.disable_torque(motor_names)

        print("[INFO] reading current arm pose")
        avg_pos = read_average_position(
            bus=bus,
            motor_names=motor_names,
            samples=args.samples,
            dt=args.dt,
        )

        raw_position = {
            name: int(round(float(value)))
            for name, value in avg_pos.items()
        }

        payload = {
            "type": "cis_so101_start_pose",
            "port": args.port,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unit": "raw_present_position_tick",
            "note": "Dashboard-readable current SO-101 arm pose. These are raw motor ticks, not normalized LeRobot actions.",
            "raw_position": raw_position,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("[OK] saved start pose")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[OK] output: {out_path}")

    finally:
        bus.disconnect()
        print("[INFO] disconnected")


if __name__ == "__main__":
    main()
