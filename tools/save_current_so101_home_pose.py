#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


ARM_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--out", default="config/so101_home_pose.json")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bus = FeetechMotorsBus(
        port=args.port,
        motors=ARM_MOTORS,
    )

    print(f"[INFO] Connecting to {args.port}")
    bus.connect()

    try:
        # 안전하게 현재 위치만 읽을 목적이면 토크를 꺼둔다.
        try:
            bus.disable_torque()
            print("[INFO] Torque disabled. Move robot to desired home pose if needed.")
        except Exception as e:
            print(f"[WARN] Failed to disable torque: {e}")

        print("[INFO] Reading current motor positions...")
        samples = []

        motor_names = list(ARM_MOTORS.keys())

        for _ in range(args.samples):
            pos = bus.sync_read("Present_Position", motor_names)
            samples.append(pos)
            time.sleep(args.dt)

        avg_pos = {}
        for name in motor_names:
            avg_pos[name] = sum(float(s[name]) for s in samples) / len(samples)

        # int로 저장. 필요하면 float 그대로 써도 됨.
        home_pose = {name: int(round(value)) for name, value in avg_pos.items()}

        payload = {
            "port": args.port,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "so101_home_pose",
            "unit": "present_position_raw_or_normalized_by_bus",
            "motors": home_pose,
        }

        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("[OK] Saved current pose:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[OK] Output: {out_path}")

    finally:
        bus.disconnect()
        print("[INFO] Disconnected")


if __name__ == "__main__":
    main()
