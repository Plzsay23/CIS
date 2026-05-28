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


def try_write(bus, names, data_name, value, normalize=False):
    """
    LeRobot/Feetech 버전에 따라 register 이름이 다를 수 있으므로
    실패하면 False만 반환한다.
    """
    try:
        if isinstance(names, str):
            bus.write(data_name, names, value, normalize=normalize)
        else:
            bus.sync_write(
                data_name,
                {name: value for name in names},
                normalize=normalize,
            )
        print(f"[OK] write {data_name} = {value}")
        return True
    except Exception as e:
        print(f"[SKIP] write {data_name} failed: {type(e).__name__}: {e}")
        return False


def try_read(bus, name, data_name, normalize=False):
    try:
        value = bus.read(data_name, name, normalize=normalize)
        print(f"[OK] read {name}.{data_name} = {value}")
        return value
    except Exception as e:
        print(f"[SKIP] read {name}.{data_name} failed: {type(e).__name__}: {e}")
        return None


def set_velocity_mode(bus, motor_names):
    """
    Feetech STS 계열을 속도 모드로 바꾼다.
    register 이름이 코드 버전마다 다를 수 있어서 후보를 순서대로 시도한다.
    """
    print("\n[STEP] Set velocity mode")

    # 가장 가능성 높은 이름부터 시도
    mode_register_candidates = [
        "Mode",
        "Operating_Mode",
        "Operation_Mode",
    ]

    for reg in mode_register_candidates:
        ok_all = True
        for name in motor_names:
            ok = try_write(
                bus,
                name,
                reg,
                OperatingMode.VELOCITY.value,
                normalize=False,
            )
            ok_all = ok_all and ok

        if ok_all:
            print(f"[OK] velocity mode set using register: {reg}")
            return reg

    raise RuntimeError(
        "속도 모드 register를 찾지 못했습니다. "
        "LeRobot Feetech control table에서 Mode/Operating_Mode 이름 확인이 필요합니다."
    )


def write_wheel_speed(bus, values):
    """
    values 예:
    {
        "left_wheel": 200,
        "rear_wheel": 0,
        "right_wheel": 0,
    }
    """
    speed_register_candidates = [
        "Goal_Speed",
        "Goal_Velocity",
        "Moving_Speed",
        "Goal_PWM",
    ]

    for reg in speed_register_candidates:
        try:
            bus.sync_write(reg, values, normalize=False)
            print(f"[OK] speed write using register: {reg}, values={values}")
            return reg
        except Exception as e:
            print(f"[SKIP] speed register {reg} failed: {type(e).__name__}: {e}")

    raise RuntimeError(
        "속도 제어 register를 찾지 못했습니다. "
        "Goal_Speed / Goal_Velocity / Moving_Speed 후보가 모두 실패했습니다."
    )


def stop_all(bus):
    print("[STEP] Stop all wheel motors")
    zero = {name: 0 for name in WHEEL_MOTORS}
    try:
        write_wheel_speed(bus, zero)
    except Exception:
        print("[WARN] stop_all failed")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--speed", type=int, default=120)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--pause", type=float, default=0.7)
    parser.add_argument("--no-handshake", action="store_true")
    args = parser.parse_args()

    motor_names = list(WHEEL_MOTORS.keys())

    print("========================================")
    print("LeKiwi wheel motor test")
    print("Port:", args.port)
    print("Motors:")
    for name, m in WHEEL_MOTORS.items():
        print(f"  {name}: id={m.id}, model={m.model}")
    print("========================================")

    bus = FeetechMotorsBus(
        port=args.port,
        motors=WHEEL_MOTORS,
    )

    try:
        print("\n[STEP] Connect")
        bus.connect(handshake=not args.no_handshake)
        print("[OK] connected")

        print("\n[STEP] Ping wheel motor IDs")
        for name, motor in WHEEL_MOTORS.items():
            model_number = bus.ping(name)
            print(f"{name} / id={motor.id} / ping result = {model_number}")

        print("\n[STEP] Read initial positions")
        for name in motor_names:
            try_read(bus, name, "Present_Position", normalize=False)

        # 7,8,9번만 토크 ON
        print("\n[STEP] Enable torque only for wheel motors")
        bus.enable_torque(motor_names)
        print("[OK] torque enabled for 7,8,9 only")

        # 속도 모드 진입
        set_velocity_mode(bus, motor_names)

        # 안전상 처음에는 전부 정지
        stop_all(bus)
        time.sleep(args.pause)

        tests = [
            ("left_wheel", {"left_wheel": args.speed, "rear_wheel": 0, "right_wheel": 0}),
            ("rear_wheel", {"left_wheel": 0, "rear_wheel": args.speed, "right_wheel": 0}),
            ("right_wheel", {"left_wheel": 0, "rear_wheel": 0, "right_wheel": args.speed}),
        ]

        for label, values in tests:
            print("\n========================================")
            print(f"[TEST] rotate only: {label}")
            print("========================================")

            write_wheel_speed(bus, values)
            time.sleep(args.duration)

            stop_all(bus)
            time.sleep(args.pause)

        print("\n[DONE] Wheel motor test finished")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C received")

    except Exception:
        print("\n[ERROR] Test failed")
        traceback.print_exc()

    finally:
        try:
            stop_all(bus)
        except Exception:
            pass

        try:
            print("[STEP] Disable torque for wheel motors only")
            bus.disable_torque(motor_names)
        except Exception:
            pass

        try:
            print("[STEP] Disconnect")
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()