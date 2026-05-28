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


def safe_write(bus, motor, register, value):
    try:
        bus.write(register, motor, value, normalize=False)
        print(f"[OK] write {motor}.{register} = {value}")
        return True
    except Exception as e:
        print(f"[SKIP] write {motor}.{register} failed: {type(e).__name__}: {e}")
        return False


def safe_read(bus, motor, register):
    try:
        value = bus.read(register, motor, normalize=False)
        print(f"[READ] {motor}.{register} = {value}")
        return value
    except Exception as e:
        print(f"[SKIP] read {motor}.{register} failed: {type(e).__name__}: {e}")
        return None


def write_velocity(bus, values):
    """
    STS3215에서는 현재 로그 기준 Goal_Velocity가 맞다.
    """
    bus.sync_write("Goal_Velocity", values, normalize=False)
    print(f"[CMD] Goal_Velocity = {values}")


def stop_all(bus):
    zero = {name: 0 for name in WHEEL_MOTORS}
    try:
        write_velocity(bus, zero)
    except Exception as e:
        print(f"[WARN] stop_all failed: {type(e).__name__}: {e}")


def read_status(bus, motor_names):
    for name in motor_names:
        safe_read(bus, name, "Present_Position")
        safe_read(bus, name, "Present_Velocity")
        safe_read(bus, name, "Present_Speed")
        safe_read(bus, name, "Present_Load")
        safe_read(bus, name, "Torque_Enable")


def configure_velocity_mode(bus, motor_names):
    print("\n[STEP] Disable torque and unlock wheel motors")
    bus.disable_torque(motor_names)

    print("\n[STEP] Set Operating_Mode = VELOCITY while torque is disabled")
    for name in motor_names:
        safe_write(bus, name, "Operating_Mode", OperatingMode.VELOCITY.value)

    print("\n[STEP] Optional acceleration settings")
    for name in motor_names:
        safe_write(bus, name, "Maximum_Acceleration", 254)
        safe_write(bus, name, "Acceleration", 254)

    print("\n[STEP] Enable torque after velocity mode is configured")
    bus.enable_torque(motor_names)

    print("\n[STEP] Confirm mode/torque")
    for name in motor_names:
        safe_read(bus, name, "Operating_Mode")
        safe_read(bus, name, "Torque_Enable")


def run_one_motor_test(bus, motor_names, target_motor, speed, duration, sample_dt):
    print("\n========================================")
    print(f"[TEST] {target_motor}: +{speed}")
    print("========================================")

    values = {name: 0 for name in motor_names}
    values[target_motor] = speed
    write_velocity(bus, values)

    t0 = time.time()
    while time.time() - t0 < duration:
        safe_read(bus, target_motor, "Present_Position")
        safe_read(bus, target_motor, "Present_Velocity")
        safe_read(bus, target_motor, "Present_Speed")
        time.sleep(sample_dt)

    stop_all(bus)
    time.sleep(0.5)

    print("\n========================================")
    print(f"[TEST] {target_motor}: -{speed}")
    print("========================================")

    values = {name: 0 for name in motor_names}
    values[target_motor] = -speed
    write_velocity(bus, values)

    t0 = time.time()
    while time.time() - t0 < duration:
        safe_read(bus, target_motor, "Present_Position")
        safe_read(bus, target_motor, "Present_Velocity")
        safe_read(bus, target_motor, "Present_Speed")
        time.sleep(sample_dt)

    stop_all(bus)
    time.sleep(0.7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--speed", type=int, default=800)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--sample-dt", type=float, default=0.25)
    parser.add_argument("--no-handshake", action="store_true")
    args = parser.parse_args()

    motor_names = list(WHEEL_MOTORS.keys())

    print("========================================")
    print("LeKiwi wheel motor velocity diagnostic")
    print("Port:", args.port)
    print("Speed:", args.speed)
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

        print("\n[STEP] Initial status")
        read_status(bus, motor_names)

        configure_velocity_mode(bus, motor_names)

        print("\n[STEP] Force stop before test")
        stop_all(bus)
        time.sleep(0.7)

        for name in motor_names:
            run_one_motor_test(
                bus=bus,
                motor_names=motor_names,
                target_motor=name,
                speed=args.speed,
                duration=args.duration,
                sample_dt=args.sample_dt,
            )

        print("\n[DONE] Diagnostic finished")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C received")

    except Exception:
        print("\n[ERROR] Diagnostic failed")
        traceback.print_exc()

    finally:
        print("\n[FINAL] Stop motors")
        try:
            stop_all(bus)
        except Exception:
            pass

        print("[FINAL] Disable torque for wheel motors only")
        try:
            bus.disable_torque(motor_names)
        except Exception:
            pass

        print("[FINAL] Disconnect")
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()