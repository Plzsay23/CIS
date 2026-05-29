#!/usr/bin/env python

import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    robot_config = LeKiwiClientConfig(remote_ip="100.105.112.48", id="lekiwi")
    teleop_arm_config = SO100LeaderConfig(port="/dev/leader", id="leader")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop-only loop...")
    print("No robot.get_observation(), no camera receiving, no Rerun logging.")

    try:
        while True:
            t0 = time.perf_counter()

            arm_action = leader_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)

            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

            robot.send_action(action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting teleop-only loop.")

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass

        try:
            leader_arm.disconnect()
        except Exception:
            pass

        try:
            keyboard.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()