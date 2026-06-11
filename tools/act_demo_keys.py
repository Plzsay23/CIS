#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String


def proc_alive(proc):
    return proc is not None and proc.poll() is None


def terminate_proc(proc, name, timeout=4.0):
    if proc is None:
        return

    if proc.poll() is not None:
        return

    print(f"[STOP] {name}")
    proc.terminate()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[KILL] {name}")
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def run_pkill(pattern):
    subprocess.run(
        ["pkill", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


class ActDemoKeys(Node):
    def __init__(self, args):
        super().__init__("act_demo_keys")
        self.args = args

        self.arm_pub = self.create_publisher(String, "/dashboard/arm_cmd", 10)
        self.estop_pub = self.create_publisher(Bool, "/emergency_stop", 10)
        self.dash_cmd_pub = self.create_publisher(Twist, "/dashboard/cmd_vel", 10)
        self.auto_cmd_pub = self.create_publisher(Twist, "/auto/cmd_vel", 10)
        self.act_cmd_pub = self.create_publisher(Twist, "/act/cmd_vel", 10)

        self.act_proc = None
        self.driver_proc = None

    def publish_zero_cmd(self):
        msg = Twist()
        for pub in (self.dash_cmd_pub, self.auto_cmd_pub, self.act_cmd_pub):
            pub.publish(msg)

    def clear_estop(self):
        msg = Bool()
        msg.data = False
        self.estop_pub.publish(msg)
        print("[E-STOP] clear")

    def set_estop(self):
        msg = Bool()
        msg.data = True
        self.estop_pub.publish(msg)
        self.publish_zero_cmd()
        print("[E-STOP] set")

    def start_driver(self):
        if proc_alive(self.driver_proc):
            print("[DRIVER] already running by this controller")
            return

        if self.args.pkill_external_driver:
            run_pkill("lekiwi_base_driver_odom_node.py")

        cmd = [
            sys.executable,
            str(Path(self.args.base_driver).expanduser()),
            "--port",
            self.args.port,
            "--calibration-json",
            str(Path(self.args.calibration_json).expanduser()),
            "--arm-acceleration",
            str(self.args.arm_acceleration),
            "--arm-home-return-seconds",
            str(self.args.arm_home_return_seconds),
            "--arm-home-return-fps",
            str(self.args.arm_home_return_fps),
        ]

        print("[DRIVER] start")
        print(" ".join(cmd))

        self.driver_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(self.args.cis_root).expanduser()),
            env=os.environ.copy(),
            start_new_session=True,
        )

        time.sleep(self.args.driver_warmup_sec)

    def stop_driver(self):
        if proc_alive(self.driver_proc):
            terminate_proc(self.driver_proc, "base_driver")
            self.driver_proc = None

        if self.args.pkill_external_driver:
            run_pkill("lekiwi_base_driver_odom_node.py")

        print("[DRIVER] stopped/released")

    def publish_home(self):
        if proc_alive(self.act_proc):
            print("[HOME] ACT is running. Stop ACT first.")
            return

        if not proc_alive(self.driver_proc):
            self.start_driver()

        time.sleep(0.5)

        msg = String()
        msg.data = "arm_home"

        print("[HOME] publish arm_home")
        for _ in range(5):
            self.arm_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.12)

        time.sleep(self.args.home_wait_sec)
        print("[HOME] done")

    def start_act(self):
        if proc_alive(self.act_proc):
            print("[ACT] already running")
            return

        print("[ACT] prepare: stop keyboard drive, zero cmd, release /dev/follower")
        if self.args.pkill_keyboard_drive:
            run_pkill("keyboard_drive_node.py")

        self.publish_zero_cmd()
        time.sleep(0.2)

        self.stop_driver()
        time.sleep(0.5)

        camera_cfg = (
            "{"
            f"wrist: {{type: opencv, index_or_path: {self.args.camera}, "
            f"width: {self.args.width}, height: {self.args.height}, fps: {self.args.fps}}}"
            "}"
        )

        cmd = [
            sys.executable,
            "-m",
            "lerobot.async_inference.robot_client",
            "--robot.type=so101_follower",
            f"--robot.port={self.args.port}",
            "--robot.id=follower",
            f"--robot.cameras={camera_cfg}",
            f"--task={self.args.task}",
            f"--server_address={self.args.server_address}",
            "--policy_type=act",
            f"--pretrained_name_or_path={self.args.pretrained_name_or_path}",
            f"--policy_device={self.args.policy_device}",
            f"--actions_per_chunk={self.args.actions_per_chunk}",
            f"--chunk_size_threshold={self.args.chunk_size_threshold}",
            f"--aggregate_fn_name={self.args.aggregate_fn_name}",
            f"--debug_visualize_queue_size={self.args.debug_visualize_queue_size}",
        ]

        print("[ACT] start")
        print(" ".join(cmd))

        self.act_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(self.args.cis_root).expanduser()),
            env=os.environ.copy(),
            start_new_session=True,
        )

    def stop_act(self):
        if proc_alive(self.act_proc):
            terminate_proc(self.act_proc, "ACT robot_client")
            self.act_proc = None

        run_pkill("lerobot.async_inference.robot_client")
        print("[ACT] stopped")

    def cleanup(self):
        self.publish_zero_cmd()
        self.stop_act()
        if self.args.stop_driver_on_exit:
            self.stop_driver()

    def print_menu(self):
        print()
        print("========== ACT DEMO KEYS ==========")
        print("d : start base driver  (/dev/follower drive/home mode)")
        print("b : stop base driver   (release /dev/follower)")
        print("h : arm home / start pose return")
        print("a : START ACT          (kills keyboard drive + stops base driver)")
        print("s : STOP ACT")
        print("e : emergency stop")
        print("c : clear emergency stop")
        print("q : quit")
        print("===================================")
        print()

    def handle_key(self, ch):
        if ch == "d":
            self.start_driver()
        elif ch == "b":
            self.stop_driver()
        elif ch == "h":
            self.publish_home()
        elif ch == "a":
            self.start_act()
        elif ch == "s":
            self.stop_act()
        elif ch == "e":
            self.set_estop()
        elif ch == "c":
            self.clear_estop()
        elif ch == "q":
            return False
        else:
            print(f"[KEY] unknown: {repr(ch)}")
        return True


def read_one_key():
    return sys.stdin.read(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cis-root", default="~/CIS")
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--camera", default="/dev/video6")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)

    parser.add_argument("--base-driver", default="~/CIS/scripts/lekiwi_base_driver_odom_node.py")
    parser.add_argument("--calibration-json", default="~/CIS/config/lekiwi.json")
    parser.add_argument("--arm-acceleration", type=int, default=160)
    parser.add_argument("--arm-home-return-seconds", type=float, default=0.8)
    parser.add_argument("--arm-home-return-fps", type=int, default=50)
    parser.add_argument("--driver-warmup-sec", type=float, default=2.0)
    parser.add_argument("--home-wait-sec", type=float, default=1.2)

    parser.add_argument("--task", default="pick_egg")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--pretrained-name-or-path", default="plzsay/pick_egg")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=70)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.3)
    parser.add_argument("--aggregate-fn-name", default="weighted_average")
    parser.add_argument("--debug-visualize-queue-size", default="True")

    parser.add_argument("--pkill-external-driver", action="store_true", default=True)
    parser.add_argument("--no-pkill-external-driver", dest="pkill_external_driver", action="store_false")
    parser.add_argument("--pkill-keyboard-drive", action="store_true", default=True)
    parser.add_argument("--stop-driver-on-exit", action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = ActDemoKeys(args)
    node.print_menu()

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        running = True
        while running:
            rclpy.spin_once(node, timeout_sec=0.02)
            ch = read_one_key()
            running = node.handle_key(ch)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
