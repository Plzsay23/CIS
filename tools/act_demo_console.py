#!/usr/bin/env python3

import argparse
import os
import select
import shlex
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


def run_pkill(pattern: str):
    subprocess.run(
        ["pkill", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def proc_alive(proc):
    return proc is not None and proc.poll() is None


def terminate_proc(proc, name: str, timeout: float = 4.0):
    if proc is None or proc.poll() is not None:
        return

    print(f"[STOP] {name}")

    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=1.5)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()


class ActDemoConsole(Node):
    def __init__(self, args):
        super().__init__("act_demo_console")
        self.args = args

        self.cmd_pub = self.create_publisher(Twist, "/safe_cmd_vel", 10)
        self.arm_pub = self.create_publisher(String, "/dashboard/arm_cmd", 10)
        self.estop_pub = self.create_publisher(Bool, "/emergency_stop", 10)

        self.driver_proc = None
        self.act_proc = None
        self.driver_log_file = None

        self.speed = args.linear_speed
        self.angular_speed = args.angular_speed
        self.last_twist = Twist()
        self.last_cmd_time = 0.0
        self.drive_enabled = True

        self.log_dir = Path(args.log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def print_menu(self):
        print()
        print("========== ACT DEMO CONSOLE ==========")
        print("w/s : forward/back")
        print("a/d : strafe left/right")
        print("z/c : rotate left/right")
        print("space : stop base")
        print("r/f : speed up/down")
        print("")
        print("h : arm home / start pose")
        print("p : START ACT client")
        print("k : STOP ACT client and return to drive mode")
        print("")
        print("u : restart base driver")
        print("b : stop base driver")
        print("e : emergency stop")
        print("n : clear emergency stop")
        print("q : quit")
        print("======================================")
        print(f"[SPEED] linear={self.speed:.3f}, angular={self.angular_speed:.3f}")
        print()

    def publish_zero(self):
        msg = Twist()
        self.last_twist = msg
        self.cmd_pub.publish(msg)

    def set_estop(self, value: bool):
        msg = Bool()
        msg.data = bool(value)
        self.estop_pub.publish(msg)

        if value:
            self.publish_zero()
            print("[E-STOP] ON")
        else:
            print("[E-STOP] OFF")

    def start_driver(self):
        if proc_alive(self.driver_proc):
            print("[DRIVER] already running")
            return

        print("[DRIVER] cleanup old driver")
        run_pkill("lekiwi_base_driver_odom_node.py")
        time.sleep(0.8)

        driver_path = str(Path(self.args.base_driver).expanduser())
        calib_path = str(Path(self.args.calibration_json).expanduser())

        cmd = [
            sys.executable,
            driver_path,
            "--port", self.args.port,
            "--calibration-json", calib_path,
            "--arm-acceleration", str(self.args.arm_acceleration),
            "--arm-home-return-seconds", str(self.args.arm_home_return_seconds),
            "--arm-home-return-fps", str(self.args.arm_home_return_fps),
        ]

        log_path = self.log_dir / "base_driver.log"
        self.driver_log_file = open(log_path, "ab", buffering=0)

        print("[DRIVER] start")
        print("[DRIVER LOG]", log_path)
        print(shlex.join(cmd))

        self.driver_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(self.args.cis_root).expanduser()),
            env=os.environ.copy(),
            stdout=self.driver_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        time.sleep(self.args.driver_warmup_sec)
        self.drive_enabled = True
        self.publish_zero()

    def stop_driver(self):
        self.publish_zero()
        self.drive_enabled = False

        if proc_alive(self.driver_proc):
            terminate_proc(self.driver_proc, "base_driver", timeout=3.0)

        self.driver_proc = None
        run_pkill("lekiwi_base_driver_odom_node.py")
        time.sleep(0.5)

        if self.driver_log_file is not None:
            try:
                self.driver_log_file.close()
            except Exception:
                pass
            self.driver_log_file = None

        print("[DRIVER] stopped/released /dev/follower")

    def home_arm(self):
        if proc_alive(self.act_proc):
            print("[HOME] ACT running. Press k first.")
            return

        if not proc_alive(self.driver_proc):
            self.start_driver()

        msg = String()
        msg.data = "arm_home"

        print("[HOME] arm_home")
        for _ in range(8):
            self.arm_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.03)
            time.sleep(0.10)

        time.sleep(self.args.home_wait_sec)
        print("[HOME] done")

    def start_act(self):
        if proc_alive(self.act_proc):
            print("[ACT] already running")
            return

        print("[ACT] stop base driver and release /dev/follower")
        self.publish_zero()
        time.sleep(0.2)
        self.stop_driver()

        camera_cfg = (
            "{"
            f"wrist: {{type: opencv, index_or_path: \"{self.args.camera}\", "
            f"width: {self.args.width}, height: {self.args.height}, fps: {self.args.fps}}}"
            "}"
        )

        cmd = [
            sys.executable,
            "-m", "lerobot.async_inference.robot_client",
            "--robot.type=so101_follower",
            f"--robot.port={self.args.port}",
            "--robot.id=follower",
            f"--robot.cameras={camera_cfg}",
            f"--task={self.args.task}",
            f"--server_address={self.args.server_address}",
            f"--policy_type={self.args.policy_type}",
            f"--pretrained_name_or_path={self.args.pretrained_name_or_path}",
            f"--policy_device={self.args.policy_device}",
            f"--actions_per_chunk={self.args.actions_per_chunk}",
            f"--chunk_size_threshold={self.args.chunk_size_threshold}",
            f"--aggregate_fn_name={self.args.aggregate_fn_name}",
        ]

        print("[ACT] start client")
        print(shlex.join(cmd))

        self.act_proc = subprocess.Popen(
            cmd,
            cwd=str(Path(self.args.cis_root).expanduser()),
            env=os.environ.copy(),
            start_new_session=True,
        )

    def stop_act(self, restart_driver=True):
        if proc_alive(self.act_proc):
            terminate_proc(self.act_proc, "ACT robot_client", timeout=4.0)

        self.act_proc = None
        run_pkill("lerobot.async_inference.robot_client")
        time.sleep(0.5)
        print("[ACT] stopped")

        if restart_driver:
            self.start_driver()

    def set_drive_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        if proc_alive(self.act_proc):
            print("[DRIVE] ACT running. Press k first.")
            return

        if not proc_alive(self.driver_proc):
            print("[DRIVE] driver not running. Press u.")
            return

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)

        self.last_twist = msg
        self.last_cmd_time = time.time()
        self.drive_enabled = True
        self.cmd_pub.publish(msg)

    def publish_drive_tick(self):
        if not self.drive_enabled:
            return
        if not proc_alive(self.driver_proc):
            return
        if proc_alive(self.act_proc):
            return

        now = time.time()

        if now - self.last_cmd_time > self.args.command_hold_sec:
            self.publish_zero()
            return

        self.cmd_pub.publish(self.last_twist)

    def handle_key(self, ch):
        if ch == "w":
            self.set_drive_cmd(vx=self.speed)
        elif ch == "s":
            self.set_drive_cmd(vx=-self.speed)
        elif ch == "a":
            self.set_drive_cmd(vy=self.speed)
        elif ch == "d":
            self.set_drive_cmd(vy=-self.speed)
        elif ch == "z":
            self.set_drive_cmd(wz=self.angular_speed)
        elif ch == "c":
            self.set_drive_cmd(wz=-self.angular_speed)
        elif ch == " ":
            self.publish_zero()
            print("[DRIVE] stop")

        elif ch == "r":
            self.speed = min(self.args.max_linear_speed, self.speed + self.args.speed_step)
            self.angular_speed = min(self.args.max_angular_speed, self.angular_speed + self.args.angular_step)
            print(f"[SPEED] linear={self.speed:.3f}, angular={self.angular_speed:.3f}")
        elif ch == "f":
            self.speed = max(self.args.min_linear_speed, self.speed - self.args.speed_step)
            self.angular_speed = max(self.args.min_angular_speed, self.angular_speed - self.args.angular_step)
            print(f"[SPEED] linear={self.speed:.3f}, angular={self.angular_speed:.3f}")

        elif ch == "h":
            self.home_arm()
        elif ch == "p":
            self.start_act()
        elif ch == "k":
            self.stop_act(restart_driver=True)

        elif ch == "u":
            self.stop_driver()
            self.start_driver()
        elif ch == "b":
            self.stop_driver()

        elif ch == "e":
            self.set_estop(True)
        elif ch == "n":
            self.set_estop(False)

        elif ch == "q":
            return False
        else:
            print(f"[KEY] unknown: {repr(ch)}")

        return True

    def cleanup(self):
        print("[CLEANUP]")
        self.publish_zero()
        self.stop_act(restart_driver=False)
        self.stop_driver()


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

    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--pretrained-name-or-path", default="plzsay/pick_egg")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--task", default="pick_egg")
    parser.add_argument("--actions-per-chunk", type=int, default=70)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.3)
    parser.add_argument("--aggregate-fn-name", default="weighted_average")

    parser.add_argument("--linear-speed", type=float, default=0.055)
    parser.add_argument("--angular-speed", type=float, default=0.45)
    parser.add_argument("--min-linear-speed", type=float, default=0.015)
    parser.add_argument("--max-linear-speed", type=float, default=0.120)
    parser.add_argument("--min-angular-speed", type=float, default=0.15)
    parser.add_argument("--max-angular-speed", type=float, default=1.20)
    parser.add_argument("--speed-step", type=float, default=0.010)
    parser.add_argument("--angular-step", type=float, default=0.10)
    parser.add_argument("--command-hold-sec", type=float, default=0.35)

    parser.add_argument("--log-dir", default="~/CIS/logs/act_demo_console")
    parser.add_argument("--no-auto-start-driver", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # 데모 방해 프로세스 정리
    run_pkill("keyboard_drive_node.py")
    run_pkill("wrist_template_match_detector_node")
    run_pkill("yolo_wrist_sports_ball_detector_node")
    run_pkill("opencv_wrist_white_egg_detector_node")
    run_pkill("egg_approach_node")

    rclpy.init()
    node = ActDemoConsole(args)
    node.print_menu()

    if not args.no_auto_start_driver:
        node.start_driver()

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        running = True

        while running:
            rclpy.spin_once(node, timeout_sec=0.01)

            readable, _, _ = select.select([sys.stdin], [], [], 0.03)
            if readable:
                ch = sys.stdin.read(1)
                running = node.handle_key(ch)

            node.publish_drive_tick()

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
