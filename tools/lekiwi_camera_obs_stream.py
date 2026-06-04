#!/usr/bin/env python3
"""Capture one camera and publish LeKiwi-style observations over ZMQ."""

import argparse
import base64
import json
import subprocess
import time

import cv2
import numpy as np
import zmq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera-key", default="top")
    parser.add_argument("--address", default="tcp://*:5556")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--rotate-180", action="store_true")
    parser.add_argument("--reopen-delay", type=float, default=1.0)
    parser.add_argument("--use-realsense", action="store_true")
    parser.add_argument("--ros-topic", default="")
    parser.add_argument("--ros-frame-id", default="top_camera_optical_frame")
    parser.add_argument("--depth-key", default="top_depth")
    parser.add_argument("--depth-ros-topic", default="")
    parser.add_argument("--depth-ros-frame-id", default="top_camera_optical_frame")
    return parser.parse_args()


def find_realsense_color_camera():
    for index in range(20):
        dev = f"/dev/video{index}"
        try:
            props = subprocess.run(
                ["udevadm", "info", "--query=property", f"--name={dev}"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout
        except FileNotFoundError:
            props = ""
        if "RealSense" not in props:
            continue

        fmt = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--get-fmt-video"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        if any(token in fmt for token in ("'YUYV'", "'MJPG'", "'RGB3'", "'BGR3'")):
            return dev
    return None


def resolve_device(requested):
    if requested != "auto":
        return requested
    return find_realsense_color_camera()


def open_capture(requested_device, width, height, fps):
    modes = [(width, height, fps), (640, 480, 30.0), (640, 480, 15.0), (424, 240, 30.0)]
    fourccs = ["YUYV", "MJPG", None]

    device = resolve_device(requested_device)
    if not device:
        raise RuntimeError("No RealSense color camera device found")
    for mode_width, mode_height, mode_fps in modes:
        for fourcc in fourccs:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode_height)
            cap.set(cv2.CAP_PROP_FPS, mode_fps)

            if not cap.isOpened():
                cap.release()
                continue

            for _ in range(15):
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(
                        f"[INFO] Opened camera {device} "
                        f"{mode_width}x{mode_height}@{mode_fps:g}"
                        f"{' ' + fourcc if fourcc else ''}",
                        flush=True,
                    )
                    return cap, device
                time.sleep(0.05)
            cap.release()

    raise RuntimeError(f"Failed to open readable camera stream from: {requested_device} -> {device}")


class RealSenseCapture:
    def __init__(self, width, height, fps):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(fps))
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = (
            self.profile.get_device()
            .first_depth_sensor()
            .get_depth_scale()
        )

    def read(self):
        frames = self.pipeline.wait_for_frames(5000)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return False, None, None
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(self.depth_scale)
        return True, color, depth_m

    def release(self):
        self.pipeline.stop()


def open_realsense_capture(width, height, fps):
    cap = RealSenseCapture(width, height, fps)
    for _ in range(15):
        ok, frame, depth_m = cap.read()
        if ok and frame is not None and depth_m is not None:
            print(f"[INFO] Opened RealSense color+depth {width}x{height}@{fps:g}", flush=True)
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError("Failed to open readable RealSense color+depth stream")


def encode_depth_png(depth_m):
    depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm * 1000.0, 0.0, 65535.0).astype(np.uint16)
    encoded_ok, buffer = cv2.imencode(".png", depth_mm)
    if not encoded_ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def main():
    args = parse_args()
    cap = None
    active_device = None
    ros_node = None
    ros_pub = None
    depth_ros_pub = None

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.address)

    print(f"[INFO] Camera observation stream binding: {args.address}", flush=True)
    if args.ros_topic:
        import rclpy
        from sensor_msgs.msg import Image

        rclpy.init(args=None)
        ros_node = rclpy.create_node("lekiwi_camera_obs_stream")
        ros_pub = ros_node.create_publisher(Image, args.ros_topic, 10)
        ros_node.get_logger().info(
            f"Camera ROS image topic: {args.ros_topic}, frame={args.ros_frame_id}"
        )
        if args.depth_ros_topic:
            depth_ros_pub = ros_node.create_publisher(Image, args.depth_ros_topic, 10)
            ros_node.get_logger().info(
                f"Camera ROS depth topic: {args.depth_ros_topic}, frame={args.depth_ros_frame_id}"
            )

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    failed_reads = 0
    try:
        while True:
            started = time.monotonic()
            if cap is None:
                try:
                    if args.use_realsense:
                        cap = open_realsense_capture(args.width, args.height, args.fps)
                        active_device = "realsense"
                    else:
                        cap, active_device = open_capture(args.device, args.width, args.height, args.fps)
                    print(
                        f"[INFO] Camera observation stream: {active_device} -> {args.address}, "
                        f"key={args.camera_key}, rotate_180={args.rotate_180}",
                        flush=True,
                    )
                    failed_reads = 0
                except RuntimeError as exc:
                    print(f"[WARN] {exc}; retrying", flush=True)
                    time.sleep(args.reopen_delay)
                    continue

            if args.use_realsense:
                ok, frame, depth_m = cap.read()
            else:
                ok, frame = cap.read()
                depth_m = None
            if not ok:
                failed_reads += 1
                print("[WARN] Camera frame read failed", flush=True)
                if failed_reads >= 30:
                    print("[WARN] Reopening camera stream", flush=True)
                    cap.release()
                    cap = None
                    active_device = None
                    failed_reads = 0
                time.sleep(0.1)
                continue
            failed_reads = 0

            if args.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                if depth_m is not None:
                    depth_m = cv2.rotate(depth_m, cv2.ROTATE_180)

            encoded_ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            if encoded_ok:
                message = {
                    args.camera_key: base64.b64encode(buffer).decode("ascii"),
                    "timestamp": time.time(),
                }
                if depth_m is not None:
                    depth_value = encode_depth_png(depth_m)
                    if depth_value is not None:
                        message[args.depth_key] = depth_value
                        message[f"{args.depth_key}_unit"] = "mm_png_uint16"
                socket.send_string(json.dumps(message))

            if ros_pub is not None:
                msg = Image()
                msg.header.stamp = ros_node.get_clock().now().to_msg()
                msg.header.frame_id = args.ros_frame_id
                msg.height = int(frame.shape[0])
                msg.width = int(frame.shape[1])
                msg.encoding = "bgr8"
                msg.is_bigendian = 0
                msg.step = int(frame.strides[0])
                msg.data = frame.tobytes()
                ros_pub.publish(msg)

                if depth_ros_pub is not None and depth_m is not None:
                    depth_msg = Image()
                    depth_msg.header.stamp = msg.header.stamp
                    depth_msg.header.frame_id = args.depth_ros_frame_id
                    depth_msg.height = int(depth_m.shape[0])
                    depth_msg.width = int(depth_m.shape[1])
                    depth_msg.encoding = "32FC1"
                    depth_msg.is_bigendian = 0
                    depth_msg.step = int(depth_m.strides[0])
                    depth_msg.data = depth_m.astype(np.float32).tobytes()
                    depth_ros_pub.publish(depth_msg)

                rclpy.spin_once(ros_node, timeout_sec=0.0)

            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if ros_node is not None:
            ros_node.destroy_node()
            rclpy.shutdown()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
