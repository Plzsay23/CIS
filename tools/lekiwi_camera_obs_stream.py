#!/usr/bin/env python3
"""Capture one camera and publish LeKiwi-style observations over ZMQ."""

import argparse
import base64
import json
import subprocess
import time

import cv2
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
    parser.add_argument("--ros-topic", default="")
    parser.add_argument("--ros-frame-id", default="top_camera_optical_frame")
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


def main():
    args = parse_args()
    cap = None
    active_device = None
    ros_node = None
    ros_pub = None

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

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    failed_reads = 0
    try:
        while True:
            started = time.monotonic()
            if cap is None:
                try:
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

            ok, frame = cap.read()
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
