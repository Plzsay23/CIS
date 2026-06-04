#!/usr/bin/env python3
"""Publish RealSense color stream as /camera/top/image_raw using serial number."""

import argparse
import time

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class TopRealSensePublisher(Node):
    def __init__(self, args):
        super().__init__("top_realsense_ros_publisher")

        self.args = args
        self.publisher = self.create_publisher(Image, args.topic, 10)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if args.serial:
            self.config.enable_device(args.serial)

        self.config.enable_stream(
            rs.stream.color,
            int(args.width),
            int(args.height),
            rs.format.bgr8,
            int(args.fps),
        )

        self.profile = self.pipeline.start(self.config)

        # 카메라 안정화용. 첫 몇 프레임은 버림.
        for _ in range(10):
            self.pipeline.wait_for_frames(timeout_ms=3000)

        period = 1.0 / float(args.fps)
        self.timer = self.create_timer(period, self.publish_frame)

        device = self.profile.get_device()
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)

        self.get_logger().info(
            f"RealSense top camera publisher started: "
            f"name={name}, serial={serial}, topic={args.topic}, "
            f"frame_id={args.frame_id}, {args.width}x{args.height}@{args.fps}"
        )

    def publish_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as e:
            self.get_logger().warn(f"RealSense frame wait failed: {e}")
            return

        color_frame = frames.get_color_frame()
        if not color_frame:
            self.get_logger().warn("No color frame from RealSense")
            return

        frame = np.asanyarray(color_frame.get_data())

        if self.args.rotate_180:
            frame = np.ascontiguousarray(frame[::-1, ::-1])

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(frame.strides[0])
        msg.data = frame.tobytes()

        self.publisher.publish(msg)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="", help="RealSense serial number. Empty = first available device.")
    parser.add_argument("--topic", default="/camera/top/image_raw")
    parser.add_argument("--frame-id", default="top_camera_optical_frame")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rotate-180", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = TopRealSensePublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()