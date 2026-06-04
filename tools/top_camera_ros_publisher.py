#!/usr/bin/env python3
"""Publish a camera device as a standalone ROS Image topic."""

import argparse

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class TopCameraPublisher(Node):
    def __init__(self, args) -> None:
        super().__init__("top_camera_ros_publisher")
        self.args = args
        self.publisher = self.create_publisher(Image, args.topic, 10)
        self.capture = cv2.VideoCapture(args.device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.capture.set(cv2.CAP_PROP_FPS, args.fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera device: {args.device}")

        period = 1.0 / args.fps if args.fps > 0 else 1.0 / 30.0
        self.timer = self.create_timer(period, self.publish_frame)
        self.get_logger().info(
            f"Camera topic publisher: {args.device} -> {args.topic}, "
            f"frame={args.frame_id}, rotate_180={args.rotate_180}"
        )

    def publish_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warn("Camera frame read failed")
            return

        if self.args.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

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

    def close(self) -> None:
        self.capture.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/top")
    parser.add_argument("--topic", default="/camera/top/image_raw")
    parser.add_argument("--frame-id", default="top_camera_optical_frame")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rotate-180", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = TopCameraPublisher(args)
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
