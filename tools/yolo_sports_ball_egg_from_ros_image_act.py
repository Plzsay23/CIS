#!/usr/bin/env python3
"""ROS Image -> YOLO sports ball class -> /egg_detection.

This version is ACT-pipeline safe: it does NOT publish /emergency_stop.
The detector semantically treats COCO sports ball as an egg.
"""

import argparse
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

SPORTS_BALL_CLASS_ID = 32


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in {"bgr8", "rgb8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
    if msg.encoding == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.copy()


class SportsBallEggFromRosImage(Node):
    def __init__(self, args):
        super().__init__("sports_ball_egg_from_ros_image_act")
        self.args = args
        self.model = YOLO(args.model)
        self.pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.create_subscription(Image, args.image_topic, self.on_image, 5)
        self.last_infer_time = 0.0
        self.last_publish_time = 0.0
        self.published_once = False
        self.confirmed_positions = []
        self.get_logger().info(
            f"YOLO sports-ball-as-egg ROS detector: {args.image_topic} -> {args.output_topic}, "
            f"model={args.model}, device={args.device}, no emergency_stop publishing"
        )

    def estimate_ground_position(self, image_width: int, image_height: int, center_x: float, bottom_y: float):
        normalized_x = (center_x - image_width * 0.5) / (image_width * 0.5)
        normalized_y = (bottom_y - image_height * 0.5) / (image_height * 0.5)

        horizontal_half_fov = math.radians(self.args.horizontal_fov_deg * 0.5)
        vertical_half_fov = math.radians(self.args.vertical_fov_deg * 0.5)
        bearing = math.atan(normalized_x * math.tan(horizontal_half_fov))
        vertical_offset = math.atan(normalized_y * math.tan(vertical_half_fov))
        ground_angle = math.radians(self.args.camera_pitch_deg) + vertical_offset

        if ground_angle <= math.radians(1.0) or ground_angle >= math.radians(89.0):
            return None

        ground_distance = self.args.camera_height / math.tan(ground_angle)
        if not self.args.min_distance <= ground_distance <= self.args.max_distance:
            return None

        x = self.args.camera_forward_offset + ground_distance * math.cos(bearing)
        y = -ground_distance * math.sin(bearing)
        return x, y, bearing, ground_distance

    def publish_detection(self, stamp, image_width, image_height, center_x, bottom_y, confidence):
        now = time.monotonic()
        if self.published_once and not self.args.repeat:
            return
        if now - self.last_publish_time < self.args.publish_cooldown:
            return

        estimate = self.estimate_ground_position(image_width, image_height, center_x, bottom_y)
        if estimate is None:
            self.get_logger().warn("sports ball detected, but ground distance estimate was rejected")
            return

        x, y, bearing, ground_distance = estimate
        self.confirmed_positions.append((x, y))
        self.confirmed_positions = self.confirmed_positions[-self.args.min_confirmations:]
        if len(self.confirmed_positions) < self.args.min_confirmations:
            return

        x = float(np.median([p[0] for p in self.confirmed_positions]))
        y = float(np.median([p[1] for p in self.confirmed_positions]))

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        self.pub.publish(msg)

        self.last_publish_time = now
        self.published_once = True
        self.get_logger().info(
            f"egg detected via sports ball: conf={confidence:.2f}, bearing={math.degrees(bearing):.1f}deg, "
            f"distance={ground_distance:.2f}m -> ({x:.2f}, {y:.2f}) in {msg.header.frame_id}"
        )

    def on_image(self, msg: Image):
        now = time.monotonic()
        if now - self.last_infer_time < 1.0 / max(self.args.infer_hz, 0.1):
            return
        self.last_infer_time = now

        try:
            image = image_msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"bad image: {e}")
            return

        result = self.model.predict(
            source=image,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            device=self.args.device,
            classes=[SPORTS_BALL_CLASS_ID],
            verbose=False,
        )[0]

        best: Optional[Tuple[float, float, float]] = None
        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                center_x = (float(xyxy[0]) + float(xyxy[2])) * 0.5
                bottom_y = float(xyxy[3])
                if best is None or confidence > best[0]:
                    best = (confidence, center_x, bottom_y)

        if best is not None:
            self.publish_detection(msg.header.stamp, image.shape[1], image.shape[0], best[1], best[2], best[0])
        else:
            self.confirmed_positions.clear()

        if self.args.view:
            cv2.imshow("sports_ball_as_egg_ros", result.plot())
            cv2.waitKey(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-topic", default="/camera/top/image_raw")
    p.add_argument("--output-topic", default="/egg_detection")
    p.add_argument("--model", default="/home/lerobot/CIS/yolov10n.pt")
    p.add_argument("--frame-id", default="base_link")
    p.add_argument("--camera-height", type=float, default=0.55)
    p.add_argument("--camera-pitch-deg", type=float, default=40.0)
    p.add_argument("--camera-forward-offset", type=float, default=0.0)
    p.add_argument("--horizontal-fov-deg", type=float, default=70.0)
    p.add_argument("--vertical-fov-deg", type=float, default=42.5)
    p.add_argument("--min-distance", type=float, default=0.15)
    p.add_argument("--max-distance", type=float, default=4.0)
    p.add_argument("--min-confirmations", type=int, default=3)
    p.add_argument("--publish-cooldown", type=float, default=2.0)
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0")
    p.add_argument("--infer-hz", type=float, default=10.0)
    p.add_argument("--view", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SportsBallEggFromRosImage(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if args.view:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
