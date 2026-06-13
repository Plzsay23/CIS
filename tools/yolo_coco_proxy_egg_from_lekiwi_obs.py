#!/usr/bin/env python3
"""Publish YOLO COCO proxy detections as ground-plane egg positions."""

import argparse
import base64
import json
import math
import time

import cv2
import numpy as np
import rclpy
import zmq
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool
from ultralytics import YOLO


COCO_PROXY_CLASS_ID = 32


def decode_base64_jpeg(value: str):
    raw = base64.b64decode(value)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_base64_depth_png(value: str):
    raw = base64.b64decode(value)
    arr = np.frombuffer(raw, dtype=np.uint8)
    depth_mm = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        return None
    return depth_mm.astype(np.float32) / 1000.0


class CocoProxyEggDetector(Node):
    def __init__(self, args) -> None:
        super().__init__("coco_proxy_egg_detector")
        self.args = args
        self.pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.estop_pub = self.create_publisher(Bool, args.estop_topic, 10)
        self.model = YOLO(args.model)
        self.last_publish_time = 0.0
        self.published_once = False
        self.confirmed_positions = []

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(args.address)

        self.get_logger().info(
            f"YOLO COCO proxy -> egg detection: {args.address}/{args.cam} -> {args.output_topic}, "
            f"frame={args.frame_id}, camera_height={args.camera_height:.2f}m, "
            f"camera_pitch={args.camera_pitch_deg:.1f}deg, vertical_fov={args.vertical_fov_deg:.1f}deg, "
            f"depth_key={args.depth_key}"
        )

    def estimate_ground_position(
        self,
        image_width: int,
        image_height: int,
        center_x: float,
        bottom_y: float,
        depth_m=None,
        box_xyxy=None,
    ):
        normalized_x = (center_x - image_width * 0.5) / (image_width * 0.5)
        normalized_y = (bottom_y - image_height * 0.5) / (image_height * 0.5)

        horizontal_half_fov = math.radians(self.args.horizontal_fov_deg * 0.5)
        vertical_half_fov = math.radians(self.args.vertical_fov_deg * 0.5)
        bearing = math.atan(normalized_x * math.tan(horizontal_half_fov))
        vertical_offset = math.atan(normalized_y * math.tan(vertical_half_fov))
        ground_angle = math.radians(self.args.camera_pitch_deg) + vertical_offset

        depth_distance = None
        if depth_m is not None and box_xyxy is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)
            roi_x1 = max(0, x1 + int(box_width * 0.25))
            roi_x2 = min(depth_m.shape[1], x2 - int(box_width * 0.25))
            roi_y1 = max(0, y1 + int(box_height * 0.25))
            roi_y2 = min(depth_m.shape[0], y2)
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
                valid = roi[(roi > self.args.min_depth) & (roi < self.args.max_depth)]
                if valid.size >= self.args.min_depth_pixels:
                    z = float(np.median(valid))
                    ray_x = math.tan(bearing)
                    ray_y = math.tan(vertical_offset)
                    camera_forward = z
                    camera_down = z * ray_y
                    pitch = math.radians(self.args.camera_pitch_deg)
                    depth_distance = camera_forward * math.cos(pitch) + camera_down * math.sin(pitch)

        if ground_angle <= math.radians(1.0) or ground_angle >= math.radians(89.0):
            if depth_distance is None:
                return None

        ground_distance = (
            depth_distance
            if depth_distance is not None and depth_distance > 0.0
            else self.args.camera_height / math.tan(ground_angle)
        )
        if not self.args.min_distance <= ground_distance <= self.args.max_distance:
            return None

        x = self.args.camera_forward_offset + ground_distance * math.cos(bearing)
        y = -ground_distance * math.sin(bearing)
        return x, y, bearing, ground_distance, depth_distance is not None

    def publish_detection(
        self,
        image_width: int,
        image_height: int,
        center_x: float,
        bottom_y: float,
        confidence: float,
        depth_m=None,
        box_xyxy=None,
    ) -> None:
        now = time.monotonic()
        if self.published_once and not self.args.repeat:
            return
        if now - self.last_publish_time < self.args.publish_cooldown:
            return

        estimate = self.estimate_ground_position(
            image_width,
            image_height,
            center_x,
            bottom_y,
            depth_m=depth_m,
            box_xyxy=box_xyxy,
        )
        if estimate is None:
            self.get_logger().warn("COCO proxy candidate detected, but its ground distance could not be estimated.")
            return

        x, y, bearing, ground_distance, used_depth = estimate
        self.confirmed_positions.append((x, y))
        self.confirmed_positions = self.confirmed_positions[-self.args.min_confirmations :]
        if len(self.confirmed_positions) < self.args.min_confirmations:
            return

        x = float(np.median([position[0] for position in self.confirmed_positions]))
        y = float(np.median([position[1] for position in self.confirmed_positions]))

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        self.pub.publish(msg)
        if self.args.stop_on_detection:
            stop_msg = Bool()
            stop_msg.data = True
            self.estop_pub.publish(stop_msg)
        self.last_publish_time = now
        self.published_once = True

        self.get_logger().info(
            f"COCO proxy confidence={confidence:.2f}, bearing={math.degrees(bearing):.1f}deg "
            f"ground_distance={ground_distance:.2f}m source={'depth' if used_depth else 'camera_geometry'} "
            f"-> egg ({msg.point.x:.2f}, {msg.point.y:.2f}) "
            f"in {msg.header.frame_id}, "
            f"emergency_stop={self.args.stop_on_detection}"
        )

    def run(self) -> None:
        frame_index = 0
        while rclpy.ok():
            raw = self.sock.recv_string()
            obs = json.loads(raw)
            if self.args.cam not in obs:
                self.get_logger().warn(f"Camera key '{self.args.cam}' missing; keys={list(obs.keys())}")
                continue

            image = decode_base64_jpeg(obs[self.args.cam])
            if image is None:
                continue
            depth_m = None
            if self.args.depth_key in obs:
                depth_m = decode_base64_depth_png(obs[self.args.depth_key])

            result = self.model.predict(
                source=image,
                imgsz=self.args.imgsz,
                conf=self.args.conf,
                device=self.args.device,
                classes=[COCO_PROXY_CLASS_ID],
                verbose=False,
            )[0]

            best = None
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()
                    center_x = (float(xyxy[0]) + float(xyxy[2])) * 0.5
                    bottom_y = float(xyxy[3])
                    if best is None or confidence > best[0]:
                        best = (confidence, center_x, bottom_y, xyxy)

            if best is not None:
                self.publish_detection(
                    image.shape[1],
                    image.shape[0],
                    best[1],
                    best[2],
                    best[0],
                    depth_m=depth_m,
                    box_xyxy=best[3],
                )
            else:
                self.confirmed_positions.clear()

            if self.args.view:
                cv2.imshow("coco_proxy_egg", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            rclpy.spin_once(self, timeout_sec=0.0)

    def close(self) -> None:
        if self.args.view:
            cv2.destroyAllWindows()
        self.sock.close()
        self.ctx.term()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="tcp://127.0.0.1:5556")
    parser.add_argument("--model", default="/home/lerobot/CIS/yolov10n.pt")
    parser.add_argument("--cam", default="top", choices=["top", "wrist"])
    parser.add_argument("--depth-key", default="top_depth")
    parser.add_argument("--output-topic", default="/egg_detection")
    parser.add_argument("--estop-topic", default="/emergency_stop")
    parser.add_argument("--stop-on-detection", action="store_true")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--camera-height", type=float, default=0.55)
    parser.add_argument("--camera-pitch-deg", type=float, default=40.0)
    parser.add_argument("--camera-forward-offset", type=float, default=0.0)
    parser.add_argument("--horizontal-fov-deg", type=float, default=70.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=42.5)
    parser.add_argument("--min-distance", type=float, default=0.15)
    parser.add_argument("--max-distance", type=float, default=4.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--min-depth-pixels", type=int, default=20)
    parser.add_argument("--min-confirmations", type=int, default=3)
    parser.add_argument("--publish-cooldown", type=float, default=2.0)
    parser.add_argument("--repeat", action="store_true", help="Publish repeated detections for multi-egg testing.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CocoProxyEggDetector(args)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
