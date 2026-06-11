#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


DEFAULT_CLASS_IDS = [32, 47]  # 32=sports ball, 47=apple


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bgr_to_image_msg(frame: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(frame.strides[0])
    msg.data = frame.tobytes()
    return msg


def parse_class_ids(value: str):
    if value is None:
        return list(DEFAULT_CLASS_IDS)

    ids = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))

    if not ids:
        return list(DEFAULT_CLASS_IDS)

    return ids


def video_index_from_path(value: str) -> Optional[int]:
    value = str(value)
    real = os.path.realpath(value)

    for candidate in (value, real):
        m = re.fullmatch(r"/dev/video(\d+)", candidate)
        if m:
            return int(m.group(1))

    if value.isdigit():
        return int(value)

    return None


def open_camera(camera: str, width: int, height: int, fps: int):
    idx = video_index_from_path(camera)
    real = os.path.realpath(camera)

    attempts = []
    attempts.append(("path_v4l2", camera, cv2.CAP_V4L2))
    attempts.append(("path_any", camera, cv2.CAP_ANY))
    if real != camera:
        attempts.append(("realpath_v4l2", real, cv2.CAP_V4L2))
        attempts.append(("realpath_any", real, cv2.CAP_ANY))
    if idx is not None:
        attempts.append(("index_v4l2", idx, cv2.CAP_V4L2))
        attempts.append(("index_any", idx, cv2.CAP_ANY))

    last = []
    for name, target, backend in attempts:
        cap = cv2.VideoCapture(target, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name, target, frame.shape

        cap.release()
        last.append(f"{name}:{target}")

    raise RuntimeError(f"failed to open camera={camera}, tried={last}")


class YoloWristSportsBallDetector(Node):
    def __init__(self, args):
        super().__init__("yolo_wrist_sports_ball_detector_node")
        self.args = args

        self.model = YOLO(args.model)
        self.class_ids = parse_class_ids(args.classes)

        self.cap, open_name, open_target, shape = open_camera(
            args.camera,
            args.width,
            args.height,
            args.fps,
        )

        for _ in range(args.warmup_frames):
            self.cap.read()

        self.pixel_pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.debug_pub = self.create_publisher(Image, args.debug_image_topic, 10)

        self.timer = self.create_timer(1.0 / max(args.rate_hz, 1.0), self.on_timer)

        self.last_detection_time = 0.0
        self.confirmed = []

        self.get_logger().info(
            f"YOLO wrist sports-ball detector started: camera={args.camera}, "
            f"opened_by={open_name}, target={open_target}, shape={shape}, "
            f"model={args.model}, classes={self.class_ids}, output={args.output_topic}"
        )

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def publish_status(self, data: dict):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def publish_pixel(self, frame_w: int, frame_h: int, box_xyxy, conf: float, cls_id: int, cls_name: str, stamp):
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        x_norm = (cx - frame_w * 0.5) / max(frame_w * 0.5, 1.0)
        y_norm = (cy - frame_h * 0.5) / max(frame_h * 0.5, 1.0)
        area_ratio = (bw * bh) / max(float(frame_w * frame_h), 1.0)

        # 너무 작은 박스는 튐
        if area_ratio < self.args.min_area_ratio:
            return False

        # 너무 큰 박스는 그리퍼/근접 오탐 가능성
        if area_ratio > self.args.max_area_ratio:
            return False

        self.confirmed.append((x_norm, y_norm, area_ratio, conf))
        self.confirmed = self.confirmed[-self.args.min_confirmations :]

        if len(self.confirmed) < self.args.min_confirmations:
            return False

        x_norm = float(np.median([v[0] for v in self.confirmed]))
        y_norm = float(np.median([v[1] for v in self.confirmed]))
        area_ratio = float(np.median([v[2] for v in self.confirmed]))
        conf = float(np.median([v[3] for v in self.confirmed]))

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        msg.point.x = x_norm
        msg.point.y = y_norm
        msg.point.z = area_ratio
        self.pixel_pub.publish(msg)

        self.last_detection_time = time.monotonic()

        self.publish_status({
            "detected": True,
            "stamp": time.time(),
            "x_norm": round(x_norm, 5),
            "y_norm": round(y_norm, 5),
            "area_ratio": round(area_ratio, 6),
            "confidence": round(conf, 4),
            "class_id": int(cls_id),
            "class_name": str(cls_name),
            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center_px": [round(cx, 1), round(cy, 1)],
        })

        return True

    def on_timer(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.publish_status({
                "detected": False,
                "reason": "camera_read_failed",
                "stamp": time.time(),
            })
            return

        if frame.shape[1] != self.args.width or frame.shape[0] != self.args.height:
            frame = cv2.resize(frame, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)

        stamp = self.get_clock().now().to_msg()
        frame_h, frame_w = frame.shape[:2]

        result = self.model.predict(
            source=frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            device=self.args.device,
            classes=self.class_ids,
            verbose=False,
        )[0]

        best = None
        if result.boxes is not None:
            for box in result.boxes:
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())
                cls_name = result.names.get(cls_id, str(cls_id))
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(float(frame_w * frame_h), 1.0)

                if area_ratio < self.args.min_area_ratio or area_ratio > self.args.max_area_ratio:
                    continue

                # confidence 우선. 여러 class가 동시에 잡히면 가장 확신 높은 후보를 계란으로 사용.
                if best is None or conf > best[0]:
                    best = (conf, xyxy, cls_id, cls_name)

        detected = False
        if best is not None:
            detected = self.publish_pixel(frame_w, frame_h, best[1], best[0], best[2], best[3], stamp)
        else:
            self.confirmed.clear()

        if not detected and self.args.publish_miss:
            self.publish_status({
                "detected": False,
                "reason": "no_sports_ball_candidate",
                "stamp": time.time(),
            })

        if self.args.publish_debug_image:
            debug = result.plot()

            # 목표 중심 표시
            tx = int(round(frame_w * 0.5 * (self.args.target_x_norm + 1.0)))
            ty = int(round(frame_h * 0.5 * (self.args.target_y_norm + 1.0)))
            cv2.drawMarker(
                debug,
                (tx, ty),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )

            self.debug_pub.publish(bgr_to_image_msg(debug, stamp, self.args.frame_id))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", default="/dev/video6")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=8)

    parser.add_argument("--model", default="/home/lerobot/CIS/yolov10n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--classes",
        default="32,47",
        help="Comma-separated COCO class ids to accept as egg candidates. 32=sports ball, 47=apple, 49=orange.",
    )

    parser.add_argument("--output-topic", default="/egg/wrist_pixel")
    parser.add_argument("--status-topic", default="/egg/wrist_yolo/status")
    parser.add_argument("--debug-image-topic", default="/camera/wrist/yolo_debug_image")
    parser.add_argument("--frame-id", default="wrist_camera")

    parser.add_argument("--min-confirmations", type=int, default=2)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-area-ratio", type=float, default=0.50)

    parser.add_argument("--target-x-norm", type=float, default=0.0)
    parser.add_argument("--target-y-norm", type=float, default=0.0)

    parser.add_argument("--publish-miss", action="store_true")
    parser.add_argument("--publish-debug-image", action="store_true", default=True)
    parser.add_argument("--no-debug-image", dest="publish_debug_image", action="store_false")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = YoloWristSportsBallDetector(args)

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
