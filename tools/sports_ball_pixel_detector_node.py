#!/usr/bin/env python3

import argparse
import json
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


SPORTS_BALL_CLASS_ID = 32


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    h = int(msg.height)
    w = int(msg.width)
    enc = msg.encoding.lower()

    if enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        if enc == "rgb8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    if enc in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        return np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class SportsBallPixelDetector(Node):
    def __init__(self, args):
        super().__init__("sports_ball_pixel_detector_node")

        self.args = args
        self.model = YOLO(args.model)

        self.last_pub_time = 0.0
        self.last_detection_time = 0.0

        self.pixel_pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)

        self.sub = self.create_subscription(
            Image,
            args.image_topic,
            self.on_image,
            10,
        )

        self.get_logger().info(
            "sports_ball_pixel_detector_node started: "
            f"image_topic={args.image_topic}, output_topic={args.output_topic}, "
            f"status_topic={args.status_topic}, model={args.model}, "
            f"class_id={args.class_id}, conf={args.conf}, device={args.device}"
        )

    def publish_status(self, data: dict):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def select_best_detection(self, result, width: int, height: int) -> Optional[Tuple[float, float, float, float, float]]:
        if result.boxes is None or len(result.boxes) == 0:
            return None

        best = None
        best_score = -1.0

        cx_img = width * 0.5
        cy_img = height * 0.5

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id != int(self.args.class_id):
                continue

            if conf < self.args.conf:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            bx = (x1 + x2) * 0.5
            by = (y1 + y2) * 0.5

            area_ratio = (bw * bh) / float(width * height)

            # 중앙에 가까운 sports ball을 우선.
            dx = (bx - cx_img) / max(cx_img, 1.0)
            dy = (by - cy_img) / max(cy_img, 1.0)
            center_penalty = (dx * dx + dy * dy) ** 0.5

            score = conf + self.args.area_weight * area_ratio - self.args.center_weight * center_penalty

            if score > best_score:
                best_score = score
                best = (x1, y1, x2, y2, conf)

        return best

    def on_image(self, msg: Image):
        now = time.monotonic()

        if self.args.max_rate > 0:
            min_dt = 1.0 / self.args.max_rate
            if now - self.last_pub_time < min_dt:
                return

        try:
            frame = image_msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"image conversion failed: {e}")
            return

        h, w = frame.shape[:2]

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.args.imgsz,
                conf=self.args.conf,
                device=self.args.device,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        det = self.select_best_detection(results[0], w, h)

        if det is None:
            if self.args.publish_miss:
                self.publish_status({
                    "detected": False,
                    "stamp": time.time(),
                    "image_topic": self.args.image_topic,
                })
            return

        x1, y1, x2, y2, conf = det

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        # -1~+1 normalized pixel offset.
        # x_norm: 왼쪽 -, 오른쪽 +
        # y_norm: 위쪽 -, 아래쪽 +
        x_norm = (cx - (w * 0.5)) / max(w * 0.5, 1.0)
        y_norm = (cy - (h * 0.5)) / max(h * 0.5, 1.0)
        area_ratio = (bw * bh) / float(w * h)

        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.args.frame_id
        out.point.x = float(x_norm)
        out.point.y = float(y_norm)
        out.point.z = float(area_ratio)
        self.pixel_pub.publish(out)

        self.publish_status({
            "detected": True,
            "stamp": time.time(),
            "image_topic": self.args.image_topic,
            "frame_id": self.args.frame_id,
            "class_id": int(self.args.class_id),
            "class_name": "sports ball",
            "conf": round(float(conf), 4),
            "image_width": int(w),
            "image_height": int(h),
            "bbox_xyxy": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ],
            "bbox_width": round(float(bw), 2),
            "bbox_height": round(float(bh), 2),
            "center_px": [
                round(float(cx), 2),
                round(float(cy), 2),
            ],
            "x_norm": round(float(x_norm), 5),
            "y_norm": round(float(y_norm), 5),
            "area_ratio": round(float(area_ratio), 6),
        })

        self.last_pub_time = now
        self.last_detection_time = now


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-topic", default="/camera/wrist/image_raw")
    parser.add_argument("--output-topic", default="/egg/wrist_pixel")
    parser.add_argument("--status-topic", default="/egg/wrist_status")
    parser.add_argument("--frame-id", default="wrist_camera_frame")

    parser.add_argument("--model", default="/home/lerobot/CIS/yolov10n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--class-id", type=int, default=SPORTS_BALL_CLASS_ID)

    parser.add_argument("--max-rate", type=float, default=10.0)
    parser.add_argument("--publish-miss", action="store_true")

    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument("--center-weight", type=float, default=0.15)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = SportsBallPixelDetector(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
