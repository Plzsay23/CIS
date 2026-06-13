#!/usr/bin/env python3

import argparse
import json
import math
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_camera_id(value: str):
    value = str(value).strip()
    if value.isdigit():
        return int(value)

    # /dev/wrist 같은 udev symlink는 OpenCV가 문자열 경로로 못 여는 경우가 있다.
    # /dev/videoN으로 resolve되면 정수 N으로 여는 것이 가장 안정적이다.
    resolved = os.path.realpath(value)

    for candidate in (value, resolved):
        m = re.fullmatch(r"/dev/video(\d+)", candidate)
        if m:
            return int(m.group(1))

    return resolved


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


class OpenCVWristWhiteEggDetector(Node):
    def __init__(self, args):
        super().__init__("opencv_wrist_white_egg_detector_node")
        self.args = args

        cam_id = parse_camera_id(args.camera)
        self.get_logger().info(f"opening camera: requested={args.camera}, resolved={cam_id}")

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().warn(f"CAP_V4L2 open failed: {cam_id}, retry default backend")
            self.cap.release()
            self.cap = cv2.VideoCapture(cam_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open camera: requested={args.camera}, resolved={cam_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(args.fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(args.warmup_frames):
            self.cap.read()

        self.pixel_pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.debug_pub = self.create_publisher(Image, args.debug_image_topic, 10)

        self.timer = self.create_timer(1.0 / max(args.rate_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f"wrist white egg detector started: camera={args.camera}, "
            f"{args.width}x{args.height}@{args.fps}, output={args.output_topic}"
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

    def build_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, self.args.min_v], dtype=np.uint8)
        upper = np.array([179, self.args.max_s, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 검출 허용 ROI 밖은 제거
        rx1 = int(round(w * self.args.roi_x_min_ratio))
        rx2 = int(round(w * self.args.roi_x_max_ratio))
        ry1 = int(round(h * self.args.roi_y_min_ratio))
        ry2 = int(round(h * self.args.roi_y_max_ratio))

        rx1 = int(clamp(rx1, 0, w))
        rx2 = int(clamp(rx2, 0, w))
        ry1 = int(clamp(ry1, 0, h))
        ry2 = int(clamp(ry2, 0, h))

        roi_mask = np.zeros_like(mask)
        if rx2 > rx1 and ry2 > ry1:
            roi_mask[ry1:ry2, rx1:rx2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        # 필요하면 특정 ignore ROI 제거
        if self.args.ignore_roi:
            ix1 = int(round(w * self.args.ignore_x_min_ratio))
            ix2 = int(round(w * self.args.ignore_x_max_ratio))
            iy1 = int(round(h * self.args.ignore_y_min_ratio))
            iy2 = int(round(h * self.args.ignore_y_max_ratio))

            ix1 = int(clamp(ix1, 0, w))
            ix2 = int(clamp(ix2, 0, w))
            iy1 = int(clamp(iy1, 0, h))
            iy2 = int(clamp(iy2, 0, h))

            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = 0

        if self.args.blur_ksize > 0:
            k = self.args.blur_ksize
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        kernel = np.ones((self.args.morph_ksize, self.args.morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.args.open_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.args.close_iter)

        return mask

    def score_contour(self, contour, width: int, height: int) -> Optional[dict]:
        area = float(cv2.contourArea(contour))
        area_ratio = area / float(width * height)

        if area < self.args.min_area_px:
            return None
        if area_ratio < self.args.min_area_ratio or area_ratio > self.args.max_area_ratio:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None

        aspect = float(w) / float(h)
        if aspect < self.args.min_aspect or aspect > self.args.max_aspect:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1.0:
            return None

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < self.args.min_circularity:
            return None

        cx = x + w * 0.5
        cy = y + h * 0.5

        x_norm = (cx - width * 0.5) / max(width * 0.5, 1.0)
        y_norm = (cy - height * 0.5) / max(height * 0.5, 1.0)

        center_dist = math.sqrt(x_norm * x_norm + y_norm * y_norm)

        score = (
            self.args.area_score_weight * area_ratio
            + self.args.circularity_score_weight * circularity
            - self.args.center_penalty_weight * center_dist
        )

        return {
            "contour": contour,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": area,
            "area_ratio": area_ratio,
            "aspect": aspect,
            "circularity": circularity,
            "center_px": (float(cx), float(cy)),
            "x_norm": float(x_norm),
            "y_norm": float(y_norm),
            "score": float(score),
        }

    def find_best(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        mask = self.build_mask(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for contour in contours:
            item = self.score_contour(contour, w, h)
            if item is None:
                continue
            if best is None or item["score"] > best["score"]:
                best = item

        return best, mask

    def make_debug(self, bgr: np.ndarray, mask: np.ndarray, candidate: Optional[dict]) -> np.ndarray:
        debug = bgr.copy()
        h, w = debug.shape[:2]

        # detect ROI 표시
        rx1 = int(round(w * self.args.roi_x_min_ratio))
        rx2 = int(round(w * self.args.roi_x_max_ratio))
        ry1 = int(round(h * self.args.roi_y_min_ratio))
        ry2 = int(round(h * self.args.roi_y_max_ratio))
        cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(
            debug,
            "DETECT ROI",
            (rx1, max(15, ry1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if self.args.ignore_roi:
            ix1 = int(round(w * self.args.ignore_x_min_ratio))
            ix2 = int(round(w * self.args.ignore_x_max_ratio))
            iy1 = int(round(h * self.args.ignore_y_min_ratio))
            iy2 = int(round(h * self.args.ignore_y_max_ratio))
            cv2.rectangle(debug, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
            cv2.putText(
                debug,
                "IGNORE",
                (ix1, max(15, iy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # 목표 중심선 표시
        tx = int(round(w * 0.5 * (self.args.target_x_norm + 1.0)))
        ty = int(round(h * 0.5 * (self.args.target_y_norm + 1.0)))
        cv2.drawMarker(debug, (tx, ty), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        if candidate is not None:
            x, y, bw, bh = candidate["bbox"]
            cx, cy = candidate["center_px"]
            cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(debug, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            cv2.putText(
                debug,
                f"x={candidate['x_norm']:.3f} y={candidate['y_norm']:.3f} area={candidate['area_ratio']:.4f}",
                (max(0, x), max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if self.args.show_mask_in_debug:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            debug = np.hstack([debug, mask_bgr])

        return debug

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

        candidate, mask = self.find_best(frame)

        if candidate is not None:
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.args.frame_id
            msg.point.x = float(candidate["x_norm"])
            msg.point.y = float(candidate["y_norm"])
            msg.point.z = float(candidate["area_ratio"])
            self.pixel_pub.publish(msg)

            x, y, bw, bh = candidate["bbox"]
            self.publish_status({
                "detected": True,
                "stamp": time.time(),
                "x_norm": round(float(candidate["x_norm"]), 5),
                "y_norm": round(float(candidate["y_norm"]), 5),
                "area_ratio": round(float(candidate["area_ratio"]), 6),
                "bbox_xywh": [x, y, bw, bh],
                "center_px": [
                    round(float(candidate["center_px"][0]), 2),
                    round(float(candidate["center_px"][1]), 2),
                ],
                "circularity": round(float(candidate["circularity"]), 4),
                "aspect": round(float(candidate["aspect"]), 4),
                "score": round(float(candidate["score"]), 6),
            })
        else:
            if self.args.publish_miss:
                self.publish_status({
                    "detected": False,
                    "reason": "no_white_egg_candidate",
                    "stamp": time.time(),
                })

        if self.args.publish_debug_image:
            debug = self.make_debug(frame, mask, candidate)
            self.debug_pub.publish(bgr_to_image_msg(debug, stamp, self.args.frame_id))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", default="/dev/wrist")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--rate-hz", type=float, default=15.0)
    parser.add_argument("--warmup-frames", type=int, default=10)

    parser.add_argument("--output-topic", default="/egg/wrist_pixel")
    parser.add_argument("--status-topic", default="/egg/wrist_detector/status")
    parser.add_argument("--debug-image-topic", default="/camera/wrist/egg_debug_image")
    parser.add_argument("--frame-id", default="wrist_camera")

    # 흰색 threshold
    parser.add_argument("--min-v", type=int, default=135)
    parser.add_argument("--max-s", type=int, default=95)

    # 검출 허용 ROI. 필요하면 중앙부만 쓰도록 좁힐 수 있음.
    parser.add_argument("--roi-x-min-ratio", type=float, default=0.00)
    parser.add_argument("--roi-x-max-ratio", type=float, default=1.00)
    parser.add_argument("--roi-y-min-ratio", type=float, default=0.00)
    parser.add_argument("--roi-y-max-ratio", type=float, default=1.00)

    # 특정 영역 무시
    parser.add_argument("--ignore-roi", action="store_true")
    parser.add_argument("--ignore-x-min-ratio", type=float, default=0.00)
    parser.add_argument("--ignore-x-max-ratio", type=float, default=0.00)
    parser.add_argument("--ignore-y-min-ratio", type=float, default=0.00)
    parser.add_argument("--ignore-y-max-ratio", type=float, default=0.00)

    # mask 후처리
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--morph-ksize", type=int, default=5)
    parser.add_argument("--open-iter", type=int, default=1)
    parser.add_argument("--close-iter", type=int, default=2)

    # contour filter
    parser.add_argument("--min-area-px", type=float, default=80.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0004)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-aspect", type=float, default=0.35)
    parser.add_argument("--max-aspect", type=float, default=2.80)
    parser.add_argument("--min-circularity", type=float, default=0.25)

    # candidate score
    parser.add_argument("--area-score-weight", type=float, default=2.0)
    parser.add_argument("--circularity-score-weight", type=float, default=0.5)
    parser.add_argument("--center-penalty-weight", type=float, default=0.10)

    # debug 표시용 목표 위치
    parser.add_argument("--target-x-norm", type=float, default=0.0)
    parser.add_argument("--target-y-norm", type=float, default=0.0)

    parser.add_argument("--publish-miss", action="store_true")
    parser.add_argument("--publish-debug-image", action="store_true", default=True)
    parser.add_argument("--no-debug-image", dest="publish_debug_image", action="store_false")
    parser.add_argument("--show-mask-in-debug", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = OpenCVWristWhiteEggDetector(args)

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
