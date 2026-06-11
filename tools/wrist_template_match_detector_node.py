#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bgr_to_image_msg(frame, stamp, frame_id):
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


def parse_scales(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def video_index_from_path(value):
    value = str(value).strip()
    if value.isdigit():
        return int(value)

    real = os.path.realpath(value)
    for candidate in (value, real):
        m = re.fullmatch(r"/dev/video(\d+)", candidate)
        if m:
            return int(m.group(1))

    return value


def open_camera(camera, width, height, fps):
    cam_id = video_index_from_path(camera)

    attempts = [
        ("v4l2", cam_id, cv2.CAP_V4L2),
        ("any", cam_id, cv2.CAP_ANY),
        ("raw_v4l2", camera, cv2.CAP_V4L2),
        ("raw_any", camera, cv2.CAP_ANY),
    ]

    for name, target, backend in attempts:
        cap = cv2.VideoCapture(target, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(5):
            cap.read()

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, name, target, frame.shape

        cap.release()

    raise RuntimeError(f"failed to open camera: {camera}")


class WristTemplateMatchDetector(Node):
    def __init__(self, args):
        super().__init__("wrist_template_match_detector_node")
        self.args = args

        self.template_dir = Path(args.template_dir).expanduser()
        self.metadata_path = self.template_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.target_x_norm = float(self.metadata["target_center_norm"]["x"])
        self.target_y_norm = float(self.metadata["target_center_norm"]["y"])

        self.scales = parse_scales(args.scales)
        self.templates = self.load_templates(args.max_templates)
        if not self.templates:
            raise RuntimeError("no templates loaded")

        self.cap, open_name, open_target, shape = open_camera(
            args.camera,
            args.width,
            args.height,
            args.fps,
        )

        self.pixel_pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.debug_pub = self.create_publisher(Image, args.debug_image_topic, 10)

        self.timer = self.create_timer(1.0 / max(args.rate_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f"wrist template detector started: camera={args.camera}, opened_by={open_name}, "
            f"target={open_target}, shape={shape}, templates={len(self.templates)}, "
            f"mode={args.mode}, output={args.output_topic}"
        )

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def load_templates(self, max_templates):
        out = []

        entries = self.metadata.get("templates", [])[:max_templates]
        for e in entries:
            path = self.template_dir / e["filename"]
            img = cv2.imread(str(path))
            if img is None:
                continue

            variants = []
            for scale in self.scales:
                h0, w0 = img.shape[:2]
                tw = int(round(w0 * scale))
                th = int(round(h0 * scale))
                if tw < 20 or th < 20:
                    continue

                resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                proc = self.preprocess(resized)

                variants.append({
                    "scale": float(scale),
                    "image": resized,
                    "proc": proc,
                    "matched_wh": [tw, th],
                })

            if not variants:
                continue

            out.append({
                "path": str(path),
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "episode_index": e.get("episode_index"),
                "frame_index": e.get("frame_index"),
                "variants": variants,
            })

        return out

    def publish_status(self, data):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def preprocess(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.args.mode == "gray":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            if self.args.blur_ksize > 0:
                k = self.args.blur_ksize
                if k % 2 == 0:
                    k += 1
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            return gray

        if self.args.mode == "edge":
            if self.args.blur_ksize > 0:
                k = self.args.blur_ksize
                if k % 2 == 0:
                    k += 1
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            return cv2.Canny(gray, self.args.canny_low, self.args.canny_high)

        raise ValueError(f"unknown mode: {self.args.mode}")

    def find_best(self, frame):
        h, w = frame.shape[:2]

        sx1 = int(round(w * self.args.search_x_min_ratio))
        sx2 = int(round(w * self.args.search_x_max_ratio))
        sy1 = int(round(h * self.args.search_y_min_ratio))
        sy2 = int(round(h * self.args.search_y_max_ratio))

        sx1 = int(clamp(sx1, 0, w))
        sx2 = int(clamp(sx2, 0, w))
        sy1 = int(clamp(sy1, 0, h))
        sy2 = int(clamp(sy2, 0, h))

        search = frame[sy1:sy2, sx1:sx2]
        if search.size == 0:
            return None

        search_proc = self.preprocess(search)
        sh, sw = search_proc.shape[:2]

        best = None

        for ti, tpl in enumerate(self.templates):
            for variant in tpl["variants"]:
                tw, th = variant["matched_wh"]
                if tw >= sw or th >= sh:
                    continue

                result = cv2.matchTemplate(search_proc, variant["proc"], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                score = float(max_val)
                if best is None or score > best["score"]:
                    bx = sx1 + int(max_loc[0])
                    by = sy1 + int(max_loc[1])
                    best = {
                        "score": score,
                        "template_index": ti,
                        "template_path": tpl["path"],
                        "episode_index": tpl["episode_index"],
                        "frame_index": tpl["frame_index"],
                        "scale": float(variant["scale"]),
                        "bbox_xyxy": [bx, by, bx + tw, by + th],
                        "template_original_wh": [tpl["width"], tpl["height"]],
                        "matched_wh": [tw, th],
                    }

        return best

    def publish_detection(self, best, frame_w, frame_h, stamp):
        x1, y1, x2, y2 = best["bbox_xyxy"]
        mw, mh = best["matched_wh"]
        tw0, th0 = best["template_original_wh"]

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        match_x_norm = 2.0 * (cx / max(frame_w, 1)) - 1.0
        match_y_norm = 2.0 * (cy / max(frame_h, 1)) - 1.0

        err_x_norm = match_x_norm - self.target_x_norm
        err_y_norm = match_y_norm - self.target_y_norm

        area_scale_ratio = (mw * mh) / max(float(tw0 * th0), 1.0)

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        msg.point.x = float(err_x_norm)
        msg.point.y = float(err_y_norm)
        msg.point.z = float(area_scale_ratio)
        self.pixel_pub.publish(msg)

        self.publish_status({
            "detected": True,
            "score": round(float(best["score"]), 5),
            "scale": round(float(best["scale"]), 4),
            "x_error_norm": round(float(err_x_norm), 5),
            "y_error_norm": round(float(err_y_norm), 5),
            "area_scale_ratio": round(float(area_scale_ratio), 5),
            "bbox_xyxy": [int(v) for v in best["bbox_xyxy"]],
            "template_index": int(best["template_index"]),
            "episode_index": best["episode_index"],
            "frame_index": best["frame_index"],
            "stamp": time.time(),
        })

    def make_debug(self, frame, best):
        debug = frame.copy()
        h, w = debug.shape[:2]

        sx1 = int(round(w * self.args.search_x_min_ratio))
        sx2 = int(round(w * self.args.search_x_max_ratio))
        sy1 = int(round(h * self.args.search_y_min_ratio))
        sy2 = int(round(h * self.args.search_y_max_ratio))
        cv2.rectangle(debug, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

        tx = int(round((self.target_x_norm + 1.0) * 0.5 * w))
        ty = int(round((self.target_y_norm + 1.0) * 0.5 * h))
        cv2.drawMarker(debug, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)

        if best is not None:
            x1, y1, x2, y2 = best["bbox_xyxy"]
            color = (0, 255, 0) if best["score"] >= self.args.min_score else (0, 0, 255)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

            text = f"score={best['score']:.3f} scale={best['scale']:.2f}"
            cv2.putText(
                debug,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

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
        h, w = frame.shape[:2]

        best = self.find_best(frame)

        if best is not None and best["score"] >= self.args.min_score:
            self.publish_detection(best, w, h, stamp)
        else:
            if self.args.publish_miss:
                self.publish_status({
                    "detected": False,
                    "reason": "low_score_or_no_match",
                    "best_score": None if best is None else round(float(best["score"]), 5),
                    "stamp": time.time(),
                })

        if self.args.publish_debug_image:
            debug = self.make_debug(frame, best)
            self.debug_pub.publish(bgr_to_image_msg(debug, stamp, self.args.frame_id))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", default="/dev/video6")
    parser.add_argument("--template-dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--rate-hz", type=float, default=8.0)

    parser.add_argument("--output-topic", default="/egg/wrist_pixel")
    parser.add_argument("--status-topic", default="/egg/wrist_template/status")
    parser.add_argument("--debug-image-topic", default="/camera/wrist/template_debug_image")
    parser.add_argument("--frame-id", default="wrist_camera")

    parser.add_argument("--mode", choices=["edge", "gray"], default="edge")
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--scales", default="0.75,0.85,0.95,1.0,1.05,1.15,1.25")
    parser.add_argument("--max-templates", type=int, default=120)

    parser.add_argument("--search-x-min-ratio", type=float, default=0.00)
    parser.add_argument("--search-x-max-ratio", type=float, default=1.00)
    parser.add_argument("--search-y-min-ratio", type=float, default=0.00)
    parser.add_argument("--search-y-max-ratio", type=float, default=0.88)

    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)

    parser.add_argument("--publish-miss", action="store_true")
    parser.add_argument("--publish-debug-image", action="store_true", default=True)
    parser.add_argument("--no-debug-image", dest="publish_debug_image", action="store_false")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = WristTemplateMatchDetector(args)

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
