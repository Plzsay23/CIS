#!/usr/bin/env python3

import argparse
import json
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


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


class RealSenseWhiteEggDetector(Node):
    def __init__(self, args):
        super().__init__("realsense_white_egg_detector_node")

        self.args = args

        self.egg_pub = self.create_publisher(PointStamped, args.output_topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.image_pub = self.create_publisher(Image, args.image_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, args.debug_image_topic, 10)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if args.serial:
            self.config.enable_device(args.serial)

        self.config.enable_stream(
            rs.stream.color,
            args.width,
            args.height,
            rs.format.bgr8,
            args.fps,
        )
        self.config.enable_stream(
            rs.stream.depth,
            args.width,
            args.height,
            rs.format.z16,
            args.fps,
        )

        self.align = rs.align(rs.stream.color)

        self.profile = self.pipeline.start(self.config)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()

        device = self.profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.device_serial = device.get_info(rs.camera_info.serial_number)

        # RealSense 안정화용 초기 프레임 discard
        for _ in range(args.warmup_frames):
            try:
                self.pipeline.wait_for_frames(timeout_ms=3000)
            except Exception:
                pass

        self.timer = self.create_timer(1.0 / max(args.rate_hz, 1.0), self.on_timer)
        self.last_detect_time = 0.0

        self.get_logger().info(
            "white egg detector started: "
            f"name={self.device_name}, serial={self.device_serial}, "
            f"output={args.output_topic}, image={args.image_topic}, "
            f"{args.width}x{args.height}@{args.fps}"
        )

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def publish_status(self, data: dict):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.status_pub.publish(msg)

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.args.frame_timeout_ms)
            frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            color = np.asanyarray(color_frame.get_data())
            return color, depth_frame
        except Exception as e:
            self.get_logger().warn(f"RealSense frame wait failed: {e}")
            return None, None

    def build_white_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 흰색: saturation 낮고 value 높은 영역.
        # 조명이 어두우면 --min-v 낮추고, 반사/노이즈가 많으면 --max-s 낮춘다.
        lower = np.array([0, 0, self.args.min_v], dtype=np.uint8)
        upper = np.array([179, self.args.max_s, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        if self.args.blur_ksize > 0:
            k = self.args.blur_ksize
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        kernel = np.ones((self.args.morph_ksize, self.args.morph_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.args.open_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.args.close_iter)

        mask = self.apply_ignore_roi(mask)

        return mask

    def apply_ignore_roi(self, mask: np.ndarray) -> np.ndarray:
        if not self.args.ignore_roi:
            return mask

        h, w = mask.shape[:2]

        x1 = int(round(w * self.args.ignore_x_min_ratio))
        x2 = int(round(w * self.args.ignore_x_max_ratio))
        y1 = int(round(h * self.args.ignore_y_min_ratio))
        y2 = int(round(h * self.args.ignore_y_max_ratio))

        x1 = int(clamp(x1, 0, w))
        x2 = int(clamp(x2, 0, w))
        y1 = int(clamp(y1, 0, h))
        y2 = int(clamp(y2, 0, h))

        if x2 <= x1 or y2 <= y1:
            return mask

        mask[y1:y2, x1:x2] = 0
        return mask

    def score_contour(self, contour, width: int, height: int) -> Optional[dict]:
        area = float(cv2.contourArea(contour))
        area_ratio = area / float(width * height)

        if area < self.args.min_area_px or area > self.args.max_area_px:
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

        # 화면 중앙에 가까운 후보 + 면적 큰 후보를 우선
        dx = (cx - width * 0.5) / max(width * 0.5, 1.0)
        dy = (cy - height * 0.5) / max(height * 0.5, 1.0)
        center_dist = math.sqrt(dx * dx + dy * dy)

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
            "score": float(score),
        }

    def find_best_candidate(self, bgr: np.ndarray) -> Tuple[Optional[dict], np.ndarray]:
        h, w = bgr.shape[:2]
        mask = self.build_white_mask(bgr)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for contour in contours:
            item = self.score_contour(contour, w, h)
            if item is None:
                continue
            if best is None or item["score"] > best["score"]:
                best = item

        return best, mask

    def depth_median_around(self, depth_frame, cx: float, cy: float) -> Optional[float]:
        r = int(self.args.depth_roi_radius)
        x0 = int(round(cx))
        y0 = int(round(cy))

        vals = []
        for yy in range(y0 - r, y0 + r + 1):
            for xx in range(x0 - r, x0 + r + 1):
                if xx < 0 or yy < 0 or xx >= self.args.width or yy >= self.args.height:
                    continue
                d = float(depth_frame.get_distance(xx, yy))
                if self.args.min_depth_m <= d <= self.args.max_depth_m:
                    vals.append(d)

        if not vals:
            return None

        return float(np.median(np.array(vals, dtype=np.float32)))

    def publish_detection(self, candidate: dict, depth_m: float, stamp):
        cx, cy = candidate["center_px"]

        # RealSense optical frame 기준:
        # deproject 결과 point[0] = camera right(+), point[1] = down(+), point[2] = forward(+)
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics,
            [float(cx), float(cy)],
            float(depth_m),
        )

        camera_right_m = float(point_3d[0])
        forward_m = float(point_3d[2])

        # ROS base_link 관례상 y는 left가 +인 경우가 많음.
        # RealSense x는 right가 +이므로 기본은 부호를 뒤집어 left+로 맞춘다.
        lateral_m = -camera_right_m
        if self.args.no_lateral_flip:
            lateral_m = camera_right_m

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.frame_id
        msg.point.x = forward_m
        msg.point.y = lateral_m
        msg.point.z = float(candidate["area_ratio"])
        self.egg_pub.publish(msg)

        self.last_detect_time = time.monotonic()

        x, y, w, h = candidate["bbox"]
        self.publish_status({
            "detected": True,
            "stamp": time.time(),
            "frame_id": self.args.frame_id,
            "forward_m": round(forward_m, 4),
            "lateral_m": round(lateral_m, 4),
            "camera_right_m": round(camera_right_m, 4),
            "depth_m": round(depth_m, 4),
            "area_ratio": round(float(candidate["area_ratio"]), 6),
            "area_px": round(float(candidate["area"]), 1),
            "aspect": round(float(candidate["aspect"]), 4),
            "circularity": round(float(candidate["circularity"]), 4),
            "bbox_xywh": [x, y, w, h],
            "center_px": [round(float(cx), 2), round(float(cy), 2)],
            "score": round(float(candidate["score"]), 6),
        })

    def make_debug_image(self, bgr: np.ndarray, mask: np.ndarray, candidate: Optional[dict], depth_m: Optional[float]) -> np.ndarray:
        debug = bgr.copy()

        if self.args.ignore_roi:
            img_h, img_w = debug.shape[:2]
            ix1 = int(round(img_w * self.args.ignore_x_min_ratio))
            ix2 = int(round(img_w * self.args.ignore_x_max_ratio))
            iy1 = int(round(img_h * self.args.ignore_y_min_ratio))
            iy2 = int(round(img_h * self.args.ignore_y_max_ratio))
            ix1 = int(clamp(ix1, 0, img_w))
            ix2 = int(clamp(ix2, 0, img_w))
            iy1 = int(clamp(iy1, 0, img_h))
            iy2 = int(clamp(iy2, 0, img_h))
            if ix2 > ix1 and iy2 > iy1:
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

        if candidate is not None:
            x, y, w, h = candidate["bbox"]
            cx, cy = candidate["center_px"]
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            text = f"egg area={candidate['area_ratio']:.4f}"
            if depth_m is not None:
                text += f" depth={depth_m:.2f}m"
            cv2.putText(
                debug,
                text,
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
        bgr, depth_frame = self.get_frames()
        if bgr is None or depth_frame is None:
            return

        stamp = self.get_clock().now().to_msg()

        # raw image도 같이 publish해서 기존 확인 명령을 유지 가능하게 함
        self.image_pub.publish(bgr_to_image_msg(bgr, stamp, self.args.image_frame_id))

        candidate, mask = self.find_best_candidate(bgr)

        depth_m = None
        if candidate is not None:
            cx, cy = candidate["center_px"]
            depth_m = self.depth_median_around(depth_frame, cx, cy)

            if depth_m is not None:
                self.publish_detection(candidate, depth_m, stamp)
            else:
                self.publish_status({
                    "detected": False,
                    "reason": "candidate_found_but_no_valid_depth",
                    "stamp": time.time(),
                    "area_ratio": round(float(candidate["area_ratio"]), 6),
                    "center_px": [
                        round(float(cx), 2),
                        round(float(cy), 2),
                    ],
                })
        else:
            if self.args.publish_miss:
                self.publish_status({
                    "detected": False,
                    "reason": "no_white_blob_candidate",
                    "stamp": time.time(),
                })

        if self.args.publish_debug_image:
            debug = self.make_debug_image(bgr, mask, candidate, depth_m)
            self.debug_image_pub.publish(bgr_to_image_msg(debug, stamp, self.args.image_frame_id))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--serial", default="", help="RealSense serial number")
    parser.add_argument("--width", type=int, default=424)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--frame-timeout-ms", type=int, default=2000)
    parser.add_argument("--warmup-frames", type=int, default=10)

    parser.add_argument("--output-topic", default="/egg_detection")
    parser.add_argument("--status-topic", default="/egg_detector/status")
    parser.add_argument("--image-topic", default="/camera/top/image_raw")
    parser.add_argument("--debug-image-topic", default="/camera/top/egg_debug_image")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--image-frame-id", default="top_camera_optical_frame")

    # 흰색 threshold
    parser.add_argument("--min-v", type=int, default=145)
    parser.add_argument("--max-s", type=int, default=85)

    # mask 후처리
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--morph-ksize", type=int, default=5)
    parser.add_argument("--open-iter", type=int, default=1)
    parser.add_argument("--close-iter", type=int, default=2)

    # contour filter
    parser.add_argument("--min-area-px", type=float, default=80.0)
    parser.add_argument("--max-area-px", type=float, default=30000.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0006)
    parser.add_argument("--max-area-ratio", type=float, default=0.30)
    parser.add_argument("--min-aspect", type=float, default=0.45)
    parser.add_argument("--max-aspect", type=float, default=2.20)
    parser.add_argument("--min-circularity", type=float, default=0.35)

    # score
    parser.add_argument("--area-score-weight", type=float, default=2.0)
    parser.add_argument("--circularity-score-weight", type=float, default=0.6)
    parser.add_argument("--center-penalty-weight", type=float, default=0.15)

    # depth
    parser.add_argument("--depth-roi-radius", type=int, default=4)
    parser.add_argument("--min-depth-m", type=float, default=0.12)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--no-lateral-flip", action="store_true")

    parser.add_argument("--publish-miss", action="store_true")
    parser.add_argument("--publish-debug-image", action="store_true", default=True)
    parser.add_argument("--no-debug-image", dest="publish_debug_image", action="store_false")
    parser.add_argument("--show-mask-in-debug", action="store_true")

    # 하단 중앙 흰색 그리퍼/마운트 오탐 방지용 ROI 제거.
    # ratio 기준이라 640x480, 424x240 모두 같은 위치에 적용된다.
    parser.add_argument("--ignore-roi", action="store_true")
    parser.add_argument("--ignore-x-min-ratio", type=float, default=0.37)
    parser.add_argument("--ignore-x-max-ratio", type=float, default=0.62)
    parser.add_argument("--ignore-y-min-ratio", type=float, default=0.52)
    parser.add_argument("--ignore-y-max-ratio", type=float, default=1.00)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = RealSenseWhiteEggDetector(args)

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
