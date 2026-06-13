#!/usr/bin/env python3
"""Detect a COCO proxy class from a ROS Image topic without commanding the robot."""

import argparse

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


COCO_PROXY_CLASS_ID = 32


def image_message_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def bgr_to_image_message(image: np.ndarray, source: Image) -> Image:
    msg = Image()
    msg.header = source.header
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(image.strides[0])
    msg.data = image.tobytes()
    return msg


class CocoProxyTopicDetector(Node):
    def __init__(self, args) -> None:
        super().__init__("coco_proxy_topic_detector")
        self.args = args
        self.model = YOLO(args.model)
        self.processing = False
        self.annotated_pub = self.create_publisher(Image, args.annotated_topic, 10)
        self.detection_pub = self.create_publisher(PointStamped, args.detection_topic, 10)
        self.create_subscription(Image, args.image_topic, self.on_image, 10)
        self.get_logger().info(
            f"COCO proxy detector test: {args.image_topic} -> {args.annotated_topic}, "
            f"detection={args.detection_topic}"
        )

    def on_image(self, msg: Image) -> None:
        if self.processing:
            return
        self.processing = True
        try:
            image = image_message_to_bgr(msg)
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
                    center_y = (float(xyxy[1]) + float(xyxy[3])) * 0.5
                    if best is None or confidence > best[0]:
                        best = (confidence, center_x, center_y)

            if best is not None:
                detection = PointStamped()
                detection.header = msg.header
                detection.point.x = best[1]
                detection.point.y = best[2]
                detection.point.z = best[0]
                self.detection_pub.publish(detection)
                self.get_logger().info(
                    f"COCO proxy detected: confidence={best[0]:.2f}, "
                    f"pixel=({best[1]:.0f}, {best[2]:.0f})"
                )

            self.annotated_pub.publish(bgr_to_image_message(result.plot(), msg))
        except Exception as exc:
            self.get_logger().error(f"Image detection failed: {exc}")
        finally:
            self.processing = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default="/camera/top/image_raw")
    parser.add_argument("--annotated-topic", default="/camera/top/yolo_annotated")
    parser.add_argument("--detection-topic", default="/coco_proxy_detection")
    parser.add_argument("--model", default="/home/lerobot/CIS/yolov10n.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CocoProxyTopicDetector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
