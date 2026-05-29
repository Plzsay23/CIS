#!/usr/bin/env python3
import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from ultralytics import YOLO


def decode_base64_jpeg(value: str):
    raw = base64.b64decode(value)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def parse_classes(classes_arg: str | None):
    if classes_arg is None or classes_arg.strip() == "":
        return None
    return [int(x.strip()) for x in classes_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="tcp://127.0.0.1:5556")
    parser.add_argument("--model", default="yolov10n.pt")
    parser.add_argument("--cam", default="top", choices=["top", "wrist"])
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--classes", default=None, help='예: "14"이면 COCO bird만 표시')
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--save-dir", default="runs/lekiwi_yolo_test")
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    classes = parse_classes(args.classes)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.address)

    print(f"[INFO] Connected to {args.address}")
    print(f"[INFO] Camera key: {args.cam}")
    print(f"[INFO] imgsz={args.imgsz}, conf={args.conf}, device={args.device}, classes={classes}")
    print("[INFO] Ctrl+C to stop.")

    frame_idx = 0
    last_t = time.perf_counter()
    fps = 0.0

    try:
        while True:
            msg = sock.recv_string()
            obs = json.loads(msg)

            if args.cam not in obs:
                print(f"[WARN] Camera key '{args.cam}' not found. keys={list(obs.keys())}")
                continue

            img = decode_base64_jpeg(obs[args.cam])
            if img is None:
                print("[WARN] Failed to decode image")
                continue

            results = model.predict(
                source=img,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                classes=classes,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            if frame_idx % args.print_every == 0:
                names = result.names
                dets = []
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        dets.append(f"{names.get(cls_id, cls_id)}:{conf:.2f}")
                print(f"[{frame_idx}] fps={fps:.1f} detections={dets}")

            if args.view:
                cv2.imshow("lekiwi_yolo_test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.save_every > 0 and frame_idx % args.save_every == 0:
                out_path = save_dir / f"{args.cam}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), annotated)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

    finally:
        if args.view:
            cv2.destroyAllWindows()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()