#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def video_index_from_path(value: str):
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
        ("index_or_path_v4l2", cam_id, cv2.CAP_V4L2),
        ("index_or_path_any", cam_id, cv2.CAP_ANY),
        ("raw_path_v4l2", camera, cv2.CAP_V4L2),
        ("raw_path_any", camera, cv2.CAP_ANY),
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
            print(f"[CAMERA] opened by {name}: target={target}, frame_shape={frame.shape}")
            return cap

        cap.release()

    raise RuntimeError(f"failed to open camera: {camera}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video6")
    parser.add_argument("--model", default="/home/lerobot/CIS/yolov10n.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--device", default="0")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--out-dir", default="/home/lerobot/CIS/debug_yolo_probe")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MODEL] loading: {args.model}")
    model = YOLO(args.model)

    cap = open_camera(args.camera, args.width, args.height, args.fps)

    best_result = None
    best_frame = None
    best_count = -1

    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] frame read failed: {i}")
            continue

        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]

        count = 0 if result.boxes is None else len(result.boxes)
        print(f"[FRAME {i}] detections={count}")

        if count > best_count:
            best_count = count
            best_result = result
            best_frame = frame.copy()

        time.sleep(0.05)

    cap.release()

    if best_frame is None or best_result is None:
        print("[RESULT] no valid frame")
        return

    raw_path = out_dir / "probe_raw.jpg"
    ann_path = out_dir / "probe_annotated.jpg"
    json_path = out_dir / "probe_detections.json"

    cv2.imwrite(str(raw_path), best_frame)

    detections = []
    names = best_result.names

    if best_result.boxes is not None:
        h, w = best_frame.shape[:2]

        for box in best_result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)

            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 6),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center_px": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                "area_ratio": round(area_ratio, 6),
            })

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    annotated = best_result.plot()
    cv2.imwrite(str(ann_path), annotated)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print("\n========== YOLO DETECTIONS ==========")
    if not detections:
        print("NO DETECTION")
    else:
        for d in detections:
            print(
                f"class={d['class_id']:>2} {d['class_name']:<15} "
                f"conf={d['confidence']:.4f} "
                f"area={d['area_ratio']:.5f} "
                f"bbox={d['bbox_xyxy']}"
            )

    print("\n[SAVED]")
    print(f"raw       : {raw_path}")
    print(f"annotated : {ann_path}")
    print(f"json      : {json_path}")


if __name__ == "__main__":
    main()
