#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torchvision

# 현재 Jetson/venv의 torchvision은 video_reader backend가 비활성인 경우가 있다.
# LeRobotDataset이 video frame을 decode할 때 pyav backend를 쓰도록 강제한다.
torchvision.set_video_backend("pyav")

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_scalar(v, default=None):
    if v is None:
        return default
    if hasattr(v, "item"):
        return v.item()
    return v


def image_to_bgr(img):
    if hasattr(img, "detach"):
        arr = img.detach().cpu().numpy()
    elif hasattr(img, "convert"):
        arr = np.array(img.convert("RGB"))
    else:
        arr = np.array(img)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # LeRobot/PIL/torch image는 보통 RGB이므로 BGR로 저장
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resolve_camera_key(ds, camera_key):
    keys = list(ds.features.keys())

    candidates = [
        camera_key,
        f"observation.images.{camera_key}",
        f"observation.image.{camera_key}",
        f"observation.{camera_key}",
        f"image.{camera_key}",
    ]

    for c in candidates:
        if c in keys:
            return c

    image_keys = [k for k in keys if "image" in k.lower() or "camera" in k.lower() or camera_key in k.lower()]
    print("[ERROR] camera key not found")
    print("requested:", camera_key)
    print("\n[available image-like keys]")
    for k in image_keys:
        print(" ", k)
    raise SystemExit(1)


def make_contact_sheet(crops, out_path, thumb_w=180):
    if not crops:
        return

    thumbs = []
    for p in crops[:40]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = max(1, int(h * scale))
        img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(img)

    if not thumbs:
        return

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    max_h = max(t.shape[0] for t in thumbs)

    sheet = np.zeros((rows * max_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y = r * max_h
        x = c * thumb_w
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t

    cv2.imwrite(str(out_path), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-key", default="wrist")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--out-dir", default="~/CIS/templates/wrist_start")
    parser.add_argument("--max-frame-index", type=int, default=2)
    parser.add_argument("--max-frames-per-episode", type=int, default=3)
    parser.add_argument("--max-templates", type=int, default=120)

    # 하단 그리퍼를 빼고, 계란+판이 있는 시작 장면 영역만 template으로 저장
    parser.add_argument("--crop-x-min", type=float, default=0.10)
    parser.add_argument("--crop-x-max", type=float, default=0.90)
    parser.add_argument("--crop-y-min", type=float, default=0.05)
    parser.add_argument("--crop-y-max", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    raw_dir = out_dir / "raw_start_frames"
    template_dir = out_dir / "templates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD]", args.repo_id)
    ds = LeRobotDataset(
        args.repo_id,
        video_backend=args.video_backend,
    )

    image_key = resolve_camera_key(ds, args.camera_key)
    print("[CAMERA KEY]", image_key)
    print("[LEN]", len(ds))

    saved = 0
    saved_by_episode = {}
    template_entries = []

    for idx in range(len(ds)):
        item = ds[idx]

        ep = int(to_scalar(item.get("episode_index"), 0))
        fr = int(to_scalar(item.get("frame_index"), saved_by_episode.get(ep, 0)))

        if fr > args.max_frame_index:
            continue

        if saved_by_episode.get(ep, 0) >= args.max_frames_per_episode:
            continue

        bgr = image_to_bgr(item[image_key])
        h, w = bgr.shape[:2]

        x1 = int(round(w * args.crop_x_min))
        x2 = int(round(w * args.crop_x_max))
        y1 = int(round(h * args.crop_y_min))
        y2 = int(round(h * args.crop_y_max))

        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("invalid crop ratios")

        crop = bgr[y1:y2, x1:x2].copy()

        stem = f"ep{ep:04d}_fr{fr:03d}_idx{idx:07d}"
        raw_path = raw_dir / f"{stem}.jpg"
        tpl_path = template_dir / f"{stem}_tpl.jpg"

        cv2.imwrite(str(raw_path), bgr)
        cv2.imwrite(str(tpl_path), crop)

        template_entries.append({
            "filename": f"templates/{tpl_path.name}",
            "raw_filename": f"raw_start_frames/{raw_path.name}",
            "episode_index": ep,
            "frame_index": fr,
            "dataset_index": idx,
            "image_width": w,
            "image_height": h,
            "crop_xyxy": [x1, y1, x2, y2],
            "template_width": int(crop.shape[1]),
            "template_height": int(crop.shape[0]),
        })

        saved_by_episode[ep] = saved_by_episode.get(ep, 0) + 1
        saved += 1

        if saved >= args.max_templates:
            break

    metadata = {
        "repo_id": args.repo_id,
        "camera_key": image_key,
        "crop_ratios": {
            "x_min": args.crop_x_min,
            "x_max": args.crop_x_max,
            "y_min": args.crop_y_min,
            "y_max": args.crop_y_max,
        },
        "target_center_norm": {
            "x": 2.0 * ((args.crop_x_min + args.crop_x_max) * 0.5) - 1.0,
            "y": 2.0 * ((args.crop_y_min + args.crop_y_max) * 0.5) - 1.0,
        },
        "templates": template_entries,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    make_contact_sheet([template_dir / Path(e["filename"]).name for e in template_entries], out_dir / "contact_sheet.jpg")

    print("[SAVED]", saved)
    print("[OUT]", out_dir)
    print("[METADATA]", out_dir / "metadata.json")
    print("[CONTACT]", out_dir / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
