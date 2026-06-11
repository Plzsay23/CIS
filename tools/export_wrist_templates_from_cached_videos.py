#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import av
import cv2
import numpy as np


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_dataset_root(repo_id: str) -> Path:
    repo_id = repo_id.strip("/")
    direct = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if direct.exists():
        return direct

    # 혹시 cache 구조가 다를 때 fallback search
    name = repo_id.split("/")[-1]
    roots = [
        Path.home() / ".cache" / "huggingface" / "lerobot",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]

    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend([p for p in root.rglob("*") if p.is_dir() and name in p.name])

    if candidates:
        candidates = sorted(candidates, key=lambda p: len(str(p)))
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find local dataset cache for {repo_id}. "
        f"Expected: {direct}"
    )


def find_camera_videos(root: Path, camera_key: str):
    all_videos = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]

    camera_key_l = camera_key.lower()
    selected = []
    for p in all_videos:
        s = str(p).lower()
        if camera_key_l in s:
            selected.append(p)

    selected = sorted(selected)

    if not selected:
        print("[ERROR] no camera videos found")
        print("[ROOT]", root)
        print("[CAMERA KEY]", camera_key)
        print("\n[ALL VIDEO CANDIDATES]")
        for p in all_videos[:80]:
            print(" ", p)
        raise SystemExit(1)

    return selected


def parse_episode_index(path: Path, fallback: int) -> int:
    s = str(path)

    patterns = [
        r"episode[_-](\d+)",
        r"episodes?[/_-](\d+)",
        r"ep[_-]?(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    return fallback


def crop_image(bgr, crop_x_min, crop_x_max, crop_y_min, crop_y_max):
    h, w = bgr.shape[:2]

    x1 = int(round(w * crop_x_min))
    x2 = int(round(w * crop_x_max))
    y1 = int(round(h * crop_y_min))
    y2 = int(round(h * crop_y_max))

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"invalid crop: {(x1, y1, x2, y2)} for image {w}x{h}")

    return bgr[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def decode_start_frames(video_path: Path, max_frame_index: int):
    frames = []

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i > max_frame_index:
                break
            bgr = frame.to_ndarray(format="bgr24")
            frames.append((i, bgr))
    finally:
        container.close()

    return frames


def make_contact_sheet(image_paths, out_path, thumb_w=180):
    thumbs = []

    for p in image_paths[:60]:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = max(1, int(round(h * scale)))
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
    parser.add_argument("--out-dir", default="~/CIS/templates/wrist_start")

    parser.add_argument("--max-frame-index", type=int, default=2)
    parser.add_argument("--max-frames-per-video", type=int, default=3)
    parser.add_argument("--max-templates", type=int, default=120)

    parser.add_argument("--crop-x-min", type=float, default=0.10)
    parser.add_argument("--crop-x-max", type=float, default=0.90)
    parser.add_argument("--crop-y-min", type=float, default=0.05)
    parser.add_argument("--crop-y-max", type=float, default=0.75)

    args = parser.parse_args()

    root = find_dataset_root(args.repo_id)
    videos = find_camera_videos(root, args.camera_key)

    out_dir = Path(args.out_dir).expanduser()
    raw_dir = out_dir / "raw_start_frames"
    template_dir = out_dir / "templates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    print("[ROOT]", root)
    print("[VIDEOS]", len(videos))
    for p in videos[:10]:
        print(" ", p)

    saved = 0
    entries = []
    template_paths = []

    for video_i, video_path in enumerate(videos):
        if saved >= args.max_templates:
            break

        ep = parse_episode_index(video_path, fallback=video_i)

        try:
            frames = decode_start_frames(video_path, args.max_frame_index)
        except Exception as e:
            print(f"[WARN] failed to decode {video_path}: {e}")
            continue

        if not frames:
            print(f"[WARN] no frames: {video_path}")
            continue

        used_in_video = 0

        for frame_index, bgr in frames:
            if saved >= args.max_templates:
                break
            if used_in_video >= args.max_frames_per_video:
                break

            crop, crop_xyxy = crop_image(
                bgr,
                args.crop_x_min,
                args.crop_x_max,
                args.crop_y_min,
                args.crop_y_max,
            )

            stem = f"ep{ep:04d}_fr{frame_index:03d}_vid{video_i:04d}"
            raw_path = raw_dir / f"{stem}.jpg"
            tpl_path = template_dir / f"{stem}_tpl.jpg"

            cv2.imwrite(str(raw_path), bgr)
            cv2.imwrite(str(tpl_path), crop)

            h, w = bgr.shape[:2]

            entries.append({
                "filename": f"templates/{tpl_path.name}",
                "raw_filename": f"raw_start_frames/{raw_path.name}",
                "source_video": str(video_path),
                "episode_index": ep,
                "frame_index": frame_index,
                "video_index": video_i,
                "image_width": int(w),
                "image_height": int(h),
                "crop_xyxy": crop_xyxy,
                "template_width": int(crop.shape[1]),
                "template_height": int(crop.shape[0]),
            })

            template_paths.append(tpl_path)
            saved += 1
            used_in_video += 1

    metadata = {
        "repo_id": args.repo_id,
        "camera_key": args.camera_key,
        "dataset_root": str(root),
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
        "templates": entries,
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    make_contact_sheet(template_paths, out_dir / "contact_sheet.jpg")

    print("[SAVED]", saved)
    print("[OUT]", out_dir)
    print("[METADATA]", out_dir / "metadata.json")
    print("[CONTACT]", out_dir / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
