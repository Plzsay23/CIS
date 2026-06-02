#!/usr/bin/env python3
"""Convert an Isaac Sim USD or top-view image into a Nav2 occupancy map.

The preferred path uses Pixar USD Python bindings (`pxr`) to inspect the stage
and project prim bounding boxes into a conservative top-down occupancy grid.
When `pxr` is unavailable, use `--topview-image` to convert an Isaac Sim
orthographic screenshot into the same Nav2 map format.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


UNKNOWN = 205
FREE = 254
OCCUPIED = 0


@dataclass
class Rect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    label: str
    path: str

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Nav2 map.pgm/map.yaml from a USD stage or top-view image."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/home/lerobot/CIS/nav_maps/lekiwi_env3.usd",
        help="USD file path. Ignored when --topview-image is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/lerobot/CIS/nav_maps/generated",
        help="Directory for generated map files.",
    )
    parser.add_argument("--name", default="lekiwi_map", help="Output map basename.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Meters per pixel.")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.5,
        help="Meters of padding around the detected stage bounds.",
    )
    parser.add_argument(
        "--floor-keywords",
        default="floor,ground,plane,terrain,walkable",
        help="Comma-separated prim name keywords treated as free floor.",
    )
    parser.add_argument(
        "--obstacle-keywords",
        default="wall,obstacle,box,rack,shelf,door,column,table,chair,cabinet,plant,barrier",
        help="Comma-separated prim name keywords treated as occupied.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.03,
        help="Minimum z extent for a prim to be considered as geometry.",
    )
    parser.add_argument(
        "--obstacle-min-z",
        type=float,
        default=0.02,
        help="Minimum world Z that intersects the robot body envelope.",
    )
    parser.add_argument(
        "--robot-height",
        type=float,
        default=0.35,
        help="Only geometry whose bottom is below this height is projected as an obstacle.",
    )
    parser.add_argument(
        "--ignore-keywords",
        default="light,camera,ceiling,sky,visual,material,physics,collider",
        help="Comma-separated prim path keywords ignored during USD projection.",
    )
    parser.add_argument(
        "--force-free-keywords",
        default="",
        help="Comma-separated prim path keywords always treated as free.",
    )
    parser.add_argument(
        "--force-obstacle-keywords",
        default="wall,barrier,fence,gate,nest,sanran",
        help="Comma-separated prim path keywords always treated as occupied if they intersect robot height.",
    )
    parser.add_argument(
        "--unknown-as-occupied",
        action="store_true",
        help="Fill unknown cells as occupied instead of unknown.",
    )
    parser.add_argument(
        "--topview-image",
        help="Fallback: top-down PNG/JPG from Isaac Sim to convert instead of USD.",
    )
    parser.add_argument(
        "--image-origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "YAW"),
        help="Map origin for --topview-image. Default centers image at world origin.",
    )
    parser.add_argument(
        "--occupied-threshold",
        type=int,
        default=80,
        help="Pixels darker than this grayscale value become occupied in image fallback.",
    )
    parser.add_argument(
        "--free-threshold",
        type=int,
        default=210,
        help="Pixels brighter than this grayscale value become free in image fallback.",
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Write stage analysis metadata next to the map.",
    )
    return parser.parse_args()


def keywords(raw: str) -> tuple[str, ...]:
    return tuple(k.strip().lower() for k in raw.split(",") if k.strip())


def classify_prim(path: str, floor_words: Iterable[str], obstacle_words: Iterable[str]) -> str:
    lower = path.lower()
    if any(word in lower for word in obstacle_words):
        return "obstacle"
    if any(word in lower for word in floor_words):
        return "floor"
    return "unknown"


def load_stage_rects(args: argparse.Namespace) -> tuple[list[Rect], list[Rect], dict]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on Isaac/pxr install
        raise RuntimeError(
            "USD conversion requires Pixar USD Python bindings (`pxr`). "
            "Run this script with Isaac Sim's python, or use --topview-image fallback."
        ) from exc

    stage_path = str(Path(args.input).expanduser().resolve())
    stage = Usd.Stage.Open(stage_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if not meters_per_unit or meters_per_unit <= 0:
        meters_per_unit = 1.0

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    floor_words = keywords(args.floor_keywords)
    obstacle_words = keywords(args.obstacle_keywords)
    ignore_words = keywords(args.ignore_keywords)
    force_free_words = keywords(args.force_free_keywords)
    force_obstacle_words = keywords(args.force_obstacle_keywords)
    floors: list[Rect] = []
    obstacles: list[Rect] = []
    debug_prims: list[dict] = []
    prims = 0

    for prim in stage.Traverse():
        prims += 1
        if not prim.IsActive() or prim.IsAbstract():
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        try:
            bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        except Exception:
            continue
        if bound.IsEmpty():
            continue

        mn = bound.GetMin()
        mx = bound.GetMax()
        min_x = float(mn[0]) * meters_per_unit
        min_y = float(mn[1]) * meters_per_unit
        min_z = float(mn[2]) * meters_per_unit
        max_x = float(mx[0]) * meters_per_unit
        max_y = float(mx[1]) * meters_per_unit
        max_z = float(mx[2]) * meters_per_unit
        if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y, min_z, max_z)):
            continue
        if max_x <= min_x or max_y <= min_y:
            continue

        prim_path = str(prim.GetPath())
        lower_path = prim_path.lower()
        if any(word in lower_path for word in ignore_words):
            continue

        label = classify_prim(prim_path, floor_words, obstacle_words)
        if any(word in lower_path for word in force_free_words):
            label = "floor"
        if any(word in lower_path for word in force_obstacle_words):
            label = "obstacle"

        rect = Rect(min_x, min_y, max_x, max_y, label, str(prim.GetPath()))

        z_extent = max_z - min_z
        intersects_robot_height = (
            z_extent >= args.z_min
            and max_z >= args.obstacle_min_z
            and min_z <= args.robot_height
        )
        if label == "floor":
            floors.append(rect)
        elif (label == "obstacle" or label == "unknown") and intersects_robot_height:
            obstacles.append(rect)

        if len(debug_prims) < 5000:
            debug_prims.append(
                {
                    "path": prim_path,
                    "label": label,
                    "min": [min_x, min_y, min_z],
                    "max": [max_x, max_y, max_z],
                    "z_extent": z_extent,
                    "intersects_robot_height": intersects_robot_height,
                }
            )

    metadata = {
        "source": stage_path,
        "meters_per_unit": meters_per_unit,
        "prim_count": prims,
        "floor_count": len(floors),
        "obstacle_count": len(obstacles),
        "robot_height": args.robot_height,
        "obstacle_min_z": args.obstacle_min_z,
        "debug_prims": debug_prims,
    }
    return floors, obstacles, metadata


def world_bounds(rects: Iterable[Rect], padding: float) -> tuple[float, float, float, float]:
    rects = list(rects)
    if not rects:
        raise RuntimeError("No usable USD prim bounds were found.")
    min_x = min(r.min_x for r in rects) - padding
    min_y = min(r.min_y for r in rects) - padding
    max_x = max(r.max_x for r in rects) + padding
    max_y = max(r.max_y for r in rects) + padding
    return min_x, min_y, max_x, max_y


def rect_to_cells(
    rect: Rect,
    min_x: float,
    min_y: float,
    resolution: float,
    height: int,
) -> tuple[int, int, int, int]:
    x0 = int(math.floor((rect.min_x - min_x) / resolution))
    x1 = int(math.ceil((rect.max_x - min_x) / resolution))
    y0 = int(math.floor((rect.min_y - min_y) / resolution))
    y1 = int(math.ceil((rect.max_y - min_y) / resolution))
    # Occupancy map images are top-left origin. ROS map origin is bottom-left.
    row0 = height - y1
    row1 = height - y0
    return x0, row0, x1, row1


def rasterize_usd(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    floors, obstacles, metadata = load_stage_rects(args)
    all_rects = floors + obstacles
    min_x, min_y, max_x, max_y = world_bounds(all_rects, args.padding)
    width = max(1, int(math.ceil((max_x - min_x) / args.resolution)))
    height = max(1, int(math.ceil((max_y - min_y) / args.resolution)))

    fill = OCCUPIED if args.unknown_as_occupied else UNKNOWN
    grid = np.full((height, width), fill, dtype=np.uint8)

    for floor in floors:
        x0, y0, x1, y1 = rect_to_cells(floor, min_x, min_y, args.resolution, height)
        grid[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = FREE

    for obs in obstacles:
        x0, y0, x1, y1 = rect_to_cells(obs, min_x, min_y, args.resolution, height)
        grid[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = OCCUPIED

    metadata.update(
        {
            "origin": [min_x, min_y, 0.0],
            "resolution": args.resolution,
            "width_px": width,
            "height_px": height,
            "bounds_m": [min_x, min_y, max_x, max_y],
        }
    )
    return grid, metadata


def rasterize_topview(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    image_path = Path(args.topview_image).expanduser().resolve()
    gray = ImageOps.grayscale(Image.open(image_path))
    arr = np.asarray(gray, dtype=np.uint8)

    grid = np.full(arr.shape, UNKNOWN, dtype=np.uint8)
    grid[arr >= args.free_threshold] = FREE
    grid[arr <= args.occupied_threshold] = OCCUPIED
    if args.unknown_as_occupied:
        grid[grid == UNKNOWN] = OCCUPIED

    height, width = grid.shape
    if args.image_origin:
        origin = [float(args.image_origin[0]), float(args.image_origin[1]), float(args.image_origin[2])]
    else:
        origin = [
            -0.5 * width * args.resolution,
            -0.5 * height * args.resolution,
            0.0,
        ]

    metadata = {
        "source": str(image_path),
        "mode": "topview_image",
        "origin": origin,
        "resolution": args.resolution,
        "width_px": width,
        "height_px": height,
    }
    return grid, metadata


def write_map(grid: np.ndarray, metadata: dict, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pgm_path = output_dir / f"{args.name}.pgm"
    yaml_path = output_dir / f"{args.name}.yaml"
    json_path = output_dir / f"{args.name}_analysis.json"

    Image.fromarray(grid, mode="L").save(pgm_path)

    origin = metadata["origin"]
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                "mode: trinary",
                f"resolution: {args.resolution:.6g}",
                f"origin: [{origin[0]:.6g}, {origin[1]:.6g}, {origin[2]:.6g}]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "",
            ]
        ),
        encoding="ascii",
    )

    if args.debug_json:
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return pgm_path, yaml_path, json_path


def main() -> None:
    args = parse_args()
    if args.resolution <= 0:
        raise SystemExit("--resolution must be > 0")

    if args.topview_image:
        grid, metadata = rasterize_topview(args)
    else:
        grid, metadata = rasterize_usd(args)

    pgm_path, yaml_path, json_path = write_map(grid, metadata, args)
    print(f"Wrote map image: {pgm_path}")
    print(f"Wrote map yaml : {yaml_path}")
    if args.debug_json:
        print(f"Wrote analysis : {json_path}")


if __name__ == "__main__":
    main()
