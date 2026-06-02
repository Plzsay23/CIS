#!/usr/bin/env python3
"""Create a hand-tuned Nav2 map for the LeKiwi Isaac arena.

This is intentionally separate from usd_to_nav2_map.py. The USD bounding-box
projection is too coarse for this long poultry-house style scene, so this
script draws the static navigation obstacles from measured/visual layout
features: outer walls, continuous feeder/waterer rows, nesting blocks, and
aisles that LeKiwi should be able to use.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


FREE = 254
OCCUPIED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hand-tuned LeKiwi Nav2 map.")
    parser.add_argument("--output-dir", default="/home/lerobot/CIS/nav_maps/generated")
    parser.add_argument("--name", default="lekiwi_map_v2")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--min-x", type=float, default=-41.0)
    parser.add_argument("--max-x", type=float, default=61.0)
    parser.add_argument("--min-y", type=float, default=-12.0)
    parser.add_argument("--max-y", type=float, default=12.0)
    parser.add_argument(
        "--y-scale",
        type=float,
        default=2.0,
        help="Stretch obstacle layout around y=0 to make aisles wider.",
    )
    return parser.parse_args()


class Map:
    def __init__(self, args: argparse.Namespace) -> None:
        self.res = args.resolution
        self.min_x = args.min_x
        self.min_y = args.min_y
        self.max_x = args.max_x
        self.max_y = args.max_y
        self.width = int(math.ceil((self.max_x - self.min_x) / self.res))
        self.height = int(math.ceil((self.max_y - self.min_y) / self.res))
        self.grid = np.full((self.height, self.width), FREE, dtype=np.uint8)

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        px = int(round((x - self.min_x) / self.res))
        py = int(round((y - self.min_y) / self.res))
        return px, self.height - 1 - py

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        left, top = self.world_to_px(min(x0, x1), max(y0, y1))
        right, bottom = self.world_to_px(max(x0, x1), min(y0, y1))
        left = max(0, min(self.width - 1, left))
        right = max(0, min(self.width - 1, right))
        top = max(0, min(self.height - 1, top))
        bottom = max(0, min(self.height - 1, bottom))
        self.grid[top : bottom + 1, left : right + 1] = OCCUPIED

    def circle(self, x: float, y: float, radius: float) -> None:
        cx, cy = self.world_to_px(x, y)
        r = max(1, int(round(radius / self.res)))
        y0 = max(0, cy - r)
        y1 = min(self.height - 1, cy + r)
        x0 = max(0, cx - r)
        x1 = min(self.width - 1, cx + r)
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        self.grid[y0 : y1 + 1, x0 : x1 + 1][mask] = OCCUPIED

    def dashed_blocks(
        self,
        start_x: float,
        end_x: float,
        y: float,
        block_w: float,
        block_h: float,
        gap: float,
    ) -> None:
        x = start_x
        while x + block_w <= end_x:
            self.rect(x, y - block_h / 2.0, x + block_w, y + block_h / 2.0)
            x += block_w + gap


def sy(y: float, scale: float) -> float:
    return y * scale


def draw_arena(m: Map, y_scale: float) -> None:
    left_cut_x = -34.0
    right_cut_x = 54.0

    # Outer house walls.
    m.rect(-40.3, sy(-5.55, y_scale), 60.4, sy(-5.35, y_scale))
    m.rect(-40.3, sy(5.35, y_scale), 60.4, sy(5.55, y_scale))
    m.rect(-40.3, sy(-5.55, y_scale), -40.05, sy(5.55, y_scale))
    m.rect(60.15, sy(-5.55, y_scale), 60.4, sy(5.55, y_scale))

    # Long feeder/waterer rails. Thin lines keep aisles usable while still
    # preventing Nav2 from planning directly through the structures.
    for y in (4.55, 2.25, 0.35, -2.20, -4.55):
        ys = sy(y, y_scale)
        m.rect(left_cut_x, ys - 0.06, right_cut_x, ys + 0.06)

    # Continuous small feeder/waterer heads along rows.
    for y in (4.05, 1.35, -0.95, -3.85):
        x = left_cut_x
        while x <= right_cut_x:
            m.circle(x, sy(y, y_scale), 0.08)
            x += 0.75

    # Cyan rectangular equipment rows seen in the Isaac top view.
    for y in (3.10, -2.95):
        m.dashed_blocks(-29.0, 52.5, sy(y, y_scale), block_w=6.0, block_h=0.42, gap=1.8)

    # Left-side nesting/utility blocks, drawn as repeated pieces rather than
    # one huge rectangle so the aisle around them remains open.
    for y in (-4.70, -3.95, -3.20, -2.45, -1.70, -0.95, -0.20, 0.55, 1.30, 2.05, 2.80, 3.55):
        ys = sy(y, y_scale)
        m.rect(-40.05, ys - 0.22, -38.65, ys + 0.22)

    # Short bottom service structures visible near the left start area.
    m.rect(-40.05, sy(-5.25, y_scale), -38.4, sy(-4.95, y_scale))

    # Right-end equipment box from the Isaac view, attached near the lower wall.
    m.rect(54.8, sy(-5.05, y_scale), 58.8, sy(-4.55, y_scale))


def write_outputs(m: Map, args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pgm_path = out_dir / f"{args.name}.pgm"
    yaml_path = out_dir / f"{args.name}.yaml"

    Image.fromarray(m.grid, mode="L").save(pgm_path)
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                "mode: trinary",
                f"resolution: {m.res}",
                f"origin: [{m.min_x}, {m.min_y}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {pgm_path}")
    print(f"Wrote {yaml_path}")
    print(f"size: {m.width} x {m.height}, origin: [{m.min_x}, {m.min_y}, 0.0]")


def main() -> None:
    args = parse_args()
    m = Map(args)
    draw_arena(m, args.y_scale)
    write_outputs(m, args)


if __name__ == "__main__":
    main()
