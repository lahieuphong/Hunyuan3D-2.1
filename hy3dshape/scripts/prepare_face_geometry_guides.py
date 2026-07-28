"""Remove 2D facial ink from shape-conditioning images.

Anime eyes, eyebrows, and mouth lines should be represented by albedo, not by
deep 3D grooves. These guides retain the silhouette, hair, ears, and clothing
while inpainting only the small interior facial-feature regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REGIONS = {
    "front": [
        ((340, 322), (56, 45)),
        ((455, 320), (54, 45)),
        ((398, 374), (34, 39)),
    ],
    "left": [
        ((286, 300), (43, 48)),
        ((274, 354), (31, 30)),
    ],
    "right": [
        ((505, 306), (46, 48)),
        ((521, 357), (31, 29)),
    ],
    "back": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def largest_dark_component(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = np.uint8((gray < 70) & (alpha > 127))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if count <= 1:
        return np.zeros_like(dark)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    protected = np.uint8(labels == component) * 255
    return cv2.dilate(protected, np.ones((5, 5), np.uint8), iterations=1)


def prepare_view(source: Path, destination: Path, view: str) -> None:
    rgba = np.asarray(Image.open(source).convert("RGBA"))
    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3]
    inpaint_mask = np.zeros(alpha.shape, dtype=np.uint8)
    for center, axes in REGIONS[view]:
        cv2.ellipse(inpaint_mask, center, axes, 0, 0, 360, 255, thickness=-1)

    if np.any(inpaint_mask):
        protected_hair = largest_dark_component(rgb, alpha)
        inpaint_mask[protected_hair > 0] = 0
        inpaint_mask[alpha < 128] = 0
        red, green, blue = (rgb[..., channel] for channel in range(3))
        skin = (
            (alpha > 127)
            & (red > 185)
            & (green > 145)
            & (blue > 135)
            & ((red.astype(np.int16) - green) < 70)
            & ((green.astype(np.int16) - blue) < 45)
        )
        skin_color = np.median(rgb[skin], axis=0)
        feather = cv2.GaussianBlur(inpaint_mask, (0, 0), sigmaX=9).astype(np.float32) / 255.0
        feather[protected_hair > 0] = 0.0
        feather[alpha < 128] = 0.0
        rgb = (
            rgb.astype(np.float32) * (1.0 - feather[..., None])
            + skin_color[None, None, :] * feather[..., None]
        ).clip(0, 255).astype(np.uint8)

    output = np.dstack((rgb, alpha))
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGBA").save(destination)


def main() -> None:
    args = parse_args()
    for view in ("front", "back", "left", "right"):
        filename = f"Gohan_{view.capitalize()}_transparent.png"
        prepare_view(args.input_dir / filename, args.output_dir / filename, view)
        print(f"Prepared {view}: {args.output_dir / filename}")


if __name__ == "__main__":
    main()
