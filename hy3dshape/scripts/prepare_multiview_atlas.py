"""Prepare the four canonical RGBA views as a calibrated texture atlas.

The output tiles use the same 15% border/recentering rule as Hunyuan3D-2mv.
This makes a simple orthographic projection line up with the generated mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    label,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy3dshape.preprocessors import ImageProcessorV2


VIEW_ORDER = ("front", "left", "back", "right")
TILE_POSITIONS = {
    "front": (0, 0),
    "left": (1, 0),
    "back": (0, 1),
    "right": (1, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for view in VIEW_ORDER:
        parser.add_argument(f"--{view}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_size = int(args.tile_size)
    processor = ImageProcessorV2(size=tile_size, border_ratio=0.15)
    atlas = Image.new("RGB", (tile_size * 2, tile_size * 2), "white")
    hair_atlas = Image.new("L", (tile_size * 2, tile_size * 2), 0)
    metadata: dict[str, object] = {
        "atlas_size": tile_size * 2,
        "tile_size": tile_size,
        "views": {},
    }

    for view in VIEW_ORDER:
        source = Image.open(getattr(args, view)).convert("RGBA")
        image, mask = processor.load_image(source, to_tensor=False)
        mask_2d = mask[..., 0] if mask.ndim == 3 else mask
        rows, columns = np.nonzero(mask_2d)
        if not len(rows):
            raise ValueError(f"{view} image has an empty alpha mask")
        background = mask_2d <= 127
        _, nearest = distance_transform_edt(background, return_indices=True)
        padded_image = image.copy()
        padded_image[background] = image[nearest[0][background], nearest[1][background]]
        image = padded_image
        # Preserve the supplied anime line work after projection and mipmapping.
        image = np.asarray(
            Image.fromarray(image).filter(
                ImageFilter.UnsharpMask(radius=1.4, percent=170, threshold=2)
            )
        )

        tile_x, tile_y = TILE_POSITIONS[view]
        head_limit = int(rows.min() + 0.20 * (rows.max() - rows.min()))
        rgb = image.astype(np.float32)
        luminance = rgb.mean(axis=2)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        # Include neutral-grey highlight streaks without leaking into warm skin.
        # The tight head crop keeps clothes out; largest-component selection
        # then discards isolated dark facial lines such as eyes and eyebrows.
        hair_like = (
            ((luminance < 115) | ((chroma < 20) & (luminance < 245)))
            & (mask_2d > 127)
        )
        hair_like[head_limit:, :] = False
        components, component_count = label(hair_like)
        if component_count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            hair = components == int(sizes.argmax())
            hair = binary_opening(hair, iterations=max(2, tile_size // 512))
            opened_components, opened_count = label(hair)
            if opened_count:
                opened_sizes = np.bincount(opened_components.ravel())
                opened_sizes[0] = 0
                hair = opened_components == int(opened_sizes.argmax())
            hair = binary_closing(hair, iterations=max(2, tile_size // 1024))
            hair = binary_fill_holes(hair)
            hair = binary_dilation(hair, iterations=max(4, tile_size // 256))
            hair_atlas.paste(
                Image.fromarray(np.uint8(hair) * 255, mode="L"),
                (tile_x * tile_size, tile_y * tile_size),
            )
        atlas.paste(Image.fromarray(image, mode="RGB"), (tile_x * tile_size, tile_y * tile_size))
        metadata["views"][view] = {
            "source": str(getattr(args, view).resolve()),
            "tile": [tile_x, tile_y],
            "bbox": [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()),
                int(rows.max()),
            ],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    hair_mask_path = args.output.with_name(f"{args.output.stem}_hair_mask.png")
    hair_atlas.save(hair_mask_path, format="PNG", optimize=True)
    metadata["hair_mask"] = str(hair_mask_path.resolve())
    atlas.save(args.output, format="PNG", optimize=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output} ({atlas.width}x{atlas.height})")
    print(f"Saved {hair_mask_path}")
    print(f"Saved {args.metadata}")


if __name__ == "__main__":
    main()
