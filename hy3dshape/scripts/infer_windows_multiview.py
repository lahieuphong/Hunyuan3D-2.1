#!/usr/bin/env python
"""Generate one GLB mesh from canonical front/left/back/right images."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHAPE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SHAPE_ROOT.parent

os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HY3DGEN_MODELS", str(REPO_ROOT / ".cache" / "hy3dgen"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(SHAPE_ROOT))

import torch
from PIL import Image

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


VIEW_NAMES = ("front", "left", "back", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one 3D mesh from four canonical object views."
    )
    for view_name in VIEW_NAMES:
        parser.add_argument(f"--{view_name}", required=True, help=f"RGBA PNG: {view_name} view")
    parser.add_argument("--output", required=True, help="Destination .glb path")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2mv")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, str], Path]:
    images: dict[str, str] = {}
    for view_name in VIEW_NAMES:
        image_path = Path(getattr(args, view_name)).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{view_name} image was not found: {image_path}")
        if image_path.suffix.lower() != ".png":
            raise ValueError(f"{view_name} image must be a PNG: {image_path}")
        with Image.open(image_path) as image:
            if image.mode != "RGBA":
                raise ValueError(
                    f"{view_name} image mode is {image.mode}, not RGBA. Remove its "
                    "background and save it as a transparent RGBA PNG."
                )
        images[view_name] = str(image_path)

    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"Output must use the .glb extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return images, output_path


def main() -> int:
    args = parse_args()
    images, output_path = validate_inputs(args)

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.octree_resolution < 32:
        raise ValueError("--octree-resolution must be at least 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Multi-view inference requires the RTX 3090.")

    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started_at = time.perf_counter()

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for view_name, image_path in images.items():
        print(f"{view_name.capitalize()} image: {image_path}", flush=True)
    print(f"Model: {args.model}/{args.subfolder}", flush=True)
    print(f"Output mesh: {output_path}", flush=True)
    print(
        f"Inference settings: steps={args.steps}, resolution={args.octree_resolution}, "
        f"guidance={args.guidance_scale}, seed={args.seed}",
        flush=True,
    )

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model,
        subfolder=args.subfolder,
        use_safetensors=True,
        variant="fp16",
        device="cuda",
        dtype=torch.float16,
    )
    meshes = pipeline(
        image=images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=generator,
        output_type="trimesh",
    )
    if not meshes or meshes[0] is None:
        raise RuntimeError("Multi-view generation finished but no mesh could be extracted.")

    mesh = meshes[0]
    mesh.export(str(output_path))
    elapsed = time.perf_counter() - started_at
    print(
        f"SUCCESS: exported {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces "
        f"to {output_path} in {elapsed:.1f} seconds.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
