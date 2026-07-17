#!/usr/bin/env python
"""Run Hunyuan3D Shape inference with an optional PEFT LoRA adapter."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHAPE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SHAPE_ROOT.parent

# Keep all downloaded weights inside the repository by default.  These values
# must be set before importing Transformers/Hugging Face-backed components.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HY3DGEN_MODELS", str(REPO_ROOT / ".cache" / "hy3dgen"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(SHAPE_ROOT))

import torch
from PIL import Image

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a GLB mesh from one RGBA image using Hunyuan3D Shape and LoRA."
    )
    parser.add_argument("--image", required=True, help="Input PNG path (RGBA is recommended).")
    parser.add_argument(
        "--adapter",
        default="",
        help="PEFT LoRA adapter directory. Omit it to use only the base model.",
    )
    parser.add_argument("--output", required=True, help="Destination .glb path.")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image was not found: {image_path}")
    if image_path.suffix.lower() != ".png":
        raise ValueError("The first Windows test expects a PNG input image.")

    with Image.open(image_path) as image:
        if image.mode != "RGBA":
            raise ValueError(
                f"Input image mode is {image.mode}, not RGBA. Remove the background and "
                "save it as a transparent RGBA PNG before inference."
            )

    if adapter_path is not None:
        config_path = adapter_path / "adapter_config.json"
        weights_path = adapter_path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "The adapter directory must contain adapter_config.json and "
                f"adapter_model.safetensors: {adapter_path}"
            )

    if output_path.suffix.lower() != ".glb":
        raise ValueError(f"Output must use the .glb extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path, adapter_path, output_path


def main() -> int:
    args = parse_args()
    image_path, adapter_path, output_path = resolve_inputs(args)

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.octree_resolution < 32:
        raise ValueError("--octree-resolution must be at least 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This inference launcher requires the RTX 3090.")

    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started_at = time.perf_counter()

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Input image: {image_path}", flush=True)
    print(f"LoRA adapter: {adapter_path if adapter_path else '(base model only)'}", flush=True)
    print(f"Output mesh: {output_path}", flush=True)
    print(
        f"Inference settings: steps={args.steps}, resolution={args.octree_resolution}, "
        f"guidance={args.guidance_scale}, seed={args.seed}",
        flush=True,
    )

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model,
        device="cuda",
        dtype=torch.float16,
    )
    if adapter_path is not None:
        pipeline.load_lora_adapter(str(adapter_path), merge=True)

    meshes = pipeline(
        image=str(image_path),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=generator,
    )
    if not meshes or meshes[0] is None:
        raise RuntimeError("Shape generation finished but no mesh could be extracted.")

    mesh = meshes[0]
    mesh.export(output_path)
    elapsed = time.perf_counter() - started_at
    print(
        f"SUCCESS: exported {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces "
        f"to {output_path} in {elapsed:.1f} seconds.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
