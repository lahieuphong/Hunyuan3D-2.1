#!/usr/bin/env python3
"""Preflight checks for native-Windows Shape DiT LoRA training."""

import sys
import warnings


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError(
            f"Use Python 3.10 or 3.11, got {sys.version.split()[0]}"
        )

    import diffusers
    import omegaconf
    import peft
    import pytorch_lightning
    import torch
    import torch.nn.functional as functional
    import torch_cluster
    import transformers
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.__version__.startswith("2.5."):
        raise RuntimeError(
            f"The supplied torch-cluster wheel requires PyTorch 2.5.x, got {torch.__version__}"
        )
    if torch.version.cuda is None or not torch.version.cuda.startswith("12.4"):
        raise RuntimeError(
            f"Expected the PyTorch CUDA 12.4 build, got CUDA runtime {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU/PyTorch build does not support BF16")

    props = torch.cuda.get_device_properties(0)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(
        f"GPU: {props.name}; compute capability: {props.major}.{props.minor}; "
        f"VRAM: {props.total_memory / 1024**3:.1f} GiB"
    )
    print(f"PyTorch Lightning: {pytorch_lightning.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Diffusers: {diffusers.__version__}")
    print(f"OmegaConf: {omegaconf.__version__}")
    print(f"torch-cluster: {torch_cluster.__version__}")

    # Verify that the matching torch-cluster CUDA extension actually loads and
    # executes; importing the Python module alone does not catch all DLL issues.
    points = torch.rand((32, 3), device="cuda", dtype=torch.float32)
    batch = torch.zeros(32, device="cuda", dtype=torch.long)
    sampled = torch_cluster.fps(points, batch, ratio=0.25, random_start=False)
    if sampled.numel() != 8:
        raise RuntimeError(
            f"torch-cluster FPS returned {sampled.numel()} points instead of 8"
        )

    # The denoiser forces CUDA SDPA flash/memory-efficient kernels. Test a
    # representative BF16 attention call before downloading the base model.
    query = torch.rand((1, 1, 32, 64), device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel([
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
    ]):
        attention = functional.scaled_dot_product_attention(query, query, query)
    if not torch.isfinite(attention).all():
        raise RuntimeError("CUDA BF16 scaled-dot-product attention returned non-finite values")

    torch.cuda.synchronize()
    print("torch-cluster CUDA FPS: OK")
    print("CUDA BF16 attention: OK")
    print("Training environment preflight PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
