# GPU preset catalog

`gpu_preset_catalog.json` is the single source of truth for the hardware
profiles shown by the WebUI. The production catalog currently contains:

- `RTX 3090 · 24 GB`: presets verified by end-to-end 1-view and 4-view runs.
- `RTX PRO 6000 Blackwell · 96 GB`: exact runtime identity verified on the
  target workstation; its preset candidates reuse the RTX 3090 verified values
  and still require Blackwell end-to-end generation benchmarks.

## Using the catalog

1. Start the multiview WebUI on a cataloged machine.
2. Open the detected GPU name in the top bar.
3. Confirm the automatically detected GPU and VRAM.
4. Apply either the `safe` or `quality` preset. The selected profile and tier
   are remembered for that browser and machine fingerprint.

Preset actions require matching VRAM, backend, dtype, and compute capability.
Named profiles also require an exact normalized GPU alias; a generic verified
profile may match by its non-overlapping VRAM range within the same
backend/dtype/compute-capability group. A `runtime-verified`
profile always requires an exact alias. The selector stays locked to the active
runtime except while displaying the immutable profile saved in History.

## Verification states

- `verified`: both presets completed end-to-end generation and therefore have
  `verified: true`.
- `runtime-verified`: the GPU identity and a CUDA/PyTorch/dependency/WebUI
  preflight stack were observed on the target machine. Each launch still
  rechecks exact identity, VRAM, backend, dtype, and compute capability before
  exposing the trial actions. Presets remain `verified: false` and must not be
  described as benchmark results.
- `estimated` and `experimental`: catalog records that are not exposed as
  runtime preset actions.

The Blackwell runtime evidence currently covers PyTorch 2.7.1+cu128, CUDA 12.8,
compute capability 12.0, 95.59 GiB visible VRAM, BF16 matrix execution, clean
dependency checks, shape imports, and WebUI preflight. It does not yet cover a
completed mesh generation.

## Adding a profile

- Give the profile a stable, unique kebab-case `id`.
- Keep generic VRAM ranges non-overlapping within each backend, dtype, and
  compute-capability group. The upper boundary is exclusive.
- Use exact normalized GPU names in `aliases`; aliases never perform substring
  matching.
- Record `compute_capability` in `major.minor` form and require it to match the
  runtime exactly.
- A `runtime-verified` profile must have at least one exact alias.
- Define exactly two presets, `safe` and `quality`, because those are the two
  actions currently exposed by the modal.
- Keep all four generation values inside the limits enforced by
  `webui/gpu_presets.py`.
- Add or update tests in `tests/test_gpu_presets.py`.

Promote `runtime-verified` to `verified` only after both presets complete with
fixed inputs and seed in both 1-view and 4-view modes without OOM, and the
resulting GLB meshes pass basic vertex/face validity checks.

The loader validates the entire catalog during application startup and fails
early with a clear error when an entry is ambiguous or unsafe.
