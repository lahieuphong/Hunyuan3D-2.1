# GPU preset catalog

`gpu_preset_catalog.json` is the single source of truth for the hardware
profiles shown by the WebUI. The file is committed with the source, so the
same list is available after cloning the repository on another machine.

## Using the catalog

1. Start the WebUI on the target machine.
2. Open **GPU Presets** in the top bar.
3. Confirm the automatically detected GPU and VRAM.
4. Select the matching GPU/VRAM profile when the automatic suggestion is not
   exact.
5. Apply either the `safe` or `quality` preset. The selected profile and tier
   are remembered for that browser and machine fingerprint.

Only entries backed by an end-to-end run may use
`"verification": "verified"`. Estimated or experimental entries must remain
clearly labelled as such. At present, the RTX 3090 24 GB profile is the only
verified profile in this workspace.

## Adding a profile

- Give the profile a stable, unique kebab-case `id`.
- Keep generic VRAM ranges non-overlapping. The upper boundary is exclusive.
- Use exact normalized GPU names in `aliases`; aliases never perform substring
  matching.
- Define exactly two presets, `safe` and `quality`, because those are the two
  actions currently exposed by the modal.
- Keep all four generation values inside the limits enforced by
  `webui/gpu_presets.py`.
- Add or update tests in `tests/test_gpu_presets.py`.

The loader validates the entire catalog during application startup and fails
early with a clear error when an entry is ambiguous or unsafe.
