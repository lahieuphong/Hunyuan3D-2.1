# GPU preset catalog

`gpu_preset_catalog.json` is the single source of truth for the hardware
profiles shown by the WebUI. The production catalog currently contains only
the verified `RTX 3090 · 24 GB` profile.

## Using the catalog

1. Start the WebUI on the verified RTX 3090 machine.
2. Open **RTX 3090** in the top bar.
3. Confirm the automatically detected GPU and VRAM.
4. Apply either the `safe` or `quality` preset. The selected profile and tier
   are remembered for that browser and machine fingerprint.

Only entries backed by an end-to-end run may use
`"verification": "verified"`. Unverified GPU/VRAM groups are intentionally
excluded until they have their own benchmark evidence.
Preset actions are exposed only when the runtime GPU name, VRAM, backend, and
dtype match a verified catalog profile.

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
