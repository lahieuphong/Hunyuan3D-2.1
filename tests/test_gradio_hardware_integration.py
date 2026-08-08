from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import gradio_app as app
from webui.gpu_presets import GpuPresetCatalog


BLACKWELL_ID = "nvidia-rtx-pro-6000-blackwell-workstation-96gb"


class _Request:
    def __init__(self, referer: str):
        self.headers = {"referer": referer}


@contextmanager
def runtime_match(match, *, multiview=True):
    previous_match = app.RUNTIME_HARDWARE_MATCH
    had_multiview = hasattr(app, "MV_MODE")
    previous_multiview = getattr(app, "MV_MODE", None)
    app.RUNTIME_HARDWARE_MATCH = match
    app.MV_MODE = multiview
    try:
        yield
    finally:
        app.RUNTIME_HARDWARE_MATCH = previous_match
        if had_multiview:
            app.MV_MODE = previous_multiview
        else:
            delattr(app, "MV_MODE")


class GradioHardwareIntegrationTests(unittest.TestCase):
    def test_browser_state_remembers_the_selected_quality_tier(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        state = app.hardware_browser_state(hardware_id, "quality")

        self.assertEqual(
            app.resolve_browser_hardware_selection(state),
            (hardware_id, "quality"),
        )

    def test_browser_state_remembers_blackwell_trial_tier(self):
        state = app.hardware_browser_state(BLACKWELL_ID, "quality")

        self.assertEqual(
            app.resolve_browser_hardware_selection(state),
            (BLACKWELL_ID, "quality"),
        )

    def test_forged_browser_state_is_clamped_to_runtime_profile(self):
        rtx_3090_id = "nvidia-rtx-3090-24gb"
        for runtime_id, forged_id in (
            (BLACKWELL_ID, rtx_3090_id),
            (rtx_3090_id, BLACKWELL_ID),
        ):
            with self.subTest(runtime_id=runtime_id, forged_id=forged_id):
                forged_state = app.hardware_browser_state(forged_id, "quality")
                self.assertEqual(
                    forged_state["runtime_fingerprint"],
                    app.RUNTIME_HARDWARE.fingerprint,
                )
                with runtime_match(app.HardwareMatch(runtime_id, "exact", True)):
                    restored = app.restore_generation_from_request(
                        forged_state,
                        _Request("http://127.0.0.1:8080/?tab=multi-view"), # type: ignore
                    )

                self.assertEqual(restored[23]["hardware_id"], runtime_id) # type: ignore
                self.assertEqual(restored[23]["preset_id"], "safe") # type: ignore
                self.assertEqual(restored[24]["value"], runtime_id)
                self.assertIn(f'data-hardware-id="{runtime_id}"', restored[25])
                self.assertIn(f'data-hardware-id="{forged_id}"', restored[25])
                self.assertIn('aria-current="true"', restored[25])
                self.assertIn('aria-disabled="true"', restored[25])

    def test_normal_ui_state_clamps_forged_profile_and_omits_duplicate_summary(self):
        rtx_3090_id = "nvidia-rtx-3090-24gb"
        with runtime_match(app.HardwareMatch(BLACKWELL_ID, "exact", True)):
            ui_state = app.get_hardware_ui_state(
                rtx_3090_id,
                30,
                5.0,
                384,
                8000,
            )

        profile_block, preset_cards = ui_state[:2]
        self.assertIn(f'data-hardware-id="{BLACKWELL_ID}"', profile_block)
        self.assertIn('aria-current="true"', profile_block)
        self.assertNotIn("hardware-profile-detail-heading", profile_block)
        self.assertNotIn("hardware-profile-summary", profile_block)
        self.assertEqual(preset_cards.count('role="button"'), 2)
        self.assertNotIn(f'data-hardware-id="{rtx_3090_id}"', preset_cards)
        self.assertTrue(ui_state[3]["interactive"])
        self.assertTrue(ui_state[4]["interactive"])
        self.assertTrue(ui_state[3]["value"].startswith("Try ·"))
        self.assertTrue(ui_state[4]["value"].startswith("Try ·"))

    def test_runtime_verified_blackwell_exposes_only_its_trial_presets(self):
        with runtime_match(app.HardwareMatch(BLACKWELL_ID, "exact", True)):
            profile = app.get_available_hardware_profile()
            applied = app.apply_hardware_preset(BLACKWELL_ID, "quality")
            hardware, preset = app.build_generation_hardware_metadata(
                BLACKWELL_ID,
                {
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "octree_resolution": 384,
                    "num_chunks": 8000,
                },
            )
            ui_state = app.get_hardware_ui_state(
                BLACKWELL_ID,
                30,
                5.0,
                384,
                8000,
            )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.verification, "runtime-verified")
        self.assertEqual(applied[:4], (30, 5.0, 384, 8000))
        self.assertEqual(applied[-1]["hardware_id"], BLACKWELL_ID)
        self.assertEqual(hardware["id"], BLACKWELL_ID)
        self.assertEqual(hardware["verification"], "runtime-verified")
        self.assertEqual(preset["id"], "quality")
        self.assertFalse(preset["verified"])
        self.assertTrue(ui_state[3]["value"].startswith("Try ·"))
        self.assertTrue(ui_state[4]["value"].startswith("Try ·"))
        self.assertIn("In use for testing", ui_state[5])
        self.assertEqual(ui_state[0].count('role="listitem"'), 2)
        self.assertIn(f'data-hardware-id="{BLACKWELL_ID}"', ui_state[0])
        self.assertIn('data-hardware-id="nvidia-rtx-3090-24gb"', ui_state[0])
        self.assertIn('data-runtime-current="true"', ui_state[0])
        self.assertIn('aria-disabled="true"', ui_state[0])

        with runtime_match(app.HardwareMatch(BLACKWELL_ID, "nearest", True)):
            self.assertIsNone(app.get_available_hardware_profile())

    def test_runtime_gate_rejects_other_catalog_gpu_in_both_directions(self):
        rtx_3090_id = "nvidia-rtx-3090-24gb"
        params = {
            "steps": 30,
            "guidance_scale": 5.0,
            "octree_resolution": 384,
            "num_chunks": 8000,
        }
        for runtime_id, blocked_id in (
            (BLACKWELL_ID, rtx_3090_id),
            (rtx_3090_id, BLACKWELL_ID),
        ):
            with self.subTest(runtime_id=runtime_id, blocked_id=blocked_id):
                with runtime_match(app.HardwareMatch(runtime_id, "exact", True)):
                    with self.assertRaises(app.gr.Error):
                        app.apply_hardware_preset(blocked_id, "quality")
                    with self.assertRaises(app.gr.Error):
                        app.select_hardware_profile(
                            blocked_id,
                            30,
                            5.0,
                            384,
                            8000,
                        )
                    hardware, preset = app.build_generation_hardware_metadata(
                        blocked_id,
                        params,
                    )
                    ui_state = app.get_hardware_ui_state(
                        runtime_id,
                        30,
                        5.0,
                        384,
                        8000,
                    )

                self.assertIsNone(hardware["id"])
                self.assertEqual(hardware["selection_source"], "api")
                self.assertIsNone(preset["hardware_id"])
                self.assertEqual(preset["id"], "custom")
                self.assertIn(f'data-hardware-id="{runtime_id}"', ui_state[0])
                self.assertIn(f'data-hardware-id="{blocked_id}"', ui_state[0])
                self.assertIn('aria-current="true"', ui_state[0])
                self.assertIn('aria-disabled="true"', ui_state[0])

    def test_saved_foreign_profile_stays_disabled_beside_current_runtime(self):
        rtx_3090_id = "nvidia-rtx-3090-24gb"
        with runtime_match(app.HardwareMatch(BLACKWELL_ID, "exact", True)):
            ui_state = app.get_hardware_ui_state(
                rtx_3090_id,
                30,
                5.0,
                384,
                8000,
                saved=True,
            )

        rendered = ui_state[0]
        self.assertIn("is-runtime-current", rendered)
        self.assertIn("is-history-saved", rendered)
        self.assertIn("Saved record · Read-only", rendered)
        self.assertIn('aria-disabled="true"', rendered)
        self.assertIn("Saved record profile details", rendered)
        self.assertIn("hardware-profile-summary", rendered)
        self.assertFalse(ui_state[3]["interactive"])
        self.assertFalse(ui_state[4]["interactive"])

    def test_verified_generic_profile_remains_available_for_vram_match(self):
        source = app.GPU_PRESET_CATALOG.get_hardware("nvidia-rtx-3090-24gb")
        self.assertIsNotNone(source)
        assert source is not None
        generic = replace(
            source,
            id="generic-verified-24gb",
            label="Generic verified 24 GB",
            aliases=(),
        )
        catalog = GpuPresetCatalog(
            schema_version=1,
            default_hardware_id=generic.id,
            hardware=(generic,),
        )
        previous_catalog = app.GPU_PRESET_CATALOG
        app.GPU_PRESET_CATALOG = catalog
        try:
            with runtime_match(app.HardwareMatch(generic.id, "vram", True)):
                self.assertEqual(app.get_available_hardware_profile(), generic)
        finally:
            app.GPU_PRESET_CATALOG = previous_catalog

    def test_empty_restore_keeps_turbo_generation_default(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        browser_state = app.hardware_browser_state(hardware_id, "quality")
        previous_turbo_mode = getattr(app, "TURBO_MODE", None)
        app.TURBO_MODE = True
        try:
            with runtime_match(app.HardwareMatch(hardware_id, "exact", True)):
                restored = app.restore_generation_from_request(
                    browser_state,
                    _Request("http://127.0.0.1:8080/?tab=single-view"), # type: ignore
                )
        finally:
            if previous_turbo_mode is None:
                delattr(app, "TURBO_MODE")
            else:
                app.TURBO_MODE = previous_turbo_mode

        self.assertEqual(len(restored), 39)
        self.assertEqual(restored[10]["value"], 5)
        self.assertEqual(restored[20]["value"], "Turbo")
        self.assertEqual(restored[12]["value"], 384)
        self.assertFalse(restored[24]["interactive"])

    def test_generation_metadata_uses_actual_form_values(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        with runtime_match(app.HardwareMatch(hardware_id, "exact", True)):
            hardware, preset = app.build_generation_hardware_metadata(
                hardware_id,
                {
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "octree_resolution": 384,
                    "num_chunks": 8000,
                },
            )

        self.assertEqual(hardware["id"], hardware_id)
        self.assertEqual(hardware["verification"], "verified")
        self.assertEqual(hardware["selection_source"], "ui")
        self.assertEqual(preset["id"], "quality")
        self.assertTrue(preset["verified"])
        self.assertEqual(preset["params_snapshot"]["octree_resolution"], 384)

    def test_incompatible_runtime_clears_profile_state_and_metadata(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        browser_state = app.hardware_browser_state(hardware_id, "quality")
        with runtime_match(app.HardwareMatch(None, "unavailable", False)):
            restored = app.restore_generation_from_request(
                browser_state,
                _Request("http://127.0.0.1:8080/?tab=single-view"), # type: ignore
            )
            hardware, preset = app.build_generation_hardware_metadata(
                hardware_id,
                {
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "octree_resolution": 384,
                    "num_chunks": 8000,
                },
            )

        self.assertIsNone(restored[23])
        self.assertIsNone(restored[24]["value"])
        self.assertFalse(restored[24]["interactive"])
        self.assertIsNone(hardware["id"])
        self.assertEqual(hardware["selection_source"], "api")
        self.assertIsNone(preset["hardware_id"])
        self.assertEqual(preset["id"], "custom")

    def test_single_view_mode_clears_profile_state_and_metadata(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        browser_state = app.hardware_browser_state(hardware_id, "quality")
        with runtime_match(
            app.HardwareMatch(hardware_id, "exact", True),
            multiview=False,
        ):
            restored = app.restore_generation_from_request(
                browser_state,
                _Request("http://127.0.0.1:8080/?tab=single-view"), # type: ignore
            )
            hardware, preset = app.build_generation_hardware_metadata(
                hardware_id,
                {
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "octree_resolution": 384,
                    "num_chunks": 8000,
                },
            )

        self.assertIsNone(restored[23])
        self.assertIsNone(restored[24]["value"])
        self.assertFalse(restored[24]["interactive"])
        self.assertIsNone(hardware["id"])
        self.assertEqual(hardware["selection_source"], "api")
        self.assertIsNone(preset["hardware_id"])
        self.assertEqual(preset["id"], "custom")

    def test_modal_bundle_requires_server_enabled_marker(self):
        custom_css, custom_js = app.load_ui_assets()

        self.assertIn(".hardware-catalog-card.is-disabled", custom_css)
        self.assertIn("cursor: not-allowed", custom_css)
        self.assertIn(
            ".hardware-catalog-card.is-disabled:hover .hardware-catalog-disabled-overlay",
            custom_css,
        )
        self.assertIn("ban:", custom_js)
        self.assertIn("@media (hover: none), (pointer: coarse)", custom_css)
        self.assertRegex(
            custom_css,
            r"(?s)#rtx3090-modal\s*>\s*\.rtx3090-modal-panel\s*>\s*"
            r"\.rtx3090-status-block\s*\{[^}]*display:\s*none\s*!important",
        )
        self.assertRegex(
            custom_css,
            r"(?s)#rtx3090-modal\s*>\s*\.rtx3090-modal-panel\s*>\s*"
            r"\.rtx-preset-actions\s*\{[^}]*display:\s*none\s*!important",
        )
        self.assertIn("Compact hardware modal", custom_css)
        self.assertNotIn("inset: auto 8px 8px", custom_css)

        self.assertIn("isHardwareModalEnabled", custom_js)
        self.assertIn("if (!isHardwareModalEnabled() || !modal()) return;", custom_js)
        self.assertIn(
            "#rtx3090-modal:not(.hardware-presets-enabled).rtx-open",
            custom_css,
        )
        self.assertIn("GPU · Recommended configuration", custom_js)
        self.assertNotIn("<span>RTX 3090 · Recommended configuration</span>", custom_js)

    def test_form_sync_keeps_legacy_history_label(self):
        _, _, status = app.get_hardware_form_state(
            None,
            30,
            5.0,
            384,
            8000,
            '<span data-history-review-active="true"></span>',
        )

        self.assertIn("Legacy record", status)
        self.assertIn("GPU was not saved", status)

    def test_history_restore_locks_profile_and_keeps_saved_quality_values(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        browser_state = app.hardware_browser_state(hardware_id, "quality")
        generation_uid = str(uuid.uuid4())
        original_values = {
            name: getattr(app, name, None)
            for name in (
                "SAVE_DIR",
                "MV_MODE",
                "HTML_HEIGHT",
                "HTML_OUTPUT_PLACEHOLDER",
            )
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / generation_uid
            folder.mkdir()
            (folder / "white_mesh.glb").write_bytes(b"glTF-test")
            (folder / "generation.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generation_uid": generation_uid,
                        "status": "completed",
                        "events": [],
                        "input_mode": "four",
                        "params": {
                            "input_mode": "four",
                            "steps": 30,
                            "guidance_scale": 5.0,
                            "seed": 1234,
                            "octree_resolution": 384,
                            "num_chunks": 8000,
                            "randomize_seed": True,
                        },
                        "inputs": {},
                        "outputs": {"mesh": "white_mesh.glb"},
                        "stats": {},
                        "hardware": {"id": hardware_id, "catalog_version": 1},
                        "preset": {
                            "id": "quality",
                            "hardware_id": hardware_id,
                            "catalog_version": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            app.SAVE_DIR = str(root)
            app.MV_MODE = True
            app.HTML_HEIGHT = 820
            app.HTML_OUTPUT_PLACEHOLDER = "placeholder"
            try:
                restored = app.restore_generation_from_request(
                    browser_state,
                    _Request(
                        f"http://127.0.0.1:8080/?tab=multi-view&generation={generation_uid}"
                    ), # type: ignore
                )
            finally:
                for name, value in original_values.items():
                    if value is None and hasattr(app, name):
                        delattr(app, name)
                    else:
                        setattr(app, name, value)

        self.assertEqual(len(restored), 39)
        self.assertIn(
            f'/generation-viewer/{generation_uid}?v=',
            restored[7],
        )
        self.assertEqual(restored[10]["value"], 30)
        self.assertEqual(restored[12]["value"], 384)
        self.assertFalse(restored[12]["interactive"])
        self.assertIn('data-profile="quality"', restored[17])
        self.assertIn("Saved", restored[17])
        self.assertIn('data-history-review-active="true"', restored[22])
        self.assertIn("Saved record · Read-only", restored[25])
        self.assertIn("This historical record is read-only", restored[25])
        self.assertIn("Saved record, read-only.", restored[25])
        self.assertNotIn("its presets are available", restored[25])
        self.assertEqual(restored[23], browser_state)
        self.assertEqual(restored[24]["value"], hardware_id)
        self.assertFalse(restored[24]["interactive"])
        self.assertIn("quality is-selected", restored[26])

    def test_removed_profile_history_keeps_saved_identity(self):
        generation_uid = str(uuid.uuid4())
        removed_hardware_id = "nvidia-16gb"
        removed_hardware_label = "NVIDIA 16 GB"
        original_values = {
            name: getattr(app, name, None)
            for name in (
                "SAVE_DIR",
                "MV_MODE",
                "HTML_HEIGHT",
                "HTML_OUTPUT_PLACEHOLDER",
            )
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / generation_uid
            folder.mkdir()
            (folder / "generation.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generation_uid": generation_uid,
                        "status": "completed",
                        "events": [],
                        "input_mode": "single",
                        "params": {
                            "input_mode": "single",
                            "steps": 30,
                            "guidance_scale": 5.0,
                            "seed": 1234,
                            "octree_resolution": 384,
                            "num_chunks": 4000,
                        },
                        "inputs": {},
                        "outputs": {},
                        "stats": {},
                        "hardware": {
                            "id": removed_hardware_id,
                            "label": removed_hardware_label,
                            "catalog_version": 1,
                        },
                        "preset": {
                            "id": "quality",
                            "hardware_id": removed_hardware_id,
                            "catalog_version": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            app.SAVE_DIR = str(root)
            app.MV_MODE = True
            app.HTML_HEIGHT = 820
            app.HTML_OUTPUT_PLACEHOLDER = "placeholder"
            try:
                restored = app.restore_generation_from_request(
                    None,
                    _Request(f"http://127.0.0.1:8080/?generation={generation_uid}"), # type: ignore
                )
            finally:
                for name, value in original_values.items():
                    if value is None and hasattr(app, name):
                        delattr(app, name)
                    else:
                        setattr(app, name, value)

        self.assertIn(removed_hardware_label, restored[17])
        self.assertIn("No longer in the catalog", restored[17])
        self.assertNotIn("GPU was not saved", restored[17])
        self.assertEqual(restored[18]["value"], "Legacy preset · Unavailable")
        self.assertEqual(restored[19]["value"], "Legacy preset · Unavailable")
        self.assertIsNone(restored[24]["value"])
        self.assertIn(removed_hardware_label, restored[26])
        self.assertIn(removed_hardware_label, restored[25])
        self.assertIn("not as the GPU from the saved record", restored[25])
        self.assertNotIn("The legacy record did not store a GPU", restored[25])
        self.assertNotIn("RTX 3090", restored[26])


if __name__ == "__main__":
    unittest.main()
