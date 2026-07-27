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
        self.assertTrue(ui_state[3]["value"].startswith("Áp dụng thử ·"))
        self.assertTrue(ui_state[4]["value"].startswith("Áp dụng thử ·"))
        self.assertIn("Đang dùng thử", ui_state[5])

        with runtime_match(app.HardwareMatch(BLACKWELL_ID, "nearest", True)):
            self.assertIsNone(app.get_available_hardware_profile())

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
                    _Request("http://127.0.0.1:8080/?tab=single-view"),
                )
        finally:
            if previous_turbo_mode is None:
                delattr(app, "TURBO_MODE")
            else:
                app.TURBO_MODE = previous_turbo_mode

        self.assertEqual(len(restored), 28)
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
                _Request("http://127.0.0.1:8080/?tab=single-view"),
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
                _Request("http://127.0.0.1:8080/?tab=single-view"),
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

        self.assertIn("isHardwareModalEnabled", custom_js)
        self.assertIn("if (!isHardwareModalEnabled() || !modal()) return;", custom_js)
        self.assertIn(
            "#rtx3090-modal:not(.hardware-presets-enabled).rtx-open",
            custom_css,
        )
        self.assertIn("GPU · Cấu hình đề xuất", custom_js)
        self.assertNotIn(
            "<span>RTX 3090 · Cấu hình đề xuất</span>", custom_js
        )

    def test_form_sync_keeps_legacy_history_label(self):
        _, _, status = app.get_hardware_form_state(
            None,
            30,
            5.0,
            384,
            8000,
            '<span data-history-review-active="true"></span>',
        )

        self.assertIn("Bản ghi cũ", status)
        self.assertIn("GPU chưa được lưu", status)

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
                    ),
                )
            finally:
                for name, value in original_values.items():
                    if value is None and hasattr(app, name):
                        delattr(app, name)
                    else:
                        setattr(app, name, value)

        self.assertEqual(len(restored), 28)
        self.assertEqual(restored[10]["value"], 30)
        self.assertEqual(restored[12]["value"], 384)
        self.assertFalse(restored[12]["interactive"])
        self.assertIn('data-profile="quality"', restored[17])
        self.assertIn("Đã lưu", restored[17])
        self.assertIn('data-history-review-active="true"', restored[22])
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
                    _Request(
                        f"http://127.0.0.1:8080/?generation={generation_uid}"
                    ),
                )
            finally:
                for name, value in original_values.items():
                    if value is None and hasattr(app, name):
                        delattr(app, name)
                    else:
                        setattr(app, name, value)

        self.assertIn(removed_hardware_label, restored[17])
        self.assertIn("Không còn trong catalog", restored[17])
        self.assertNotIn("GPU chưa được lưu", restored[17])
        self.assertEqual(restored[18]["value"], "Preset cũ · Không khả dụng")
        self.assertEqual(restored[19]["value"], "Preset cũ · Không khả dụng")
        self.assertIsNone(restored[24]["value"])
        self.assertIn(removed_hardware_label, restored[26])
        self.assertNotIn("RTX 3090", restored[26])

if __name__ == "__main__":
    unittest.main()
