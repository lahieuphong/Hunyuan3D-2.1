from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

import gradio_app as app


class _Request:
    def __init__(self, referer: str):
        self.headers = {"referer": referer}


class GradioHardwareIntegrationTests(unittest.TestCase):
    def test_browser_state_remembers_the_selected_quality_tier(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        state = app.hardware_browser_state(hardware_id, "quality")

        self.assertEqual(
            app.resolve_browser_hardware_selection(state),
            (hardware_id, "quality"),
        )

    def test_empty_restore_keeps_turbo_generation_default(self):
        hardware_id = "nvidia-rtx-3090-24gb"
        browser_state = app.hardware_browser_state(hardware_id, "quality")
        previous_turbo_mode = getattr(app, "TURBO_MODE", None)
        app.TURBO_MODE = True
        try:
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

    def test_generation_metadata_uses_actual_form_values(self):
        hardware_id = "nvidia-rtx-3090-24gb"
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
        self.assertEqual(hardware["selection_source"], "ui")
        self.assertEqual(preset["id"], "quality")
        self.assertEqual(preset["params_snapshot"]["octree_resolution"], 384)

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


if __name__ == "__main__":
    unittest.main()
