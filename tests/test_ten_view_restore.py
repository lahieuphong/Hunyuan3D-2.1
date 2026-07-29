from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from PIL import Image

import gradio_app as app
from hy3dshape.ten_view import TEN_VIEW_KEYS
from webui.history import list_generation_history


class _Request:
    def __init__(self, referer: str):
        self.headers = {"referer": referer}


class TenViewRestoreTests(unittest.TestCase):
    def test_history_restores_all_ten_native_gradio_images(self):
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
            (folder / "white_mesh.glb").write_bytes(b"glTF-ten-view-test")
            inputs = {}
            for index, key in enumerate(TEN_VIEW_KEYS):
                filename = f"input_{key}.png"
                Image.new(
                    "RGBA",
                    (8, 8),
                    (index, index, index, 255),
                ).save(folder / filename)
                inputs[key] = filename

            (folder / "generation.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "generation_uid": generation_uid,
                        "status": "completed",
                        "events": [],
                        "input_mode": "ten",
                        "params": {
                            "input_mode": "ten",
                            "views_provided": list(TEN_VIEW_KEYS),
                            "views_used": list(TEN_VIEW_KEYS),
                            "steps": 10,
                            "guidance_scale": 5.0,
                            "seed": 1234,
                            "octree_resolution": 256,
                            "num_chunks": 8000,
                        },
                        "inputs": inputs,
                        "outputs": {"mesh": "white_mesh.glb"},
                        "stats": {},
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
                        "http://127.0.0.1:8080/"
                        f"?tab=ten-view&generation={generation_uid}"
                    ),
                )
                history_item = list_generation_history(root)["items"][0]
            finally:
                for name, value in original_values.items():
                    if value is None and hasattr(app, name):
                        delattr(app, name)
                    else:
                        setattr(app, name, value)

        self.assertEqual(len(restored), 38)
        self.assertEqual(restored[0], "ten")
        for legacy_image_update in restored[1:6]:
            self.assertIsNone(legacy_image_update["value"])
        self.assertEqual(restored[16]["value"], "Generate 3D · 10 Images")
        self.assertIn('data-input-mode="ten"', restored[22])
        self.assertEqual(
            [Path(update["value"]).name for update in restored[28:]],
            [f"input_{key}.png" for key in TEN_VIEW_KEYS],
        )
        self.assertTrue(all(update["interactive"] is False for update in restored[28:]))
        self.assertEqual(history_item["input_mode"], "ten")
        self.assertEqual(history_item["view_count"], 10)
        self.assertIn("input_front.png", history_item["assets"]["thumbnail_url"])


if __name__ == "__main__":
    unittest.main()
