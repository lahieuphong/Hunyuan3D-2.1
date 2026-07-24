from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from webui.history import list_generation_history


class GenerationHistoryMetadataTests(unittest.TestCase):
    def _create_generation(self, root: Path, **manifest_updates):
        generation_uid = str(uuid.uuid4())
        folder = root / generation_uid
        folder.mkdir()
        (folder / "white_mesh.glb").write_bytes(b"glTF-history-test")
        manifest = {
            "schema_version": 1,
            "generation_uid": generation_uid,
            "status": "completed",
            "created_at": "2026-07-24T08:00:00+00:00",
            "completed_at": "2026-07-24T08:01:00+00:00",
            "events": [],
            "params": {
                "input_mode": "four",
                "steps": 30,
                "guidance_scale": 5.0,
                "seed": 1234,
                "octree_resolution": 384,
                "num_chunks": 8000,
                "views_used": ["front", "back", "left", "right"],
            },
            "inputs": {},
            "outputs": {"mesh": "white_mesh.glb"},
            "stats": {
                "params": {
                    "steps": 5,
                    "guidance_scale": 1.0,
                    "octree_resolution": 196,
                    "num_chunks": 1000,
                }
            },
        }
        manifest.update(manifest_updates)
        (folder / "generation.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return generation_uid

    def test_catalog_identity_is_sanitized_without_overriding_actual_params(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_generation(
                root,
                hardware={
                    "catalog_version": 1,
                    "id": "nvidia-rtx-3090-24gb",
                    "label": "NVIDIA RTX 3090 · 24 GB",
                    "selection_source": "ui",
                    "private_value": "must-not-leak",
                    "runtime": {
                        "name": "NVIDIA GeForce RTX 3090",
                        "backend": "cuda",
                        "total_vram_bytes": 25_769_803_776,
                        "capability": "8.6",
                        "dtype": "torch.float16",
                        "detected": True,
                        "error": "must-not-leak",
                    },
                },
                preset={
                    "catalog_version": 1,
                    "hardware_id": "nvidia-rtx-3090-24gb",
                    "id": "quality",
                    "source": "catalog",
                    "params_snapshot": {
                        "steps": 5,
                        "guidance_scale": 1.0,
                        "octree_resolution": 196,
                        "num_chunks": 1000,
                    },
                    "private_value": "must-not-leak",
                },
            )

            item = list_generation_history(root)["items"][0]

            self.assertEqual(
                item["parameters"],
                {
                    "steps": 30,
                    "guidance_scale": 5.0,
                    "seed": 1234,
                    "octree_resolution": 384,
                    "num_chunks": 8000,
                },
            )
            self.assertEqual(
                item["hardware"],
                {
                    "id": "nvidia-rtx-3090-24gb",
                    "label": "NVIDIA RTX 3090 · 24 GB",
                    "catalog_version": 1,
                    "selection_source": "ui",
                    "runtime": {
                        "name": "NVIDIA GeForce RTX 3090",
                        "backend": "cuda",
                        "total_vram_bytes": 25_769_803_776,
                        "capability": "8.6",
                        "dtype": "torch.float16",
                        "detected": True,
                    },
                },
            )
            self.assertEqual(
                item["preset"],
                {
                    "id": "quality",
                    "hardware_id": "nvidia-rtx-3090-24gb",
                    "catalog_version": 1,
                    "source": "catalog",
                },
            )
            self.assertNotIn("params_snapshot", item["preset"])

    def test_old_schema_v1_history_remains_tracked_and_has_no_metadata_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_generation(root)

            item = list_generation_history(root)["items"][0]

            self.assertFalse(item["legacy"])
            self.assertNotIn("hardware", item)
            self.assertNotIn("preset", item)

    def test_invalid_optional_metadata_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_generation(root, hardware="not-an-object", preset=["quality"])

            item = list_generation_history(root)["items"][0]

            self.assertFalse(item["legacy"])
            self.assertNotIn("hardware", item)
            self.assertNotIn("preset", item)


if __name__ == "__main__":
    unittest.main()
