from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gradio_app import mount_generation_static_files


class GenerationStaticMountTests(unittest.TestCase):
    def test_serves_generations_and_env_maps_but_not_private_siblings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            storage_root = Path(temporary_directory) / "webui"
            generation_root = storage_root / "generations"
            private_root = storage_root / "training"
            environment_maps = Path(temporary_directory) / "environment-maps"
            generation = generation_root / "generation-id"
            generation.mkdir(parents=True)
            private_root.mkdir(parents=True)
            environment_maps.mkdir(parents=True)
            (generation / "mesh.glb").write_bytes(b"generation")
            (private_root / "adapter.safetensors").write_bytes(b"private")
            (environment_maps / "gradient.jpg").write_bytes(b"environment")

            app = FastAPI()
            mount_generation_static_files(app, generation_root, environment_maps)
            client = TestClient(app)

            self.assertEqual(
                client.get("/static/generation-id/mesh.glb").content,
                b"generation",
            )
            self.assertEqual(
                client.get("/static/env_maps/gradient.jpg").content,
                b"environment",
            )
            self.assertEqual(
                client.get("/static/../training/adapter.safetensors").status_code,
                404,
            )
            self.assertEqual(
                client.get(
                    "/static/%2e%2e/training/adapter.safetensors"
                ).status_code,
                404,
            )


if __name__ == "__main__":
    unittest.main()
