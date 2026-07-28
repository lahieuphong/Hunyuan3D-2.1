from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import trimesh

import gradio_app as app


def read_glb_json(path: str | Path) -> dict:
    payload = Path(path).read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)

    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise AssertionError("Export did not produce a valid GLB 2.0 container")
    if chunk_type != b"JSON":
        raise AssertionError("The first GLB chunk is not JSON")

    return json.loads(payload[20:20 + json_length].decode("utf-8"))


class MeshExportTests(unittest.TestCase):
    def test_geometry_only_glb_contains_vertex_normals(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = app.export_mesh(
                mesh,
                temp_dir,
                textured=False,
                type="glb",
            )
            document = read_glb_json(output_path)

        primitive = document["meshes"][0]["primitives"][0]
        self.assertIn("POSITION", primitive["attributes"])
        self.assertIn("NORMAL", primitive["attributes"])


if __name__ == "__main__":
    unittest.main()
