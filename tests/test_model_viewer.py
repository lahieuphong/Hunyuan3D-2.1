from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import trimesh

from webui.model_viewer import (
    export_wireframe_glb,
    is_wireframe_glb,
    normalize_viewer_locale,
    render_model_viewer_document,
    resolve_generation_assets,
    stored_generation_file,
    viewer_message,
)


VARIANT_ORDER = ("original", "white", "wireframe")


def read_glb_json(path: str | Path) -> dict:
    payload = Path(path).read_bytes()
    if len(payload) < 20:
        raise AssertionError("GLB payload is truncated")

    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise AssertionError("Export did not produce a valid GLB 2.0 container")
    if chunk_type != b"JSON":
        raise AssertionError("The first GLB chunk is not JSON")

    return json.loads(payload[20:20 + json_length].decode("utf-8"))


def export_triangle_glb(path: Path) -> Path:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh.export(path, include_normals=True)
    return path


class _ViewerMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons: list[dict[str, str | None]] = []
        self.viewer_config_parts: list[str] = []
        self._inside_viewer_config = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "button":
            self.buttons.append(attributes)
        if tag == "script" and attributes.get("id") == "viewer-config":
            self._inside_viewer_config = True

    def handle_endtag(self, tag):
        if tag == "script" and self._inside_viewer_config:
            self._inside_viewer_config = False

    def handle_data(self, data):
        if self._inside_viewer_config:
            self.viewer_config_parts.append(data)

    @property
    def viewer_config(self) -> dict:
        return json.loads("".join(self.viewer_config_parts))


def mode_buttons(parser: _ViewerMarkupParser) -> dict[str, dict[str, str | None]]:
    return {
        str(button["data-view-mode"]): button
        for button in parser.buttons
        if "data-view-mode" in button
    }


class StoredGenerationFileTests(unittest.TestCase):
    def test_accepts_only_direct_regular_glb_inside_generation_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / "generation"
            folder.mkdir()
            safe_mesh = export_triangle_glb(folder / "white_mesh.glb")

            nested = folder / "nested"
            nested.mkdir()
            export_triangle_glb(nested / "nested_mesh.glb")
            (folder / "notes.txt").write_text("not a GLB", encoding="utf-8")
            outside_mesh = export_triangle_glb(root / "outside.glb")

            resolved = stored_generation_file(
                folder,
                "white_mesh.glb",
                suffix=".glb",
            )
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved).resolve(), safe_mesh.resolve())

            rejected = (
                "../outside.glb",
                "nested/nested_mesh.glb",
                str(outside_mesh.resolve()),
                "notes.txt",
                None,
                42,
            )
            for filename in rejected:
                with self.subTest(filename=filename):
                    self.assertIsNone(
                        stored_generation_file(folder, filename, suffix=".glb")
                    )

    def test_rejects_symlink_even_when_it_points_to_a_valid_glb(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / "generation"
            folder.mkdir()
            outside_mesh = export_triangle_glb(root / "outside.glb")
            link = folder / "linked.glb"
            try:
                os.symlink(outside_mesh, link)
            except (NotImplementedError, OSError):
                # Creating symlinks may require Developer Mode or elevation on
                # Windows. The other path-containment cases remain portable.
                return

            self.assertIsNone(
                stored_generation_file(folder, link.name, suffix=".glb")
            )


class GenerationAssetResolverTests(unittest.TestCase):
    def test_resolves_all_three_modes_and_defaults_to_explicit_original(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            mesh = trimesh.creation.icosphere(subdivisions=1)
            export_triangle_glb(folder / "gohan_face_sharp_final.glb")
            export_triangle_glb(folder / "white_mesh.glb")
            export_wireframe_glb(mesh, folder / "wireframe_mesh.glb")
            manifest = {
                "outputs": {
                    "mesh": "white_mesh.glb",
                    "default_variant": "original",
                    "variants": {
                        "original": {
                            "file": "gohan_face_sharp_final.glb",
                            "render_mode": "embedded",
                        },
                        "white": {
                            "file": "white_mesh.glb",
                            "render_mode": "clay",
                        },
                        "wireframe": {
                            "file": "wireframe_mesh.glb",
                            "render_mode": "lines",
                        },
                    },
                }
            }

            assets = resolve_generation_assets(
                folder,
                manifest=manifest,
                ensure_wireframe=False,
            )

            self.assertIsNotNone(assets)
            assert assets is not None
            self.assertEqual(list(assets.variants), list(VARIANT_ORDER))
            self.assertEqual(assets.default_mode, "original")
            expected = {
                "original": (
                    "gohan_face_sharp_final.glb",
                    "embedded",
                ),
                "white": ("white_mesh.glb", "clay"),
                "wireframe": ("wireframe_mesh.glb", "lines"),
            }
            for mode, (filename, render_mode) in expected.items():
                with self.subTest(mode=mode):
                    variant = assets.variants[mode]
                    self.assertEqual(variant.mode, mode)
                    self.assertEqual(variant.filename, filename)
                    self.assertEqual(variant.render_mode, render_mode)
                    self.assertTrue(
                        os.path.samefile(variant.path, folder / filename)
                    )

    def test_white_only_legacy_generation_falls_back_to_white(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            export_triangle_glb(folder / "white_mesh.glb")

            assets = resolve_generation_assets(
                folder,
                manifest=None,
                ensure_wireframe=False,
            )

            self.assertIsNotNone(assets)
            assert assets is not None
            self.assertEqual(list(assets.variants), ["white"])
            self.assertEqual(assets.default_mode, "white")
            self.assertNotIn("original", assets.variants)
            self.assertNotIn("wireframe", assets.variants)

    def test_explicit_original_wins_over_legacy_textured_candidate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            export_triangle_glb(folder / "selected_original.glb")
            export_triangle_glb(folder / "textured_mesh.glb")
            export_triangle_glb(folder / "white_mesh.glb")
            manifest = {
                "outputs": {
                    "mesh": "white_mesh.glb",
                    "default_variant": "original",
                    "variants": {
                        "original": {
                            "file": "selected_original.glb",
                            "render_mode": "embedded",
                        },
                    },
                }
            }

            assets = resolve_generation_assets(
                folder,
                manifest=manifest,
                ensure_wireframe=False,
            )

            self.assertIsNotNone(assets)
            assert assets is not None
            self.assertEqual(
                assets.variants["original"].filename,
                "selected_original.glb",
            )
            self.assertEqual(assets.default_mode, "original")

    def test_invalid_explicit_assets_do_not_escape_or_become_modes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / "generation"
            folder.mkdir()
            export_triangle_glb(folder / "white_mesh.glb")
            export_triangle_glb(root / "outside.glb")
            nested = folder / "nested"
            nested.mkdir()
            export_triangle_glb(nested / "nested.glb")
            (folder / "not_mesh.txt").write_text("invalid", encoding="utf-8")

            for invalid_file in (
                "../outside.glb",
                "nested/nested.glb",
                str((root / "outside.glb").resolve()),
                "not_mesh.txt",
            ):
                with self.subTest(invalid_file=invalid_file):
                    assets = resolve_generation_assets(
                        folder,
                        manifest={
                            "outputs": {
                                "default_variant": "original",
                                "variants": {
                                    "original": {
                                        "file": invalid_file,
                                        "render_mode": "embedded",
                                    },
                                    "white": {
                                        "file": "white_mesh.glb",
                                        "render_mode": "clay",
                                    },
                                },
                            }
                        },
                        ensure_wireframe=False,
                    )
                    self.assertIsNotNone(assets)
                    assert assets is not None
                    self.assertEqual(list(assets.variants), ["white"])
                    self.assertEqual(assets.default_mode, "white")


class WireframeExportTests(unittest.TestCase):
    def test_icosphere_export_contains_only_gltf_line_primitives(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "wireframe_mesh.glb"

            exported = export_wireframe_glb(mesh, target)
            document = read_glb_json(exported)

            self.assertTrue(os.path.samefile(exported, target))
            self.assertTrue(target.is_file())
            self.assertTrue(is_wireframe_glb(target))
            primitives = [
                primitive
                for gltf_mesh in document.get("meshes", ())
                for primitive in gltf_mesh.get("primitives", ())
            ]
            self.assertTrue(primitives)
            self.assertTrue(
                all(primitive.get("mode", 4) == 1 for primitive in primitives)
            )
            self.assertTrue(
                all(
                    "POSITION" in primitive.get("attributes", {})
                    for primitive in primitives
                )
            )
            scene = document["scenes"][document.get("scene", 0)]
            self.assertEqual(
                scene["extras"]["wireframe_generator_version"],
                2,
            )

            triangle_path = export_triangle_glb(
                Path(temporary_directory) / "triangles.glb"
            )
            self.assertFalse(is_wireframe_glb(triangle_path))

    def test_dense_mesh_is_clustered_for_a_readable_wireframe_preview(self):
        mesh = trimesh.creation.icosphere(subdivisions=5)
        full_resolution_line_vertices = len(mesh.edges_unique) * 2
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "wireframe_mesh.glb"

            export_wireframe_glb(mesh, target)
            document = read_glb_json(target)

        primitive = document["meshes"][0]["primitives"][0]
        position_accessor = document["accessors"][
            primitive["attributes"]["POSITION"]
        ]
        self.assertGreater(position_accessor["count"], 0)
        self.assertLess(
            position_accessor["count"],
            full_resolution_line_vertices,
        )

    def test_asset_resolution_rebuilds_a_stale_canonical_wireframe(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            white_mesh = export_triangle_glb(folder / "white_mesh.glb")
            assets = resolve_generation_assets(folder, ensure_wireframe=True)

            self.assertIsNotNone(assets)
            assert assets is not None
            wireframe = assets.variants["wireframe"].path
            stale_time = max(0, white_mesh.stat().st_mtime_ns - 1_000_000_000)
            os.utime(wireframe, ns=(stale_time, stale_time))
            self.assertLess(
                wireframe.stat().st_mtime_ns,
                white_mesh.stat().st_mtime_ns,
            )

            refreshed_assets = resolve_generation_assets(
                folder,
                ensure_wireframe=True,
            )

            self.assertIsNotNone(refreshed_assets)
            assert refreshed_assets is not None
            self.assertGreaterEqual(
                refreshed_assets.variants["wireframe"].path.stat().st_mtime_ns,
                white_mesh.stat().st_mtime_ns,
            )


class ModelViewerDocumentTests(unittest.TestCase):
    def test_normalizes_supported_viewer_locales_and_defaults_to_english(self):
        self.assertEqual(normalize_viewer_locale("en-US"), "en")
        self.assertEqual(normalize_viewer_locale("zh_Hans"), "zh-CN")
        self.assertEqual(normalize_viewer_locale("zh-CN"), "zh-CN")
        self.assertEqual(normalize_viewer_locale("vi"), "en")
        self.assertEqual(normalize_viewer_locale(None), "en")
        self.assertEqual(
            viewer_message("generationNotFound", "en"),
            "Generation not found",
        )
        self.assertEqual(
            viewer_message("generationNotFound", "zh-CN"),
            "未找到生成记录",
        )

    def test_renders_simplified_chinese_static_viewer_copy(self):
        document = render_model_viewer_document(
            {
                "original": "/static/generation/original.glb",
                "white": "/static/generation/white_mesh.glb",
                "wireframe": "/static/generation/wireframe_mesh.glb",
            },
            "original",
            650,
            500,
            locale="zh-CN",
        )
        parser = _ViewerMarkupParser()
        parser.feed(document)

        self.assertIn('<html lang="zh-CN">', document)
        self.assertIn("Hunyuan3D-2mv · 3D 模型查看器", document)
        self.assertIn('aria-label="生成的 3D 模型预览"', document)
        self.assertIn(">点击并拖动以旋转<", document)
        self.assertIn(">滚动以缩放<", document)
        self.assertIn(">原始模型<", document)
        self.assertIn(">白模<", document)
        self.assertIn(">线框<", document)
        self.assertEqual(parser.viewer_config["locale"], "zh-CN")
        self.assertEqual(
            parser.viewer_config["messages"]["zh-CN"]["fullscreen"],
            "全屏",
        )

    def test_viewer_can_follow_parent_and_persisted_locale_at_runtime(self):
        document = render_model_viewer_document(
            {"white": "/static/generation/white_mesh.glb"},
            "white",
            650,
            500,
        )

        self.assertIn("hunyuan3d.ui-locale.v1", document)
        self.assertIn("window.parent.currentUiLocale", document)
        self.assertIn('window.parent.addEventListener("ui-language-change"', document)
        self.assertIn('window.addEventListener("storage"', document)
        self.assertIn("applyLocale(currentLocale)", document)

    def test_renders_three_modes_without_camera_presets_and_keeps_toolbar(self):
        variant_sources = {
            "original": "/static/generation/original.glb",
            "white": "/static/generation/white_mesh.glb",
            "wireframe": "/static/generation/wireframe_mesh.glb",
        }

        document = render_model_viewer_document(
            variant_sources,
            "original",
            810,
            500,
        )
        parser = _ViewerMarkupParser()
        parser.feed(document)
        buttons = mode_buttons(parser)

        self.assertEqual(list(buttons), list(VARIANT_ORDER))
        self.assertEqual(
            [
                mode
                for mode, button in buttons.items()
                if button.get("aria-pressed") == "true"
            ],
            ["original"],
        )
        self.assertNotIn("camera-preset", document)
        toolbar_actions = {
            button["data-action"]
            for button in parser.buttons
            if "data-action" in button
        }
        self.assertEqual(
            toolbar_actions,
            {"fullscreen", "reset", "rotate", "grid"},
        )

        config = parser.viewer_config
        self.assertEqual(config["defaultMode"], "original")
        self.assertEqual(
            config["variants"],
            {
                mode: {"src": source}
                for mode, source in variant_sources.items()
            },
        )

    def test_white_only_document_disables_original_and_wireframe(self):
        document = render_model_viewer_document(
            {
                "original": None,
                "white": "/static/generation/white_mesh.glb",
                "wireframe": None,
            },
            "original",
            650,
            500,
        )
        parser = _ViewerMarkupParser()
        parser.feed(document)
        buttons = mode_buttons(parser)

        self.assertEqual(list(buttons), list(VARIANT_ORDER))
        self.assertEqual(buttons["white"].get("aria-pressed"), "true")
        self.assertEqual(buttons["original"].get("aria-pressed"), "false")
        self.assertEqual(buttons["wireframe"].get("aria-pressed"), "false")
        for unavailable in ("original", "wireframe"):
            with self.subTest(mode=unavailable):
                self.assertIn("disabled", buttons[unavailable])
                self.assertEqual(
                    buttons[unavailable].get("aria-disabled"),
                    "true",
                )
        self.assertNotIn("disabled", buttons["white"])
        self.assertEqual(parser.viewer_config["defaultMode"], "white")

    def test_viewer_config_is_json_safe_for_inline_script_context(self):
        hostile_url = (
            "/static/generation/original.glb"
            "?label=</script><script>alert(1)</script>&separator=\u2028"
        )
        document = render_model_viewer_document(
            {
                "original": hostile_url,
                "white": "/static/generation/white_mesh.glb",
                "wireframe": None,
            },
            "original",
            650,
            500,
        )
        parser = _ViewerMarkupParser()
        parser.feed(document)

        self.assertNotIn("</script><script>alert(1)</script>", document)
        self.assertIn("\\u003c", document)
        self.assertEqual(
            parser.viewer_config["variants"]["original"]["src"],
            hostile_url,
        )


if __name__ == "__main__":
    unittest.main()
