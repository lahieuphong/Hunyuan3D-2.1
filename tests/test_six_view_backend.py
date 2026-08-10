from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from hy3dshape.six_view import (
    SIX_VIEW_ANGLES,
    SIX_VIEW_AUXILIARY_KEYS,
    SIX_VIEW_CONDITIONING_STRATEGY,
    SIX_VIEW_KEYS,
)
from hy3dshape.ten_view import CANONICAL_VIEW_KEYS, TEN_VIEW_KEYS
from hy3dshape.texture_bake.generation import (
    _view_frame,
    create_original_variant,
    glb_color_payload,
)
from webui.generation_inputs import (
    GenerationInputError,
    build_generation_input_bundle,
    normalize_input_mode,
    ordered_six_view_images,
)


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGBA", (48, 48), (*color, 255))


def _six_images(*, top: tuple[int, int, int] = (240, 20, 30)):
    colors = {
        "front": (110, 110, 110),
        "back": (120, 120, 120),
        "left": (130, 130, 130),
        "right": (140, 140, 140),
        "top": top,
        "bottom": (20, 220, 40),
    }
    return {name: _image(colors[name]) for name in SIX_VIEW_KEYS}


def _white_mesh(path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.export(path, file_type="glb", include_normals=True)


class SixViewContractTests(unittest.TestCase):
    def test_contract_has_four_native_and_two_polar_cameras(self):
        self.assertEqual(
            SIX_VIEW_KEYS,
            ("front", "back", "left", "right", "top", "bottom"),
        )
        self.assertEqual(SIX_VIEW_AUXILIARY_KEYS, ("top", "bottom"))
        self.assertEqual(tuple(SIX_VIEW_ANGLES), SIX_VIEW_KEYS)
        self.assertEqual(SIX_VIEW_ANGLES["top"], (0.0, 90.0))
        self.assertEqual(SIX_VIEW_ANGLES["bottom"], (0.0, -90.0))
        self.assertTrue(set(CANONICAL_VIEW_KEYS).isdisjoint(SIX_VIEW_AUXILIARY_KEYS))

    def test_mode_aliases_and_positional_mapping_are_stable(self):
        self.assertEqual(normalize_input_mode("six"), "six")
        self.assertEqual(normalize_input_mode("6-view"), "six")
        self.assertEqual(normalize_input_mode("six-view"), "six")
        images = list(_six_images().values())
        self.assertEqual(
            ordered_six_view_images(images),
            dict(zip(SIX_VIEW_KEYS, images, strict=True)),
        )
        with self.assertRaisesRegex(GenerationInputError, "expected 6 images"):
            ordered_six_view_images(images[:-1])

    def test_bundle_conditions_shape_with_only_native_cardinal_views(self):
        images = _six_images()
        bundle = build_generation_input_bundle(
            "six",
            None,
            {},
            None,
            images,
        )

        self.assertEqual(bundle.mode, "six")
        self.assertEqual(tuple(bundle.provided_images), SIX_VIEW_KEYS)
        self.assertEqual(tuple(bundle.conditioning_images), CANONICAL_VIEW_KEYS)
        self.assertIs(bundle.primary_image, images["front"])
        self.assertTrue(
            set(bundle.conditioning_images).isdisjoint(SIX_VIEW_AUXILIARY_KEYS)
        )
        self.assertEqual(bundle.metadata["conditioned_view_count"], 4)
        self.assertEqual(
            bundle.metadata["conditioning_strategy"],
            SIX_VIEW_CONDITIONING_STRATEGY,
        )
        self.assertEqual(bundle.metadata["views_provided"], list(SIX_VIEW_KEYS))
        self.assertEqual(
            bundle.metadata["shape_views_used"],
            list(CANONICAL_VIEW_KEYS),
        )
        self.assertEqual(
            bundle.metadata["texture_projection_views"],
            list(SIX_VIEW_KEYS),
        )
        self.assertFalse(bundle.metadata["experimental_conditioning"])

    def test_missing_and_unknown_six_view_inputs_are_rejected(self):
        missing_bottom = _six_images()
        missing_bottom.pop("bottom")
        with self.assertRaisesRegex(GenerationInputError, "Bottom"):
            build_generation_input_bundle(
                "six", None, {}, None, missing_bottom
            )

        unknown = _six_images()
        unknown["diagonal"] = _image((1, 2, 3))
        with self.assertRaisesRegex(GenerationInputError, "unsupported camera keys"):
            build_generation_input_bundle("six", None, {}, None, unknown)

    def test_existing_ten_view_positional_argument_is_unchanged(self):
        ten_images = {
            key: _image((index, index, index))
            for index, key in enumerate(TEN_VIEW_KEYS)
        }
        bundle = build_generation_input_bundle(
            "ten",
            None,
            {},
            ten_images,
        )
        self.assertEqual(bundle.mode, "ten")
        self.assertEqual(tuple(bundle.provided_images), TEN_VIEW_KEYS)


class SixViewProjectionTests(unittest.TestCase):
    def test_polar_frames_are_real_opposite_orthographic_cameras(self):
        top = _view_frame("top", *SIX_VIEW_ANGLES["top"])
        bottom = _view_frame("bottom", *SIX_VIEW_ANGLES["bottom"])

        np.testing.assert_allclose(top.to_camera, (0.0, 0.0, 1.0), atol=1e-10)
        np.testing.assert_allclose(bottom.to_camera, (0.0, 0.0, -1.0), atol=1e-10)
        np.testing.assert_allclose(top.right, (1.0, 0.0, 0.0), atol=1e-10)
        np.testing.assert_allclose(bottom.right, (1.0, 0.0, 0.0), atol=1e-10)
        self.assertAlmostEqual(float(np.dot(top.up, top.to_camera)), 0.0)
        self.assertAlmostEqual(float(np.dot(bottom.up, bottom.to_camera)), 0.0)

    def test_preferred_six_view_path_consumes_every_camera_and_publishes_glb(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            mesh_path = folder / "white_mesh.glb"
            _white_mesh(mesh_path)
            stages: list[tuple[str, str]] = []

            result = create_original_variant(
                mesh_path,
                _six_images(),
                folder,
                prefer_six_view=True,
                stage_callback=lambda stage, message: stages.append((stage, message)),
            )

            self.assertEqual(result.method, "six-view-orthographic-vertex-projection")
            self.assertEqual(result.views_used, SIX_VIEW_KEYS)
            self.assertEqual(
                result.source_strategy,
                "native-four-shape-six-view-color",
            )
            self.assertIsNone(result.fallback_reason)
            self.assertEqual(glb_color_payload(result.output_path), "vertex-color")
            self.assertTrue(result.output_path.is_file())
            self.assertTrue(any(stage == "baking_original" for stage, _ in stages))
            self.assertTrue(any("Top and Bottom" in message for _, message in stages))
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["views_used"], list(SIX_VIEW_KEYS))
            self.assertEqual(
                report["method"],
                "six-view-orthographic-vertex-projection",
            )
            self.assertEqual(
                report["source_strategy"],
                "native-four-shape-six-view-color",
            )
            self.assertIsNone(report["fallback_reason"])

    def test_changing_top_reference_changes_the_projected_mesh_colors(self):
        projected_colors = []
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            root = Path(temporary_directory)
            for index, top_color in enumerate(((250, 20, 20), (20, 40, 250))):
                folder = root / str(index)
                folder.mkdir()
                mesh_path = folder / "white_mesh.glb"
                _white_mesh(mesh_path)
                result = create_original_variant(
                    mesh_path,
                    _six_images(top=top_color),
                    folder,
                    prefer_six_view=True,
                )
                loaded = trimesh.load(result.output_path, force="mesh", process=False)
                projected_colors.append(
                    np.asarray(loaded.visual.vertex_colors, dtype=np.uint8)
                )

        self.assertEqual(projected_colors[0].shape, projected_colors[1].shape)
        self.assertFalse(np.array_equal(projected_colors[0], projected_colors[1]))


if __name__ == "__main__":
    unittest.main()
