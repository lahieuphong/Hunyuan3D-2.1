from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.color import ColorVisuals

import gradio_app as app
import hy3dshape.texture_bake.generation as original_generation
from hy3dshape.texture_bake.face_hair_rc import (
    FaceHairRCResult,
    FaceHairRCView,
)
from hy3dshape.texture_bake.generation import (
    OriginalVariantError,
    OriginalVariantResult,
    create_original_variant,
    glb_color_payload,
    resolve_blender_executable,
)
from hy3dshape.texture_bake.rc_evaluator import (
    FaceHairRCCandidateEvaluator,
)
from hy3dshape.ten_view import TEN_VIEW_KEYS
from webui.model_viewer import resolve_generation_assets


def _rgba(color: tuple[int, int, int], size: tuple[int, int] = (48, 72)) -> Image.Image:
    image = Image.new("RGBA", size, (*color, 0))
    left = size[0] // 4
    image.paste(
        (*color, 255),
        (left, 4, size[0] - left, size[1] - 4),
    )
    return image


def _white_mesh(path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh.export(path, file_type="glb", include_normals=True)


def _passing_rc_result() -> FaceHairRCResult:
    views = tuple(
        FaceHairRCView(
            name=name,
            score=85.0,
            semantic_score=85.0,
            color_score=85.0,
            detail_score=85.0,
            foreground_iou=0.9,
            face_leakage=0.05,
            hair_leakage=0.05,
            face_pixels=100,
            hair_pixels=100,
        )
        for name in TEN_VIEW_KEYS
    )
    return FaceHairRCResult(
        score=85.0,
        raw_score=85.0,
        passed_hard_gates=True,
        hard_gate_failures=(),
        mean_view_score=85.0,
        worst_quartile_score=85.0,
        semantic_score=85.0,
        color_score=85.0,
        detail_score=85.0,
        foreground_iou=0.9,
        worst_quartile_face_leakage=0.05,
        worst_quartile_hair_leakage=0.05,
        views=views,
    )


class OriginalVariantGenerationTests(unittest.TestCase):
    def test_single_view_fallback_exports_real_vertex_colors(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)

            result = create_original_variant(
                white_path,
                {"front": _rgba((238, 72, 24))},
                folder,
            )

            self.assertEqual(result.method, "multi-view-vertex-color-fallback")
            self.assertEqual(result.views_used, ("front",))
            self.assertEqual(glb_color_payload(result.output_path), "vertex-color")
            self.assertTrue((folder / "vertex_color_report.json").is_file())
            document = json.loads(
                (folder / "vertex_color_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["views_used"], ["front"])
            gate = json.loads(
                (folder / "original_quality_gate.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(gate["rc"]["status"], "not_evaluated")
            self.assertFalse(gate["rc"]["required"])
            self.assertEqual(gate["promotion"]["mode"], "structural-only")
            self.assertEqual(
                result.to_metadata()["quality_gate_report"],
                "original_quality_gate.json",
            )

    def test_quality_evaluator_runs_before_candidate_promotion(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            calls = []

            def evaluate(candidate_path, view_names, images):
                self.assertFalse((folder / "textured_mesh.glb").exists())
                calls.append((candidate_path.name, view_names, tuple(images)))
                return {
                    "score": 91.25,
                    "passed_hard_gates": True,
                    "hard_gate_failures": [],
                }

            result = create_original_variant(
                white_path,
                {"front": _rgba((238, 72, 24))},
                folder,
                quality_evaluator=evaluate,
            )

            self.assertEqual(
                calls,
                [
                    (
                        "textured_mesh.pending.glb",
                        ("front",),
                        ("front",),
                    )
                ],
            )
            self.assertTrue(result.output_path.is_file())
            assert result.quality_gate is not None
            self.assertEqual(
                result.quality_gate["promotion"]["mode"],
                "rc-hard-gate",
            )
            self.assertEqual(
                result.quality_gate["rc"]["result"]["score"],
                91.25,
            )

    def test_failed_quality_evaluator_cannot_publish_candidate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)

            with self.assertRaisesRegex(
                OriginalVariantError,
                "rejected by the face/hair RC hard gates",
            ):
                create_original_variant(
                    white_path,
                    {"front": _rgba((238, 72, 24))},
                    folder,
                    quality_evaluator=lambda *_: {
                        "score": 42.0,
                        "passed_hard_gates": False,
                        "hard_gate_failures": ["face_leakage_above_limit"],
                    },
                )

            self.assertFalse((folder / "textured_mesh.glb").exists())
            self.assertTrue(
                (folder / "original_quality_gate_rejected.json").is_file()
            )

    def test_incomplete_ten_view_rc_schema_cannot_publish_candidate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            pending = folder / "textured_mesh.invalid.pending.glb"
            output = folder / "textured_mesh.glb"
            mesh = trimesh.creation.icosphere(subdivisions=1)
            mesh.visual = ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(
                    np.array((20, 140, 235, 255), dtype=np.uint8),
                    (len(mesh.vertices), 1),
                ),
            )
            mesh.export(pending, file_type="glb", include_normals=True)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }

            with self.assertRaisesRegex(
                OriginalVariantError,
                "incomplete or non-finite metrics",
            ):
                original_generation._promote_original_candidate(
                    pending,
                    output,
                    color_payload="vertex-color",
                    method="test-incomplete-rc",
                    views_used=TEN_VIEW_KEYS,
                    source_strategy="native-ten-view",
                    images=images,
                    quality_evaluator=lambda *_: {
                        "passed_hard_gates": True,
                        "score": 99.0,
                    },
                    stage_callback=None,
                    candidate_identifier="invalid",
                    preserve_candidate=True,
                )

            self.assertFalse(output.exists())


    def test_shape_uses_four_views_but_returns_all_ten_processed_references(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            source_images = {
                key: Image.new(
                    "RGBA",
                    (32, 48),
                    (index + 1, 40, 80, 255),
                )
                for index, key in enumerate(TEN_VIEW_KEYS)
            }
            remover_calls = []
            captured_conditioning = {}

            def remove_background(rgb_image):
                remover_calls.append(rgb_image.getpixel((0, 0))[0])
                result = rgb_image.convert("RGBA")
                alpha = Image.new("L", result.size, 255)
                alpha.putpixel((0, 0), 0)
                result.putalpha(alpha)
                return result

            def generate_shape(**kwargs):
                captured_conditioning.update(kwargs["image"])
                return object()

            generated_mesh = trimesh.creation.icosphere(subdivisions=1)
            with (
                patch.object(app, "MV_MODE", True, create=True),
                patch.object(
                    app,
                    "args",
                    SimpleNamespace(
                        model_path="test/model",
                        subfolder="shape",
                        texgen_model_path="disabled",
                    ),
                    create=True,
                ),
                patch.object(
                    app,
                    "gen_save_folder",
                    return_value=str(folder),
                ),
                patch.object(
                    app,
                    "get_background_remover",
                    return_value=remove_background,
                ),
                patch.object(
                    app,
                    "i23d_worker",
                    side_effect=generate_shape,
                    create=True,
                ),
                patch.object(
                    app,
                    "export_to_trimesh",
                    return_value=[generated_mesh],
                    create=True,
                ),
            ):
                (
                    _mesh,
                    _main_image,
                    _save_folder,
                    stats,
                    _seed,
                    texture_images,
                ) = app._gen_shape(
                    input_mode="ten",
                    ten_view_images=source_images,
                    steps=1,
                )

            cardinal_views = ("front", "left", "back", "right")
            self.assertEqual(tuple(captured_conditioning), cardinal_views)
            self.assertEqual(tuple(texture_images), TEN_VIEW_KEYS)
            self.assertEqual(
                remover_calls,
                list(range(1, len(TEN_VIEW_KEYS) + 1)),
            )
            self.assertEqual(
                stats["params"]["automatic_background_removal"],
                list(TEN_VIEW_KEYS),
            )
            for key in TEN_VIEW_KEYS:
                self.assertEqual(
                    texture_images[key].getchannel("A").getextrema(),
                    (0, 255),
                )
                if key in cardinal_views:
                    self.assertIs(
                        captured_conditioning[key],
                        texture_images[key],
                    )
                else:
                    self.assertNotIn(key, captured_conditioning)
    def test_vertex_fallback_uses_all_ten_supplied_views(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }

            with (
                patch.object(
                    original_generation,
                    "_default_ten_view_quality_evaluator",
                    return_value=lambda *_: _passing_rc_result(),
                ),
                patch(
                    "hy3dshape.texture_bake.generation._bake_ten_view_with_blender",
                    side_effect=OriginalVariantError("forced ten-view failure"),
                ),
                patch(
                    "hy3dshape.texture_bake.generation._bake_with_blender",
                    side_effect=OriginalVariantError("forced bake failure"),
                ),
            ):
                result = create_original_variant(white_path, images, folder)

            self.assertEqual(result.method, "multi-view-vertex-color-fallback")
            self.assertEqual(result.views_used, TEN_VIEW_KEYS)
            self.assertEqual(result.source_strategy, "all-supplied-views")
            self.assertIn("forced bake failure", result.fallback_reason or "")
            self.assertEqual(glb_color_payload(result.output_path), "vertex-color")
            assert result.quality_gate is not None
            self.assertTrue(result.quality_gate["rc"]["required"])

    def test_ten_view_api_autowires_default_rc_evaluator(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }
            evaluator = object()
            expected = OriginalVariantResult(
                output_path=folder / "textured_mesh.glb",
                color_payload="texture",
                method="visibility-aware-cardinal-bake",
                views_used=("front", "left", "back", "right"),
                source_strategy="cardinal-four",
                seconds=1.0,
            )

            with (
                patch.object(
                    original_generation,
                    "_default_ten_view_quality_evaluator",
                    return_value=evaluator,
                ) as factory,
                patch.object(
                    original_generation,
                    "_bake_with_blender",
                    return_value=expected,
                ) as cardinal,
            ):
                result = create_original_variant(white_path, images, folder)

            self.assertIs(result, expected)
            factory.assert_called_once_with(blender_path=None, timeout=600)
            self.assertIs(
                cardinal.call_args.kwargs["quality_evaluator"], evaluator
            )
            self.assertEqual(
                cardinal.call_args.kwargs["candidate_identifier"], "cardinal"
            )
            self.assertTrue(cardinal.call_args.kwargs["preserve_candidate"])

    def test_all_ten_views_choose_ten_view_blender_when_preferred(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }
            expected = OriginalVariantResult(
                output_path=folder / "textured_mesh.glb",
                color_payload="texture",
                method="strict-visibility-ten-view-consensus",
                views_used=TEN_VIEW_KEYS,
                source_strategy="native-ten-view",
                seconds=1.0,
            )

            with (
                patch(
                    "hy3dshape.texture_bake.generation._bake_ten_view_with_blender",
                    return_value=expected,
                ) as ten_view,
                patch(
                    "hy3dshape.texture_bake.generation._bake_with_blender",
                ) as cardinal,
            ):
                result = create_original_variant(
                    white_path,
                    images,
                    folder,
                    prefer_ten_view=True,
                )

            self.assertIs(result, expected)
            ten_view.assert_called_once()
            cardinal.assert_not_called()

    def test_failed_ten_view_bake_falls_back_to_cardinal_bake(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }
            expected = OriginalVariantResult(
                output_path=folder / "textured_mesh.glb",
                color_payload="texture",
                method="visibility-aware-cardinal-bake",
                views_used=("front", "left", "back", "right"),
                source_strategy="cardinal-four",
                seconds=1.0,
            )

            with (
                patch(
                    "hy3dshape.texture_bake.generation._bake_ten_view_with_blender",
                    side_effect=OriginalVariantError("ten-view failed"),
                ) as ten_view,
                patch(
                    "hy3dshape.texture_bake.generation._bake_with_blender",
                    return_value=expected,
                ) as cardinal,
            ):
                result = create_original_variant(
                    white_path,
                    images,
                    folder,
                    prefer_ten_view=True,
                )

            self.assertIs(result, expected)
            ten_view.assert_called_once()
            cardinal.assert_called_once()

    def test_all_ten_views_use_clean_cardinal_bake_by_default(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }
            expected = OriginalVariantResult(
                output_path=folder / "textured_mesh.glb",
                color_payload="texture",
                method="visibility-aware-cardinal-bake",
                views_used=("front", "left", "back", "right"),
                source_strategy="cardinal-four",
                seconds=1.0,
            )

            with (
                patch(
                    "hy3dshape.texture_bake.generation._bake_ten_view_with_blender",
                ) as ten_view,
                patch(
                    "hy3dshape.texture_bake.generation._bake_with_blender",
                    return_value=expected,
                ) as cardinal,
            ):
                result = create_original_variant(white_path, images, folder)

            self.assertIs(result, expected)
            cardinal.assert_called_once()
            ten_view.assert_not_called()

    def test_failed_cardinal_bake_uses_ten_view_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }
            expected = OriginalVariantResult(
                output_path=folder / "textured_mesh.glb",
                color_payload="texture",
                method="strict-visibility-ten-view-consensus",
                views_used=TEN_VIEW_KEYS,
                source_strategy="native-ten-view",
                seconds=1.0,
            )

            with (
                patch(
                    "hy3dshape.texture_bake.generation._bake_ten_view_with_blender",
                    return_value=expected,
                ) as ten_view,
                patch(
                    "hy3dshape.texture_bake.generation._bake_with_blender",
                    side_effect=OriginalVariantError("cardinal failed"),
                ) as cardinal,
            ):
                result = create_original_variant(white_path, images, folder)

            self.assertIs(result, expected)
            cardinal.assert_called_once()
            ten_view.assert_called_once()

    def test_ten_view_rc_publishes_ranked_degraded_candidate_without_overwrite(
        self,
    ):
        with tempfile.TemporaryDirectory(
            dir=Path.cwd()
        ) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }

            def rejected_candidate(
                identifier: str,
                *,
                leakage: float,
                score: float,
                color: tuple[int, int, int],
            ):
                candidate_path = (
                    folder
                    / f"textured_mesh.{identifier}.pending.glb"
                )
                mesh = trimesh.creation.icosphere(subdivisions=1)
                mesh.visual = ColorVisuals(
                    mesh=mesh,
                    vertex_colors=np.tile(
                        np.array((*color, 255), dtype=np.uint8),
                        (len(mesh.vertices), 1),
                    ),
                )
                mesh.export(
                    candidate_path,
                    file_type="glb",
                    include_normals=True,
                )
                view = FaceHairRCView(
                    name="front",
                    score=score,
                    semantic_score=score,
                    color_score=70.0,
                    detail_score=70.0,
                    foreground_iou=0.9,
                    face_leakage=leakage,
                    hair_leakage=leakage,
                    face_pixels=100,
                    hair_pixels=100,
                )
                rc_result = FaceHairRCResult(
                    score=49.0,
                    raw_score=score,
                    passed_hard_gates=False,
                    hard_gate_failures=(
                        "face_leakage_above_limit",
                    ),
                    mean_view_score=score,
                    worst_quartile_score=score,
                    semantic_score=score,
                    color_score=70.0,
                    detail_score=70.0,
                    foreground_iou=0.9,
                    worst_quartile_face_leakage=leakage,
                    worst_quartile_hair_leakage=leakage,
                    views=(view,),
                )
                rejection_path = (
                    folder
                    / (
                        "original_quality_gate_rejected."
                        f"{identifier}.json"
                    )
                )
                rejection_path.write_text(
                    json.dumps(
                        {
                            "candidate": candidate_path.name,
                            "evaluation": rc_result.to_dict(),
                        }
                    ),
                    encoding="utf-8",
                )
                report_path = folder / f"report.{identifier}.json"
                report_path.write_text("{}", encoding="utf-8")
                candidate = (
                    original_generation._RejectedOriginalCandidate(
                        identifier=identifier,
                        candidate_path=candidate_path,
                        color_payload="vertex-color",
                        method=f"{identifier}-method",
                        views_used=TEN_VIEW_KEYS,
                        source_strategy=f"{identifier}-source",
                        seconds=1.0,
                        evaluation=rc_result.to_dict(),
                        rc_result=rc_result,
                        rejection_path=rejection_path,
                        report_path=report_path,
                    )
                )
                return original_generation._OriginalCandidateRejected(
                    candidate
                )

            ten_rejection = rejected_candidate(
                "ten-view",
                leakage=0.18,
                score=55.0,
                color=(20, 90, 220),
            )
            cardinal_rejection = rejected_candidate(
                "cardinal",
                leakage=0.08,
                score=55.0,
                color=(240, 110, 20),
            )
            vertex_rejection = rejected_candidate(
                "vertex",
                leakage=0.28,
                score=55.0,
                color=(80, 180, 80),
            )
            with (
                patch(
                    "hy3dshape.texture_bake.generation."
                    "_bake_ten_view_with_blender",
                    side_effect=ten_rejection,
                ) as ten_view,
                patch(
                    "hy3dshape.texture_bake.generation."
                    "_bake_with_blender",
                    side_effect=cardinal_rejection,
                ) as cardinal,
                patch(
                    "hy3dshape.texture_bake.generation."
                    "_vertex_color_fallback",
                    side_effect=vertex_rejection,
                ) as vertex,
            ):
                result = create_original_variant(
                    white_path,
                    images,
                    folder,
                    prefer_ten_view=True,
                    quality_evaluator=lambda *_: {},
                )

            self.assertEqual(result.method, "cardinal-method")
            self.assertEqual(
                result.quality_gate["promotion"]["mode"],
                "rc-best-available-degraded",
            )
            self.assertFalse(
                result.quality_gate["promotion"]["passed_hard_gates"]
            )
            self.assertEqual(
                [
                    item["identifier"]
                    for item in result.quality_gate["candidate_ranking"]
                ],
                ["cardinal", "ten-view", "vertex"],
            )
            for identifier in ("ten-view", "cardinal", "vertex"):
                self.assertTrue(
                    (
                        folder
                        / f"textured_mesh.{identifier}.pending.glb"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        folder
                        / (
                            "original_quality_gate_rejected."
                            f"{identifier}.json"
                        )
                    ).is_file()
                )
            self.assertEqual(
                glb_color_payload(folder / "textured_mesh.glb"),
                "vertex-color",
            )
            self.assertEqual(
                ten_view.call_args.kwargs["candidate_identifier"],
                "ten-view",
            )
            self.assertEqual(
                cardinal.call_args.kwargs["candidate_identifier"],
                "cardinal",
            )
            self.assertEqual(
                vertex.call_args.kwargs["candidate_identifier"],
                "vertex",
            )


    def test_ten_view_rc_scores_every_candidate_before_publishing_best_pass(
        self,
    ):
        with tempfile.TemporaryDirectory(
            dir=Path.cwd()
        ) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / 'white_mesh.glb'
            _white_mesh(white_path)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }

            def evaluated_candidate(
                identifier: str,
                *,
                passed: bool,
                worst_quartile: float,
                detail: float,
                color: tuple[int, int, int],
            ):
                candidate_path = (
                    folder
                    / f'textured_mesh.{identifier}.pending.glb'
                )
                mesh = trimesh.creation.icosphere(subdivisions=1)
                mesh.visual = ColorVisuals(
                    mesh=mesh,
                    vertex_colors=np.tile(
                        np.array((*color, 255), dtype=np.uint8),
                        (len(mesh.vertices), 1),
                    ),
                )
                mesh.export(
                    candidate_path,
                    file_type='glb',
                    include_normals=True,
                )
                view = FaceHairRCView(
                    name='front',
                    score=worst_quartile,
                    semantic_score=worst_quartile,
                    color_score=82.0,
                    detail_score=detail,
                    foreground_iou=0.92,
                    face_leakage=0.04,
                    hair_leakage=0.03,
                    face_pixels=100,
                    hair_pixels=100,
                )
                failures = () if passed else ('face_leakage_above_limit',)
                rc_result = FaceHairRCResult(
                    score=worst_quartile,
                    raw_score=worst_quartile,
                    passed_hard_gates=passed,
                    hard_gate_failures=failures,
                    mean_view_score=worst_quartile,
                    worst_quartile_score=worst_quartile,
                    semantic_score=worst_quartile,
                    color_score=82.0,
                    detail_score=detail,
                    foreground_iou=0.92,
                    worst_quartile_face_leakage=0.04,
                    worst_quartile_hair_leakage=0.03,
                    views=(view,),
                )
                decision_path = (
                    folder / f'original_quality_gate.{identifier}.json'
                )
                decision_path.write_text('{}', encoding='utf-8')
                candidate = original_generation._EvaluatedOriginalCandidate(
                    identifier=identifier,
                    candidate_path=candidate_path,
                    color_payload='vertex-color',
                    method=f'{identifier}-method',
                    views_used=TEN_VIEW_KEYS,
                    source_strategy=f'{identifier}-source',
                    seconds=1.0,
                    evaluation=rc_result.to_dict(),
                    rc_result=rc_result,
                    rejection_path=decision_path,
                )
                return (
                    original_generation._OriginalCandidateEvaluated(
                        candidate
                    ),
                    candidate_path.read_bytes(),
                )

            ten_signal, _ = evaluated_candidate(
                'ten-view',
                passed=True,
                worst_quartile=74.0,
                detail=62.0,
                color=(20, 90, 220),
            )
            cardinal_signal, cardinal_bytes = evaluated_candidate(
                'cardinal',
                passed=True,
                worst_quartile=84.0,
                detail=81.0,
                color=(240, 110, 20),
            )
            vertex_signal, _ = evaluated_candidate(
                'vertex',
                passed=False,
                worst_quartile=52.0,
                detail=35.0,
                color=(80, 180, 80),
            )
            with (
                patch(
                    'hy3dshape.texture_bake.generation.'
                    '_bake_ten_view_with_blender',
                    side_effect=ten_signal,
                ) as ten_view,
                patch(
                    'hy3dshape.texture_bake.generation._bake_with_blender',
                    side_effect=cardinal_signal,
                ) as cardinal,
                patch(
                    'hy3dshape.texture_bake.generation.'
                    '_vertex_color_fallback',
                    side_effect=vertex_signal,
                ) as vertex,
            ):
                result = create_original_variant(
                    white_path,
                    images,
                    folder,
                    prefer_ten_view=True,
                    quality_evaluator=lambda *_: {},
                )

            ten_view.assert_called_once()
            cardinal.assert_called_once()
            vertex.assert_called_once()
            self.assertEqual(result.method, 'cardinal-method')
            self.assertEqual(
                result.quality_gate['promotion']['mode'],
                'rc-best-scored',
            )
            self.assertTrue(
                result.quality_gate['promotion']['passed_hard_gates']
            )
            self.assertEqual(
                (folder / 'textured_mesh.glb').read_bytes(),
                cardinal_bytes,
            )

    def test_canonical_colored_and_white_files_resolve_to_three_modes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            white_path = folder / "white_mesh.glb"
            _white_mesh(white_path)
            colored = trimesh.creation.icosphere(subdivisions=1)
            color = np.tile(
                np.array([20, 140, 235, 255], dtype=np.uint8),
                (len(colored.vertices), 1),
            )
            colored.visual = ColorVisuals(
                mesh=colored,
                vertex_colors=color,
            )
            colored.export(
                folder / "textured_mesh.glb",
                file_type="glb",
                include_normals=True,
            )

            assets = resolve_generation_assets(folder, ensure_wireframe=True)

            self.assertIsNotNone(assets)
            assert assets is not None
            self.assertEqual(
                tuple(assets.variants),
                ("original", "white", "wireframe"),
            )
            self.assertEqual(assets.default_mode, "original")
            self.assertEqual(assets.primary.filename, "textured_mesh.glb")

    def test_explicit_blender_path_is_resolved(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            executable = Path(temporary_directory) / "blender.exe"
            executable.write_bytes(b"test")

            self.assertEqual(
                resolve_blender_executable(executable),
                executable.resolve(),
            )

    def test_shape_generation_completes_with_all_three_variants(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            save_root = Path(temporary_directory)
            generation_uid = str(uuid.uuid4())
            folder = save_root / generation_uid
            folder.mkdir()
            mesh = trimesh.creation.icosphere(subdivisions=1)
            image = _rgba((230, 86, 28))
            stats = {
                "generation": {
                    "uid": generation_uid,
                    "status": "processing",
                    "inputs": {"front": "input_front.png"},
                },
                "model": {"shapegen": "test/shape", "texgen": "disabled"},
                "params": {"input_mode": "single"},
                "time": {"shape generation": 0.1},
                "number_of_faces": len(mesh.faces),
                "number_of_vertices": len(mesh.vertices),
            }
            shape_result = (
                mesh,
                image,
                str(folder),
                stats,
                1234,
                {"front": image},
            )

            with (
                patch.object(app, "_gen_shape", return_value=shape_result),
                patch.object(
                    app,
                    "generation_uid_from_request",
                    return_value=generation_uid,
                ),
                patch.object(
                    app,
                    "build_generation_hardware_metadata",
                    return_value=({}, {}),
                ),
                patch.object(
                    app,
                    "args",
                    SimpleNamespace(low_vram_mode=False),
                    create=True,
                ),
                patch.object(app, "SAVE_DIR", str(save_root), create=True),
                patch.object(app, "HTML_HEIGHT", 650, create=True),
                patch.object(app, "HTML_WIDTH", 790, create=True),
                patch.object(
                    app,
                    "HTML_OUTPUT_PLACEHOLDER",
                    "",
                    create=True,
                ),
            ):
                result = app.shape_generation()

            manifest = json.loads(
                (folder / "generation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                manifest["outputs"]["mesh"],
                "textured_mesh.glb",
            )
            self.assertEqual(
                manifest["outputs"]["default_variant"],
                "original",
            )
            self.assertEqual(
                tuple(manifest["outputs"]["variants"]),
                ("original", "white", "wireframe"),
            )
            self.assertEqual(
                glb_color_payload(folder / "textured_mesh.glb"),
                "vertex-color",
            )
            viewer_document = (folder / "white_mesh.html").read_text(encoding="utf-8")
            self.assertIn('"defaultMode":"original"', viewer_document)
            self.assertNotIn(
                'data-view-mode="original" data-icon="palette" '
                'aria-pressed="false" disabled',
                viewer_document,
            )
            self.assertEqual(
                Path(result[0]["value"]),
                folder / "textured_mesh.glb",
            )


    def test_ten_view_ui_prefers_native_ten_view_original(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            save_root = Path(temporary_directory)
            generation_uid = str(uuid.uuid4())
            folder = save_root / generation_uid
            folder.mkdir()
            mesh = trimesh.creation.icosphere(subdivisions=1)
            images = {
                key: _rgba(((index * 23) % 255, 90, 210))
                for index, key in enumerate(TEN_VIEW_KEYS, start=1)
            }
            stats = {
                "generation": {
                    "uid": generation_uid,
                    "status": "processing",
                    "inputs": {
                        key: f"input_{key}.png"
                        for key in TEN_VIEW_KEYS
                    },
                },
                "model": {"shapegen": "test/shape", "texgen": "disabled"},
                "params": {"input_mode": "ten"},
                "time": {"shape generation": 0.1},
                "number_of_faces": len(mesh.faces),
                "number_of_vertices": len(mesh.vertices),
            }
            shape_result = (
                mesh,
                images["front"],
                str(folder),
                stats,
                1234,
                images,
            )

            def create_colored_original(
                _mesh_path,
                _images,
                output_folder,
                **_kwargs,
            ):
                colored = trimesh.creation.icosphere(subdivisions=1)
                color = np.tile(
                    np.array([20, 140, 235, 255], dtype=np.uint8),
                    (len(colored.vertices), 1),
                )
                colored.visual = ColorVisuals(
                    mesh=colored,
                    vertex_colors=color,
                )
                output_path = Path(output_folder) / "textured_mesh.glb"
                colored.export(
                    output_path,
                    file_type="glb",
                    include_normals=True,
                )
                return OriginalVariantResult(
                    output_path=output_path,
                    color_payload="vertex-color",
                    method="strict-visibility-ten-view-consensus",
                    views_used=TEN_VIEW_KEYS,
                    source_strategy="native-ten-view",
                    seconds=1.0,
                    quality_gate={
                        "rc": {
                            "required": True,
                            "status": "not_evaluated",
                        }
                    },
                )

            with (
                patch.object(app, "_gen_shape", return_value=shape_result),
                patch.object(
                    app,
                    "create_original_variant",
                    side_effect=create_colored_original,
                ) as create_original,
                patch.object(
                    app,
                    "generation_uid_from_request",
                    return_value=generation_uid,
                ),
                patch.object(
                    app,
                    "build_generation_hardware_metadata",
                    return_value=({}, {}),
                ),
                patch.object(
                    app,
                    "args",
                    SimpleNamespace(low_vram_mode=False),
                    create=True,
                ),
                patch.object(app, "SAVE_DIR", str(save_root), create=True),
                patch.object(app, "HTML_HEIGHT", 650, create=True),
                patch.object(app, "HTML_WIDTH", 790, create=True),
                patch.object(app, "HTML_OUTPUT_PLACEHOLDER", "", create=True),
            ):
                app.shape_generation(input_mode="ten")

            self.assertTrue(
                create_original.call_args.kwargs["prefer_ten_view"]
            )
            self.assertIsInstance(
                create_original.call_args.kwargs["quality_evaluator"],
                FaceHairRCCandidateEvaluator,
            )

if __name__ == "__main__":
    unittest.main()
