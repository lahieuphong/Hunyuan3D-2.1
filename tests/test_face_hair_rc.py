"""Tests for deterministic face/hair reprojection consistency scoring."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from hy3dshape.texture_bake.face_hair_rc import (
    rank_face_hair_candidates,
    score_face_hair_rc,
    select_best_face_hair_candidate,
)
from hy3dshape.texture_bake.semantics import SemanticClass, classify_anime_colors


SKIN = (235, 166, 122, 255)
HAIR = (13, 191, 209, 255)
INK = (18, 25, 34, 255)


def _head_view(*, mirror: bool = False) -> np.ndarray:
    image = np.zeros((128, 128, 4), dtype=np.uint8)

    # Cyan silhouette with several spikes.
    cv2.ellipse(image, (64, 52), (37, 43), 0, 0, 360, HAIR, -1)
    spikes = (
        ((34, 36), (40, 5), (51, 33)),
        ((46, 25), (57, 1), (62, 31)),
        ((59, 22), (73, 0), (74, 33)),
        ((75, 25), (93, 6), (86, 41)),
        ((87, 36), (112, 24), (92, 54)),
    )
    for spike in spikes:
        cv2.fillConvexPoly(image, np.asarray(spike, dtype=np.int32), HAIR)

    # Skin is drawn on top of the hair silhouette, like bangs around a face.
    cv2.ellipse(image, (64, 71), (23, 30), 0, 0, 360, SKIN, -1)
    cv2.ellipse(image, (40, 70), (7, 12), 0, 0, 360, SKIN, -1)
    cv2.ellipse(image, (88, 70), (7, 12), 0, 0, 360, SKIN, -1)

    # Bangs and line art make detail/boundary loss measurable.
    cv2.fillConvexPoly(
        image,
        np.asarray(((43, 45), (52, 68), (61, 43)), dtype=np.int32),
        HAIR,
    )
    cv2.fillConvexPoly(
        image,
        np.asarray(((58, 42), (66, 66), (73, 42)), dtype=np.int32),
        HAIR,
    )
    cv2.line(image, (48, 69), (59, 67), INK, 2)
    cv2.line(image, (69, 67), (80, 69), INK, 2)
    cv2.line(image, (64, 70), (61, 82), INK, 1)
    cv2.line(image, (57, 89), (70, 89), INK, 2)
    cv2.line(image, (39, 42), (54, 22), INK, 2)
    cv2.line(image, (74, 25), (87, 43), INK, 2)
    if mirror:
        image = np.ascontiguousarray(image[:, ::-1])
    return image


def _semantic_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = classify_anime_colors(image[..., :3])
    foreground = image[..., 3] > 0
    face = foreground & (labels == int(SemanticClass.SKIN_LIKE))
    hair = foreground & (labels == int(SemanticClass.COOL_SATURATED))
    return face, hair


class FaceHairRCTests(unittest.TestCase):
    def test_identical_aligned_views_receive_full_passing_score(self):
        references = {
            "front": _head_view(),
            "left": _head_view(mirror=True),
        }

        result = score_face_hair_rc(references, references)

        self.assertTrue(result.passed_hard_gates)
        self.assertEqual(result.hard_gate_failures, ())
        self.assertGreater(result.score, 99.9)
        self.assertGreater(result.worst_quartile_score, 99.9)
        self.assertEqual(tuple(view.name for view in result.views), ("front", "left"))
        self.assertEqual(result.to_dict()["passed_hard_gates"], True)

    def test_face_hair_color_swap_is_leakage_and_fails_hard_gates(self):
        reference = _head_view()
        rendered = reference.copy()
        face, hair = _semantic_masks(reference)
        rendered[face] = HAIR
        rendered[hair] = SKIN

        result = score_face_hair_rc(rendered, reference)

        self.assertFalse(result.passed_hard_gates)
        self.assertLessEqual(result.score, 49.0)
        self.assertIn("face_leakage_above_limit", result.hard_gate_failures)
        self.assertIn("hair_leakage_above_limit", result.hard_gate_failures)
        self.assertGreater(result.views[0].face_leakage or 0.0, 0.95)
        self.assertGreater(result.views[0].hair_leakage or 0.0, 0.95)

    def test_worst_quartile_exposes_one_bad_side(self):
        reference = _head_view()
        bad = reference.copy()
        face, hair = _semantic_masks(reference)
        bad[face] = HAIR
        bad[hair] = SKIN
        rendered = [reference.copy(), reference.copy(), reference.copy(), bad]
        references = [reference] * 4

        result = score_face_hair_rc(
            rendered,
            references,
            view_names=("front", "right", "back", "left"),
        )

        self.assertLess(result.worst_quartile_score, result.mean_view_score)
        self.assertFalse(result.passed_hard_gates)
        self.assertGreater(
            result.worst_quartile_face_leakage or 0.0,
            0.95,
        )

    def test_blurring_line_art_lowers_boundary_detail(self):
        reference = _head_view()
        rendered = cv2.GaussianBlur(reference, (15, 15), 4.0)
        rendered[..., 3] = reference[..., 3]

        exact = score_face_hair_rc(reference, reference)
        blurred = score_face_hair_rc(rendered, reference)

        self.assertLess(blurred.detail_score, exact.detail_score - 20.0)
        self.assertLess(blurred.score, exact.score)

    def test_file_paths_are_loaded_as_rgba_without_channel_swap(self):
        reference = _head_view()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "head.png"
            written = cv2.imwrite(
                str(path),
                cv2.cvtColor(reference, cv2.COLOR_RGBA2BGRA),
            )
            self.assertTrue(written)

            result = score_face_hair_rc(path, path)

        self.assertTrue(result.passed_hard_gates)
        self.assertGreater(result.score, 99.9)

    def test_ranking_is_gate_first_and_ties_break_by_candidate_id(self):
        reference = _head_view()
        good = score_face_hair_rc(reference, reference)
        bad_render = reference.copy()
        face, hair = _semantic_masks(reference)
        bad_render[face] = HAIR
        bad_render[hair] = SKIN
        bad = score_face_hair_rc(bad_render, reference)

        ranking = rank_face_hair_candidates(
            {
                "zeta_equal": good,
                "failed": bad,
                "alpha_equal": good,
            }
        )
        winner_id, winner = select_best_face_hair_candidate(
            {
                "zeta_equal": good,
                "failed": bad,
                "alpha_equal": good,
            }
        )

        self.assertEqual(ranking, ("alpha_equal", "zeta_equal", "failed"))
        self.assertEqual(winner_id, "alpha_equal")
        self.assertIs(winner, good)

    def test_equal_head_quality_uses_leakage_before_global_score(self):
        reference = _head_view()
        baseline = score_face_hair_rc(
            [reference] * 10,
            [reference] * 10,
            view_names=[f"view_{index}" for index in range(10)],
        )

        smeared_views = list(baseline.views)
        smeared_views[7] = replace(
            smeared_views[7],
            face_leakage=0.30,
            hair_leakage=0.08,
        )
        globally_stronger_but_smeared = replace(
            baseline,
            score=98.0,
            raw_score=98.0,
            mean_view_score=99.0,
            worst_quartile_score=80.0,
            worst_quartile_face_leakage=0.10,
            worst_quartile_hair_leakage=0.05,
            views=tuple(smeared_views),
        )

        clean_views = tuple(
            replace(view, face_leakage=0.06, hair_leakage=0.05)
            for view in baseline.views
        )
        globally_weaker_but_clean = replace(
            baseline,
            score=78.0,
            raw_score=78.0,
            mean_view_score=82.0,
            worst_quartile_score=80.0,
            worst_quartile_face_leakage=0.06,
            worst_quartile_hair_leakage=0.05,
            views=clean_views,
        )

        self.assertEqual(
            rank_face_hair_candidates(
                {
                    "higher_global_score": globally_stronger_but_smeared,
                    "cleaner_face_hair": globally_weaker_but_clean,
                }
            ),
            ("cleaner_face_hair", "higher_global_score"),
        )

    def test_ranking_uses_worst_quartile_before_average_score(self):
        reference = _head_view()
        baseline = score_face_hair_rc(
            [reference] * 10,
            [reference] * 10,
            view_names=[f"view_{index}" for index in range(10)],
        )
        common_views = tuple(
            replace(view, face_leakage=0.09, hair_leakage=0.08)
            for view in baseline.views
        )
        worse_quartile = replace(
            baseline,
            score=96.0,
            raw_score=96.0,
            mean_view_score=98.0,
            worst_quartile_score=75.0,
            worst_quartile_face_leakage=0.09,
            worst_quartile_hair_leakage=0.08,
            views=common_views,
        )
        better_quartile = replace(
            baseline,
            score=80.0,
            raw_score=80.0,
            mean_view_score=82.0,
            worst_quartile_score=79.0,
            worst_quartile_face_leakage=0.06,
            worst_quartile_hair_leakage=0.05,
            views=common_views,
        )

        self.assertEqual(
            rank_face_hair_candidates(
                {
                    "higher_average": worse_quartile,
                    "better_worst_quartile": better_quartile,
                }
            ),
            ("better_worst_quartile", "higher_average"),
        )

    def test_washed_zero_leakage_fallback_loses_to_textured_candidate(self):
        reference = _head_view()
        baseline = score_face_hair_rc(
            [reference] * 10,
            [reference] * 10,
            view_names=[f"view_{index}" for index in range(10)],
        )

        washed_views = tuple(
            replace(
                view,
                score=38.0,
                semantic_score=55.0,
                color_score=15.59,
                detail_score=26.31,
                face_leakage=0.0,
                hair_leakage=0.0,
            )
            for view in baseline.views
        )
        washed_fallback = replace(
            baseline,
            score=38.95,
            raw_score=38.95,
            passed_hard_gates=False,
            hard_gate_failures=(
                "foreground_iou_below_minimum",
                "color_consistency_below_minimum",
                "boundary_detail_below_minimum",
            ),
            mean_view_score=39.1,
            worst_quartile_score=38.6,
            color_score=15.59,
            detail_score=26.31,
            worst_quartile_face_leakage=0.0,
            worst_quartile_hair_leakage=0.0,
            views=washed_views,
        )

        textured_views = tuple(
            replace(view, face_leakage=0.48, hair_leakage=0.14)
            for view in baseline.views
        )
        textured_candidate = replace(
            baseline,
            score=49.0,
            raw_score=51.88,
            passed_hard_gates=False,
            hard_gate_failures=(
                "foreground_iou_below_minimum",
                "face_leakage_above_limit",
                "hair_leakage_above_limit",
                "single_view_leakage_above_limit",
                "color_consistency_below_minimum",
            ),
            mean_view_score=54.86,
            worst_quartile_score=46.36,
            color_score=26.17,
            detail_score=69.14,
            worst_quartile_face_leakage=0.48,
            worst_quartile_hair_leakage=0.14,
            views=textured_views,
        )

        self.assertEqual(
            rank_face_hair_candidates(
                {"washed_fallback": washed_fallback, "textured": textured_candidate}
            ),
            ("textured", "washed_fallback"),
        )

    def test_real_like_head_quality_ranks_current_before_regional_and_vertex(self):
        reference = _head_view()
        baseline = score_face_hair_rc(
            [reference] * 10,
            [reference] * 10,
            view_names=[f"view_{index}" for index in range(10)],
        )

        current_views = list(baseline.views)
        current_views[0] = replace(
            current_views[0],
            score=42.539,
            face_leakage=0.66,
            hair_leakage=0.18,
        )
        current_views = tuple(
            replace(view, face_leakage=0.553, hair_leakage=0.140)
            if index
            else view
            for index, view in enumerate(current_views)
        )
        current_f = replace(
            baseline,
            score=49.0,
            raw_score=51.883,
            passed_hard_gates=False,
            hard_gate_failures=(
                "foreground_iou_below_minimum",
                "face_leakage_above_limit",
                "hair_leakage_above_limit",
                "single_view_leakage_above_limit",
                "color_consistency_below_minimum",
            ),
            mean_view_score=54.856,
            worst_quartile_score=46.362,
            color_score=26.175,
            detail_score=69.136,
            worst_quartile_face_leakage=0.553,
            worst_quartile_hair_leakage=0.140,
            views=current_views,
        )

        regional_views = list(baseline.views)
        regional_views[0] = replace(
            regional_views[0],
            score=43.330,
            face_leakage=0.55,
            hair_leakage=0.13,
        )
        regional_views = tuple(
            replace(view, face_leakage=0.472, hair_leakage=0.100)
            if index
            else view
            for index, view in enumerate(regional_views)
        )
        washed_regional = replace(
            baseline,
            score=49.0,
            raw_score=51.5,
            passed_hard_gates=False,
            hard_gate_failures=(
                "foreground_iou_below_minimum",
                "face_leakage_above_limit",
                "single_view_leakage_above_limit",
                "color_consistency_below_minimum",
            ),
            mean_view_score=54.3,
            worst_quartile_score=45.805,
            color_score=28.771,
            detail_score=65.294,
            worst_quartile_face_leakage=0.472,
            worst_quartile_hair_leakage=0.100,
            views=regional_views,
        )

        vertex_views = tuple(
            replace(
                view,
                score=38.136,
                color_score=15.594,
                detail_score=26.308,
                face_leakage=0.0,
                hair_leakage=0.0,
            )
            for view in baseline.views
        )
        vertex_fallback = replace(
            baseline,
            score=38.686,
            raw_score=38.686,
            passed_hard_gates=False,
            hard_gate_failures=(
                "foreground_iou_below_minimum",
                "color_consistency_below_minimum",
                "boundary_detail_below_minimum",
            ),
            mean_view_score=39.0,
            worst_quartile_score=38.686,
            color_score=15.594,
            detail_score=26.308,
            worst_quartile_face_leakage=0.0,
            worst_quartile_hair_leakage=0.0,
            views=vertex_views,
        )

        self.assertEqual(
            rank_face_hair_candidates(
                {
                    "regional": washed_regional,
                    "vertex": vertex_fallback,
                    "current_f": current_f,
                }
            ),
            ("current_f", "regional", "vertex"),
        )

    def test_misaligned_shapes_are_rejected_instead_of_resized(self):
        reference = _head_view()

        with self.assertRaisesRegex(ValueError, "mismatched shapes"):
            score_face_hair_rc(reference[:100], reference)


if __name__ == "__main__":
    unittest.main()
