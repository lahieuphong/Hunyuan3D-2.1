"""Regression coverage for face-detail-aware RC candidate ranking."""

from __future__ import annotations

import unittest

from hy3dshape.texture_bake.face_hair_rc import (
    FaceHairRCResult,
    FaceHairRCView,
    rank_face_hair_candidates,
)


def _candidate(
    *,
    worst_quartile: float,
    detail: float,
    face_leakage: float,
    hair_leakage: float,
) -> FaceHairRCResult:
    view = FaceHairRCView(
        name="front",
        score=worst_quartile,
        semantic_score=55.0,
        color_score=27.0,
        detail_score=detail,
        foreground_iou=0.75,
        face_leakage=face_leakage,
        hair_leakage=hair_leakage,
        face_pixels=4000,
        hair_pixels=10000,
    )
    return FaceHairRCResult(
        score=49.0,
        raw_score=51.0,
        passed_hard_gates=False,
        hard_gate_failures=(
            "face_leakage_above_limit",
            "color_consistency_below_minimum",
        ),
        mean_view_score=54.0,
        worst_quartile_score=worst_quartile,
        semantic_score=55.0,
        color_score=27.0,
        detail_score=detail,
        foreground_iou=0.75,
        worst_quartile_face_leakage=face_leakage,
        worst_quartile_hair_leakage=hair_leakage,
        views=(view,),
    )


class HeadClarityRankingTests(unittest.TestCase):
    def test_small_quartile_gain_cannot_hide_washed_face_detail(self):
        clearer = _candidate(
            worst_quartile=46.362,
            detail=69.136,
            face_leakage=0.553,
            hair_leakage=0.140,
        )
        washed_but_lower_leakage = _candidate(
            worst_quartile=46.744,
            detail=64.063,
            face_leakage=0.434,
            hair_leakage=0.091,
        )

        self.assertEqual(
            rank_face_hair_candidates(
                {
                    "washed_but_lower_leakage": washed_but_lower_leakage,
                    "clearer_face_detail": clearer,
                }
            ),
            ("clearer_face_detail", "washed_but_lower_leakage"),
        )


if __name__ == "__main__":
    unittest.main()
