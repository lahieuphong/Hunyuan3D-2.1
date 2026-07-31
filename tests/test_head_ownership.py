"""Focused tests for adaptive cyan-hair/face semantic ownership."""

from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.head_ownership import (
    HeadSemanticClass,
    apply_head_color_guard,
    classify_head_ownership,
    estimate_head_palette,
    score_head_consistency,
)


class HeadPaletteTests(unittest.TestCase):
    def test_palette_adapts_to_reference_skin_and_turquoise_hair(self):
        skin = np.asarray((226, 174, 143), dtype=np.uint8)
        hair = np.asarray((14, 174, 190), dtype=np.uint8)
        orange = np.asarray((247, 105, 5), dtype=np.uint8)
        pixels = np.vstack(
            (
                np.tile(skin, (20, 1)),
                np.tile(hair, (30, 1)),
                np.tile(orange, (12, 1)),
            )
        )
        palette = estimate_head_palette(pixels)

        np.testing.assert_allclose(
            palette.skin_rgb,
            skin.astype(np.float64) / 255.0,
            atol=1.0 / 255.0,
        )
        np.testing.assert_allclose(
            palette.hair_rgb,
            hair.astype(np.float64) / 255.0,
            atol=1.0 / 255.0,
        )
        self.assertEqual(palette.skin_sample_count, 20)
        self.assertEqual(palette.hair_sample_count, 30)

    def test_dominant_cyan_mode_is_not_moved_by_blue_clothing(self):
        cyan = np.asarray((20, 190, 195), dtype=np.uint8)
        dark_blue = np.asarray((15, 55, 175), dtype=np.uint8)
        pixels = np.vstack(
            (
                np.tile(cyan, (40, 1)),
                np.tile(dark_blue, (12, 1)),
            )
        )
        palette = estimate_head_palette(pixels)

        np.testing.assert_allclose(
            palette.hair_rgb,
            cyan.astype(np.float64) / 255.0,
            atol=1.0 / 255.0,
        )


class HeadClassificationTests(unittest.TestCase):
    def test_classifies_skin_hair_ink_highlight_and_rejects_orange_cloth(self):
        pixels = np.asarray(
            [
                (224, 169, 137),
                (15, 180, 190),
                (8, 8, 10),
                (247, 247, 244),
                (250, 101, 3),
            ],
            dtype=np.uint8,
        )
        result = classify_head_ownership(pixels)

        self.assertEqual(result.labels[0], int(HeadSemanticClass.FACE))
        self.assertEqual(result.labels[1], int(HeadSemanticClass.HAIR))
        self.assertEqual(result.labels[2], int(HeadSemanticClass.INK))
        self.assertEqual(result.labels[3], int(HeadSemanticClass.HIGHLIGHT))
        self.assertEqual(result.labels[4], int(HeadSemanticClass.UNKNOWN))

    def test_geometry_prior_resolves_noisy_skin_and_hair_samples(self):
        pixels = np.asarray(
            [
                (190, 151, 127),
                (32, 130, 145),
            ],
            dtype=np.uint8,
        )
        face_prior = np.asarray((0.95, 0.05))
        hair_prior = 1.0 - face_prior
        result = classify_head_ownership(
            pixels,
            face_prior=face_prior,
            hair_prior=hair_prior,
        )

        self.assertGreater(result.face_probability[0], 0.9)
        self.assertGreater(result.hair_probability[1], 0.9)


class HeadGuardTests(unittest.TestCase):
    def test_guard_repairs_crossings_but_preserves_black_ink_and_alpha(self):
        skin = np.asarray((224, 170, 140, 255), dtype=np.uint8)
        cyan = np.asarray((15, 180, 190, 240), dtype=np.uint8)
        ink = np.asarray((5, 6, 7, 222), dtype=np.uint8)
        highlight = np.asarray((248, 248, 245, 200), dtype=np.uint8)
        # Cyan is incorrectly painted on face; skin is incorrectly painted
        # on hair. Ink and white highlights must be byte-identical.
        pixels = np.vstack((cyan, skin, ink, highlight))
        expected_face = np.asarray((1.0, 0.0, 1.0, 0.0))
        expected_hair = np.asarray((0.0, 1.0, 0.0, 1.0))

        result = apply_head_color_guard(
            pixels,
            expected_face,
            expected_hair,
        )

        self.assertEqual(result.corrected.dtype, np.uint8)
        np.testing.assert_array_equal(result.corrected[2], ink)
        np.testing.assert_array_equal(result.corrected[3], highlight)
        np.testing.assert_array_equal(result.corrected[:, 3], pixels[:, 3])
        self.assertGreater(result.corrected[0, 0], pixels[0, 0])
        self.assertGreater(result.corrected[0, 0], result.corrected[0, 2])
        self.assertGreater(result.corrected[1, 1], result.corrected[1, 0])
        self.assertGreater(result.replacement_strength[0], 0.8)
        self.assertGreater(result.replacement_strength[1], 0.8)
        self.assertEqual(result.repaired_sample_count, 2)

    def test_uncertain_boundary_is_not_repainted(self):
        cyan = np.asarray(((12, 174, 188),), dtype=np.uint8)
        result = apply_head_color_guard(
            cyan,
            np.asarray((0.5,)),
            np.asarray((0.5,)),
        )

        np.testing.assert_array_equal(result.corrected, cyan)
        self.assertEqual(result.repaired_sample_count, 0)

    def test_guard_improves_semantic_rc_score(self):
        clean_skin = np.asarray((225, 173, 143), dtype=np.uint8)
        clean_hair = np.asarray((15, 176, 190), dtype=np.uint8)
        pixels = np.vstack(
            (
                clean_hair,  # bleed on face
                clean_skin,
                clean_skin,  # bleed on hair
                clean_hair,
            )
        )
        expected_face = np.asarray((1.0, 1.0, 0.0, 0.0))
        expected_hair = 1.0 - expected_face
        before = score_head_consistency(
            pixels,
            expected_face,
            expected_hair,
        )
        guarded = apply_head_color_guard(
            pixels,
            expected_face,
            expected_hair,
        )
        after = score_head_consistency(
            guarded.corrected,
            expected_face,
            expected_hair,
            palette=guarded.observed.palette,
        )

        self.assertLess(before.rc_score, 60.0)
        self.assertGreater(after.rc_score, before.rc_score + 30.0)
        self.assertLess(after.face_hair_bleed, before.face_hair_bleed)
        self.assertLess(after.hair_face_bleed, before.hair_face_bleed)


if __name__ == "__main__":
    unittest.main()
