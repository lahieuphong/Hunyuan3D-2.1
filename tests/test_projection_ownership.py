from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.projection_ownership import (
    apply_side_garment_ownership_guard,
)


class SideGarmentOwnershipTests(unittest.TestCase):
    def _fixture(self):
        vertices = np.asarray(
            [
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.4, 0.0, 0.52),
                (0.42, 0.0, 0.53),
                (0.8, 0.0, 0.52),
                (0.4, 0.0, 0.62),
                (-0.4, 0.0, 0.52),
            ],
            dtype=np.float32,
        )
        warm = (0.82, 0.12, 0.03)
        skin = (0.86, 0.58, 0.48)
        dark = (0.02, 0.02, 0.02)
        samples = np.full((4, len(vertices), 3), warm, dtype=np.float32)
        samples[1, 2] = skin
        samples[1, 3] = dark
        samples[1, 4] = skin
        samples[1, 5] = skin
        samples[3, 6] = skin
        weights = np.zeros((len(vertices), 4), dtype=np.float32)
        weights[:, 0] = 0.25
        weights[2:, 1] = 0.75
        weights[6, 1] = 0.0
        weights[6, 3] = 0.75
        edges = np.asarray(((2, 3),), dtype=np.int32)
        return vertices, samples, weights, edges

    def test_rejects_dominant_side_arm_sample_and_expands_to_outline(self):
        vertices, samples, weights, edges = self._fixture()

        adjusted, repaired, stats = apply_side_garment_ownership_guard(
            vertices,
            samples,
            weights,
            edges,
        )

        self.assertTrue(repaired[2])
        self.assertTrue(repaired[3], "dark outline neighbor should be absorbed")
        self.assertEqual(adjusted[2, 1], 0.0)
        self.assertEqual(adjusted[3, 1], 0.0)
        self.assertEqual(adjusted[2, 0], weights[2, 0])
        self.assertEqual(stats.core_vertices, 2)
        self.assertEqual(stats.expanded_vertices, 1)

    def test_preserves_outer_real_arm_belt_and_both_sides_symmetrically(self):
        vertices, samples, weights, edges = self._fixture()

        adjusted, repaired, stats = apply_side_garment_ownership_guard(
            vertices,
            samples,
            weights,
            edges,
        )

        self.assertFalse(repaired[4], "outer real arm must be preserved")
        self.assertFalse(repaired[5], "belt-height sample must be preserved")
        self.assertTrue(repaired[6], "negative side should use Right view")
        np.testing.assert_allclose(adjusted[4], weights[4])
        np.testing.assert_allclose(adjusted[5], weights[5])
        self.assertEqual(adjusted[6, 3], 0.0)
        self.assertEqual(stats.positive_side_vertices, 2)
        self.assertEqual(stats.negative_side_vertices, 1)

    def test_weight_ratio_prevents_ambiguous_rejection(self):
        vertices, samples, weights, edges = self._fixture()
        weights[2, 0] = 0.45
        weights[2, 1] = 0.55

        adjusted, repaired, stats = apply_side_garment_ownership_guard(
            vertices,
            samples,
            weights,
            edges,
        )

        self.assertFalse(repaired[2])
        np.testing.assert_allclose(adjusted[2], weights[2])
        self.assertEqual(stats.core_vertices, 1, "other side core remains")


if __name__ == "__main__":
    unittest.main()
