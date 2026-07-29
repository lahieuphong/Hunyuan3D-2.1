from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.semantic_prior import (
    apply_surface_semantic_prior,
)
from hy3dshape.texture_bake.semantics import SemanticClass


class SemanticSeedAgreementTests(unittest.TestCase):
    def _run(self, colors, weights):
        vertex_count = colors.shape[1]
        return apply_surface_semantic_prior(
            colors,
            weights,
            np.empty((0, 2), dtype=np.int32),
            np.tile((0.0, 0.0, 1.0), (vertex_count, 1)),
            prior_iterations=0,
        )

    def test_one_dominant_wrong_view_cannot_create_a_trusted_seed(self):
        warm = (0.82, 0.12, 0.03)
        skin = (0.86, 0.58, 0.48)
        colors = np.asarray([[warm], [skin]], dtype=np.float32)
        weights = np.asarray(((0.20, 0.80),), dtype=np.float32)

        result = self._run(colors, weights)

        self.assertEqual(result.report["trusted_seed_vertices"], 0)
        self.assertEqual(
            result.surface_labels[0],
            int(SemanticClass.UNKNOWN),
        )

    def test_two_agreeing_views_win_even_if_wrong_view_has_more_weight(self):
        warm = (0.82, 0.12, 0.03)
        skin = (0.86, 0.58, 0.48)
        colors = np.asarray([[warm], [warm], [skin]], dtype=np.float32)
        weights = np.asarray(((0.08, 0.08, 0.84),), dtype=np.float32)

        result = self._run(colors, weights)

        self.assertEqual(result.report["trusted_seed_vertices"], 1)
        self.assertEqual(
            result.surface_labels[0],
            int(SemanticClass.WARM_SATURATED),
        )
        self.assertEqual(result.adjusted_weights[0, 2], 0.0)

    def test_subthreshold_view_does_not_count_as_semantic_vote(self):
        warm = (0.82, 0.12, 0.03)
        skin = (0.86, 0.58, 0.48)
        colors = np.asarray([[warm], [warm], [skin]], dtype=np.float32)
        weights = np.asarray(((0.50, 0.49, 0.01),), dtype=np.float32)

        result = self._run(colors, weights)

        self.assertEqual(result.report["trusted_seed_vertices"], 1)
        self.assertEqual(
            result.surface_labels[0],
            int(SemanticClass.WARM_SATURATED),
        )


if __name__ == "__main__":
    unittest.main()
