"""Tests for surface-local palette rejection before texture baking."""

from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.semantic_prior import (
    apply_surface_semantic_prior,
)


class SurfaceSemanticPriorTests(unittest.TestCase):
    def test_rejects_skin_singleton_inside_warm_cloth_surface(self):
        colors = np.asarray(
            (
                ((0.9, 0.25, 0.05), (0.88, 0.65, 0.55)),
                ((0.85, 0.20, 0.04), (0.0, 0.0, 0.0)),
            )
        )
        weights = np.asarray(((1.0, 1.0), (1.0, 0.0)))
        edges = np.asarray(((0, 1),))
        normals = np.asarray(((0, 0, 1), (0, 0, 1)))

        result = apply_surface_semantic_prior(
            colors,
            weights,
            edges,
            normals,
        )

        np.testing.assert_allclose(result.adjusted_weights[1], (0.0, 0.0))
        self.assertEqual(result.report["singleton_rejected_vertices"], 1)

    def test_keeps_dark_line_art_compatible_with_warm_surface(self):
        colors = np.asarray(
            (
                ((0.9, 0.25, 0.05), (0.05, 0.04, 0.03)),
                ((0.85, 0.20, 0.04), (0.0, 0.0, 0.0)),
            )
        )
        weights = np.asarray(((1.0, 1.0), (1.0, 0.0)))
        edges = np.asarray(((0, 1),))
        normals = np.asarray(((0, 0, 1), (0, 0, 1)))

        result = apply_surface_semantic_prior(
            colors,
            weights,
            edges,
            normals,
        )

        self.assertGreater(result.adjusted_weights[1, 0], 0.0)
        self.assertEqual(result.report["singleton_rejected_vertices"], 0)

    def test_creates_barrier_between_different_trusted_regions(self):
        colors = np.asarray(
            (
                ((0.9, 0.25, 0.05), (0.25, 0.35, 0.85)),
                ((0.85, 0.20, 0.04), (0.20, 0.30, 0.80)),
            )
        )
        weights = np.ones((2, 2))
        edges = np.asarray(((0, 1),))
        normals = np.asarray(((0, 0, 1), (0, 0, 1)))

        result = apply_surface_semantic_prior(
            colors,
            weights,
            edges,
            normals,
        )

        self.assertEqual(len(result.diffusion_edges), 0)
        self.assertEqual(result.report["semantic_barrier_edges"], 1)


if __name__ == "__main__":
    unittest.main()
