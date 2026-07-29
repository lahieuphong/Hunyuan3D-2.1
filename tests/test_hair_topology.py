"""Tests for conservative topology-only polygon-mask cleanup."""

from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.hair_topology import (
    expand_enclosed_polygon_mask,
)


class PolygonMaskExpansionTests(unittest.TestCase):
    def test_fills_triangle_whose_all_vertices_touch_mask(self):
        # Center triangle 3 uses vertices 0, 1, 2. Each vertex touches one
        # different marked triangle, so the center is an enclosed hole.
        loops = np.asarray(
            (
                0, 3, 4,
                1, 5, 6,
                2, 7, 8,
                0, 1, 2,
            )
        )
        totals = np.asarray((3, 3, 3, 3))
        mask = np.asarray((True, True, True, False))
        centers = np.asarray(
            ((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1))
        )

        result, stats = expand_enclosed_polygon_mask(
            mask,
            loops,
            totals,
            centers,
            minimum_z=0.5,
        )

        self.assertTrue(result[3])
        self.assertEqual(stats.added, 1)

    def test_does_not_cross_two_vertex_boundary(self):
        loops = np.asarray((0, 1, 2, 0, 1, 3))
        totals = np.asarray((3, 3))
        mask = np.asarray((True, False))
        centers = np.asarray(((0, 0, 1), (0, 0, 1)))

        result, stats = expand_enclosed_polygon_mask(
            mask,
            loops,
            totals,
            centers,
            minimum_z=0.5,
        )

        np.testing.assert_array_equal(result, mask)
        self.assertEqual(stats.added, 0)

    def test_respects_minimum_height(self):
        loops = np.asarray((0, 3, 4, 1, 5, 6, 2, 7, 8, 0, 1, 2))
        totals = np.asarray((3, 3, 3, 3))
        mask = np.asarray((True, True, True, False))
        centers = np.asarray(
            ((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 0.2))
        )

        result, _ = expand_enclosed_polygon_mask(
            mask,
            loops,
            totals,
            centers,
            minimum_z=0.5,
        )

        self.assertFalse(result[3])


if __name__ == "__main__":
    unittest.main()
