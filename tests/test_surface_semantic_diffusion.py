"""Tests for conservative semantic-label propagation on mesh surfaces."""

from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.surface_semantics import (
    diffuse_surface_labels,
)


class SurfaceLabelDiffusionTests(unittest.TestCase):
    def test_propagates_nearest_label_by_bfs_layer(self):
        labels = np.asarray((3, 0, 0, 4), dtype=np.int8)
        known = np.asarray((True, False, False, True))
        edges = np.asarray(((0, 1), (1, 2), (2, 3)))
        normals = np.tile((0.0, 0.0, 1.0), (4, 1))

        result, filled, stats = diffuse_surface_labels(
            labels,
            known,
            edges,
            normals,
            max_iterations=4,
        )

        np.testing.assert_array_equal(result, (3, 3, 4, 4))
        np.testing.assert_array_equal(filled, (False, True, True, False))
        self.assertEqual(stats.filled_per_hop, (2,))

    def test_tied_votes_remain_unresolved(self):
        labels = np.asarray((2, 0, 3), dtype=np.int8)
        known = np.asarray((True, False, True))
        edges = np.asarray(((0, 1), (2, 1)))
        normals = np.tile((0.0, 0.0, 1.0), (3, 1))

        result, filled, stats = diffuse_surface_labels(
            labels,
            known,
            edges,
            normals,
            minimum_vote_fraction=0.6,
        )

        np.testing.assert_array_equal(result, labels)
        self.assertFalse(filled.any())
        self.assertEqual(stats.remaining, 1)

    def test_majority_vote_wins(self):
        labels = np.asarray((2, 2, 3, 0), dtype=np.int8)
        known = np.asarray((True, True, True, False))
        edges = np.asarray(((0, 3), (1, 3), (2, 3)))
        normals = np.tile((0.0, 0.0, 1.0), (4, 1))

        result, _, _ = diffuse_surface_labels(
            labels,
            known,
            edges,
            normals,
            minimum_vote_fraction=0.6,
        )

        self.assertEqual(result[3], 2)

    def test_normal_barrier_prevents_cross_fold_fill(self):
        labels = np.asarray((2, 0), dtype=np.int8)
        known = np.asarray((True, False))
        edges = np.asarray(((0, 1),))
        normals = np.asarray(((0, 0, 1), (1, 0, 0)))

        result, filled, stats = diffuse_surface_labels(
            labels,
            known,
            edges,
            normals,
            minimum_normal_dot=0.5,
        )

        np.testing.assert_array_equal(result, labels)
        self.assertFalse(filled.any())
        self.assertEqual(stats.usable_edges, 0)


if __name__ == "__main__":
    unittest.main()
