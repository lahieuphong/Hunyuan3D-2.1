"""Tests for feathered outer-arm palette repair."""

from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.region_palette import apply_arm_palette_repair


class ArmPaletteRepairTests(unittest.TestCase):
    def test_repairs_only_outer_arm_band_and_keeps_inner_surface(self):
        vertices = np.asarray(
            (
                (-1.0, 0.0, 0.2),
                (-0.8, 0.0, 0.2),
                (0.0, 0.0, 0.2),
                (1.0, 0.0, 0.2),
            )
        )
        samples = np.asarray(
            (
                (
                    (0.8, 0.5, 0.4),
                    (0.8, 0.5, 0.4),
                    (0.8, 0.5, 0.4),
                    (0.8, 0.5, 0.4),
                ),
                (
                    (0.1, 0.2, 0.8),
                    (0.1, 0.2, 0.8),
                    (0.1, 0.2, 0.8),
                    (0.1, 0.2, 0.8),
                ),
            )
        )
        fallback = np.zeros((4, 3), dtype=np.float32)
        mix = np.zeros(4, dtype=np.float32)

        colors, repaired_mix, stats = apply_arm_palette_repair(
            vertices,
            samples,
            fallback,
            mix,
            width_start_fraction=0.30,
            width_full_fraction=0.45,
            vertical_bottom_fraction=0.0,
            wrist_bottom_fraction=0.3,
            wrist_top_fraction=0.4,
            vertical_top_fraction=1.0,
        )

        self.assertEqual(stats.repaired_vertices, 3)
        self.assertEqual(repaired_mix[2], 0.0)
        self.assertEqual(repaired_mix[0], 1.0)
        self.assertEqual(repaired_mix[3], 1.0)
        self.assertGreater(repaired_mix[1], 0.0)
        np.testing.assert_allclose(colors[2], (0, 0, 0))

    def test_wrist_zone_uses_cool_palette(self):
        vertices = np.asarray(((-1.0, 0.0, 0.0), (1.0, 0.0, 1.0)))
        samples = np.asarray(
            (
                ((0.8, 0.5, 0.4), (0.1, 0.2, 0.8)),
                ((0.8, 0.5, 0.4), (0.1, 0.2, 0.8)),
            )
        )
        fallback = np.zeros((2, 3), dtype=np.float32)
        mix = np.zeros(2, dtype=np.float32)

        colors, repaired_mix, _ = apply_arm_palette_repair(
            vertices,
            samples,
            fallback,
            mix,
            width_start_fraction=0.1,
            width_full_fraction=0.2,
            vertical_bottom_fraction=0.0,
            wrist_bottom_fraction=0.4,
            wrist_top_fraction=0.6,
            vertical_top_fraction=1.0,
        )

        # Neither endpoint sits in the wrist band; they remain skin-colored.
        self.assertTrue(np.all(repaired_mix == 1.0))
        self.assertGreater(colors[0, 0], colors[0, 2])


if __name__ == "__main__":
    unittest.main()
