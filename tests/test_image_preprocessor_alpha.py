"""Regression tests for robust RGBA subject recentering."""

from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.preprocessors import ImageProcessorV2


class ImagePreprocessorAlphaTests(unittest.TestCase):
    def test_recenter_ignores_nearly_transparent_edge_speckles(self):
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        image[30:70, 40:60, :3] = (240, 120, 40)
        image[30:70, 40:60, 3] = 255
        image[0, 0, 3] = 1

        _, alpha = ImageProcessorV2.recenter(image, border_ratio=0.2)

        foreground = alpha[..., 0] > 127
        rows, columns = np.nonzero(foreground)
        self.assertGreater(len(rows), 2_000)
        self.assertLess(abs(float(rows.mean()) - 50.0), 2.0)
        self.assertLess(abs(float(columns.mean()) - 50.0), 2.0)

    def test_recenter_falls_back_for_intentionally_faint_alpha(self):
        image = np.zeros((32, 32, 4), dtype=np.uint8)
        image[8:24, 10:22, :3] = 255
        image[8:24, 10:22, 3] = 4

        recentered, alpha = ImageProcessorV2.recenter(image, border_ratio=0.2)

        self.assertEqual(recentered.shape, (32, 32, 3))
        self.assertEqual(alpha.shape, (32, 32, 1))
        self.assertGreater(np.count_nonzero(alpha), 0)
