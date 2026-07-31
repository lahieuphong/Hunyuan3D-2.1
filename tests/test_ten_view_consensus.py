from __future__ import annotations

import argparse
import unittest

import numpy as np

from hy3dshape.texture_bake.ten_view_consensus import (
    TEN_VIEW_ANGLES,
    add_regional_ownership_arguments,
    blend_consensus_colors,
    restore_chroma,
    robust_consensus_weights,
    ten_view_frames,
)


class TenViewConsensusTests(unittest.TestCase):
    def test_regional_ownership_is_default_on_with_explicit_opt_out(self):
        parser = argparse.ArgumentParser()
        add_regional_ownership_arguments(parser)

        self.assertTrue(parser.parse_args([]).enable_regional_ownership)
        self.assertTrue(
            parser.parse_args(
                ["--enable-regional-ownership"]
            ).enable_regional_ownership
        )
        self.assertFalse(
            parser.parse_args(
                ["--disable-regional-ownership"]
            ).enable_regional_ownership
        )

    def test_camera_frames_cover_all_declared_views(self):
        frames = ten_view_frames()

        self.assertEqual(tuple(frames), tuple(TEN_VIEW_ANGLES))
        np.testing.assert_allclose(
            frames["front"].to_camera,
            (0.0, -1.0, 0.0),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            frames["back"].to_camera,
            (0.0, 1.0, 0.0),
            atol=1.0e-7,
        )
        self.assertGreater(frames["high_front"].to_camera[2], 0.0)

    def test_isolated_skin_sample_is_suppressed_on_orange_surface(self):
        orange = np.asarray((0.92, 0.28, 0.025), dtype=np.float64)
        skin = np.asarray((0.98, 0.67, 0.45), dtype=np.float64)
        colors = np.asarray(
            [[orange, orange * 0.96, skin]],
            dtype=np.float64,
        )
        weights = np.asarray([[0.65, 0.55, 0.90]], dtype=np.float64)

        adjusted, report = robust_consensus_weights(colors, weights)
        blended, valid = blend_consensus_colors(colors, adjusted)

        self.assertTrue(valid[0])
        self.assertLess(adjusted[0, 2], 0.20)
        self.assertGreater(adjusted[0, 0] + adjusted[0, 1], 0.80)
        self.assertLess(
            np.linalg.norm(blended[0] - orange),
            np.linalg.norm(blended[0] - skin),
        )
        self.assertGreaterEqual(report["strongly_suppressed_entries"], 1)

    def test_single_visible_view_is_preserved(self):
        colors = np.asarray(
            [[[0.1, 0.3, 0.8], [0.9, 0.2, 0.1]]],
            dtype=np.float64,
        )
        weights = np.asarray([[0.0, 0.7]], dtype=np.float64)

        adjusted, _ = robust_consensus_weights(colors, weights)

        np.testing.assert_allclose(adjusted, [[0.0, 1.0]], atol=1.0e-7)

    def test_winner_mix_preserves_source_palette(self):
        colors = np.asarray(
            [[[0.95, 0.24, 0.02], [0.72, 0.38, 0.12]]],
            dtype=np.float64,
        )
        weights = np.asarray([[0.8, 0.2]], dtype=np.float64)

        blended, _ = blend_consensus_colors(
            colors,
            weights,
            winner_mix=0.25,
        )

        self.assertGreater(blended[0, 0], 0.90)
        self.assertLess(blended[0, 1], 0.30)

    def test_chroma_restore_leaves_neutral_and_strengthens_orange(self):
        colors = np.asarray(
            [[0.45, 0.45, 0.45], [0.70, 0.34, 0.10]],
            dtype=np.float64,
        )

        restored = restore_chroma(colors, amount=0.12)

        np.testing.assert_allclose(restored[0], colors[0], atol=1.0e-6)
        self.assertGreater(
            restored[1].max() - restored[1].min(),
            colors[1].max() - colors[1].min(),
        )


if __name__ == "__main__":
    unittest.main()
