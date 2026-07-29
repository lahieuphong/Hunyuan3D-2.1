from __future__ import annotations

import unittest

import numpy as np

from hy3dshape.texture_bake.arm_corner_repair import (
    apply_arm_repair_plan_to_corners,
    build_arm_repair_plan,
)
from hy3dshape.texture_bake.semantics import SemanticClass


class ArmRepairPlanTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray(
            [
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.9, 0.0, 0.56),
                (-0.9, 0.0, 0.56),
                (0.9, 0.0, 0.64),
            ],
            dtype=np.float32,
        )
        skin = (0.86, 0.58, 0.48)
        cool = (0.05, 0.20, 0.55)
        self.samples = np.asarray(
            [
                [skin, skin, skin, cool, skin],
                [skin, cool, cool, skin, cool],
            ],
            dtype=np.float32,
        )

    def test_trusted_skin_overrides_wrist_height_and_other_labels_use_heuristic(
        self,
    ):
        labels = np.asarray(
            [
                SemanticClass.UNKNOWN,
                SemanticClass.UNKNOWN,
                SemanticClass.SKIN_LIKE,
                SemanticClass.WARM_SATURATED,
                SemanticClass.COOL_SATURATED,
            ],
            dtype=np.int16,
        )

        colors, strength, stats = build_arm_repair_plan(
            self.vertices,
            self.samples,
            expected_labels=labels,
            width_start_fraction=0.36,
            width_full_fraction=0.43,
            vertical_bottom_fraction=0.44,
            wrist_bottom_fraction=0.535,
            wrist_top_fraction=0.585,
            vertical_top_fraction=0.72,
            cool_value_scale=0.4,
        )

        self.assertGreater(strength[2], 0.0)
        self.assertGreater(colors[2, 0], colors[2, 2])
        self.assertGreater(strength[3], 0.0)
        self.assertGreater(colors[3, 2], colors[3, 0])
        self.assertGreater(strength[4], 0.0)
        self.assertGreater(colors[4, 2], colors[4, 0])
        self.assertEqual(stats.protected_surface_vertices, 0)


class ArmCornerRepairTests(unittest.TestCase):
    def test_shared_vertex_repairs_visible_arm_corner_not_hidden_cloth_corner(self):
        vertices = np.asarray(
            [
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.9, 0.0, 0.60),
            ],
            dtype=np.float32,
        )
        loop_vertices = np.asarray((2, 2), dtype=np.int32)
        polygon_for_loop = np.asarray((0, 1), dtype=np.int32)
        vertex_visibility = np.zeros((3, 4), dtype=bool)
        vertex_visibility[2, 1] = True
        polygon_visibility = np.zeros((2, 4), dtype=bool)
        polygon_visibility[0, 1] = True
        base_colors = np.asarray(
            ((0.8, 0.1, 0.02), (0.8, 0.1, 0.02)),
            dtype=np.float32,
        )
        base_mix = np.zeros(2, dtype=np.float32)
        plan_colors = np.zeros((3, 3), dtype=np.float32)
        plan_colors[2] = (0.85, 0.55, 0.45)
        plan_strength = np.zeros(3, dtype=np.float32)
        plan_strength[2] = 1.0

        colors, mix, stats = apply_arm_repair_plan_to_corners(
            vertices,
            loop_vertices,
            polygon_for_loop,
            vertex_visibility,
            polygon_visibility,
            base_colors,
            base_mix,
            plan_colors,
            plan_strength,
        )

        np.testing.assert_allclose(colors[0], plan_colors[2])
        np.testing.assert_allclose(colors[1], base_colors[1])
        np.testing.assert_allclose(mix, (1.0, 0.0))
        self.assertEqual(stats.applied_corners, 1)
        self.assertEqual(stats.hidden_rejected_corners, 1)
        self.assertEqual(stats.applied_outside_visibility, 0)

    def test_left_and_right_choose_symmetric_outward_channels(self):
        vertices = np.asarray(
            [
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.9, 0.0, 0.6),
                (-0.9, 0.0, 0.6),
            ],
            dtype=np.float32,
        )
        loops = np.asarray((2, 3), dtype=np.int32)
        polygons = np.asarray((0, 1), dtype=np.int32)
        vertex_visibility = np.zeros((4, 4), dtype=bool)
        vertex_visibility[2, 1] = True
        vertex_visibility[3, 3] = True
        polygon_visibility = np.zeros((2, 4), dtype=bool)
        polygon_visibility[0, 1] = True
        polygon_visibility[1, 3] = True
        plan_colors = np.full((4, 3), 0.5, dtype=np.float32)
        plan_strength = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float32)

        _, mix, stats = apply_arm_repair_plan_to_corners(
            vertices,
            loops,
            polygons,
            vertex_visibility,
            polygon_visibility,
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            plan_colors,
            plan_strength,
        )

        np.testing.assert_allclose(mix, (1.0, 1.0))
        self.assertEqual(stats.applied_corners, 2)
        self.assertEqual(stats.applied_outside_visibility, 0)


if __name__ == "__main__":
    unittest.main()
