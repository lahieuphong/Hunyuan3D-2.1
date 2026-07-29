"""Regression tests for visibility-aware four-view texture-bake primitives."""

from __future__ import annotations

import importlib.util
import unittest

import hy3dshape
import numpy as np

from hy3dshape.texture_bake import (
    CANONICAL_VIEW_FRAMES,
    SemanticClass,
    alpha_bounds,
    alpha_confidence_map,
    angular_confidence,
    apply_semantic_color_guards,
    blend_view_colors,
    blender_raycast_visibility,
    classify_anime_colors,
    compute_view_confidence,
    depth_visibility_confidence,
    diffuse_surface_colors,
    fit_four_view_calibrations,
    fit_orthographic_from_alpha,
    rasterize_depth_buffer,
)


def box_vertices() -> np.ndarray:
    return np.asarray(
        [(x, y, z) for x in (-1.0, 1.0) for y in (-0.5, 0.5) for z in (-2.0, 2.0)],
        dtype=np.float64,
    )


def four_view_alpha() -> dict[str, np.ndarray]:
    front = np.zeros((100, 100), dtype=np.uint8)
    side = np.zeros_like(front)
    # Inclusive bounds are x=30..70/y=10..90 and x=40..60/y=10..90.
    front[10:91, 30:71] = 255
    side[10:91, 40:61] = 255
    return {
        "front": front,
        "left": side,
        "back": front.copy(),
        "right": side.copy(),
    }


class PackageLayoutTests(unittest.TestCase):
    def test_repository_extensions_keep_core_shape_modules_importable(self):
        self.assertIsNotNone(importlib.util.find_spec("hy3dshape.utils"))

    def test_repository_extensions_preserve_core_root_exports(self):
        pipeline = hy3dshape.Hunyuan3DDiTFlowMatchingPipeline
        self.assertEqual(pipeline.__module__, "hy3dshape.pipelines")


class CalibrationTests(unittest.TestCase):
    def test_alpha_bounds_accepts_rgba(self):
        rgba = np.zeros((12, 16, 4), dtype=np.uint8)
        rgba[2:10, 4:13, 3] = 255
        bounds = alpha_bounds(rgba)
        self.assertEqual(
            (bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max),
            (4, 2, 12, 9),
        )

    def test_shared_height_fit_preserves_side_view_width(self):
        calibrations = fit_four_view_calibrations(
            four_view_alpha(),
            box_vertices(),
            fit_mode="height",
            shared_scale=True,
        )
        for calibration in calibrations.values():
            self.assertAlmostEqual(calibration.pixels_per_unit_u, 20.0)
            self.assertAlmostEqual(calibration.pixels_per_unit_v, 20.0)

        front = calibrations["front"].project(
            np.asarray([(-1, 0, 2), (1, 0, -2)], dtype=np.float64)
        )
        np.testing.assert_allclose(front.pixels, ((30, 10), (70, 90)))
        self.assertTrue(front.inside_image.all())

        left = calibrations["left"].project(
            np.asarray([(0, -0.5, 2), (0, 0.5, -2)], dtype=np.float64)
        )
        np.testing.assert_allclose(left.pixels, ((40, 10), (60, 90)))

        back = calibrations["back"].project(
            np.asarray([(1, 0, 2), (-1, 0, -2)], dtype=np.float64)
        )
        np.testing.assert_allclose(back.pixels, ((30, 10), (70, 90)))

    def test_anisotropic_fit_is_explicit_not_default(self):
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[10:91, 20:81] = 255
        calibration = fit_orthographic_from_alpha(
            alpha,
            box_vertices(),
            "front",
            fit_mode="anisotropic",
        )
        self.assertAlmostEqual(calibration.pixels_per_unit_u, 30.0)
        self.assertAlmostEqual(calibration.pixels_per_unit_v, 20.0)

    def test_calibration_round_trip_plane_coordinates(self):
        calibration = fit_four_view_calibrations(four_view_alpha(), box_vertices())[
            "right"
        ]
        projected = calibration.project(
            np.asarray([(0.25, -0.25, 0.5), (-0.25, 0.4, -1.5)])
        )
        restored = calibration.unproject_plane(projected.pixels)
        expected_u, expected_v, _ = calibration.frame.plane_coordinates(
            np.asarray([(0.25, -0.25, 0.5), (-0.25, 0.4, -1.5)])
        )
        np.testing.assert_allclose(restored, np.column_stack((expected_u, expected_v)))


class VisibilityTests(unittest.TestCase):
    def test_angular_confidence_rejects_back_and_grazing_faces(self):
        normals = np.asarray(
            [
                (0, -1, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, -0.5, 0.8660254),
                (0, 0, 0),
            ],
            dtype=np.float64,
        )
        confidence = angular_confidence(normals, CANONICAL_VIEW_FRAMES["front"])
        self.assertAlmostEqual(confidence[0], 1.0)
        self.assertEqual(confidence[1], 0.0)
        self.assertEqual(confidence[2], 0.0)
        self.assertGreater(confidence[3], 0.0)
        self.assertLess(confidence[3], 1.0)
        self.assertEqual(confidence[4], 0.0)

    def test_alpha_confidence_excludes_padding_and_feathers_inside(self):
        alpha = np.zeros((11, 11), dtype=np.uint8)
        alpha[2:9, 2:9] = 255
        confidence = alpha_confidence_map(alpha, feather_pixels=2.0)
        self.assertEqual(confidence[1, 5], 0.0)
        self.assertGreater(confidence[2, 5], 0.0)
        self.assertLess(confidence[2, 5], confidence[5, 5])
        self.assertAlmostEqual(float(confidence[5, 5]), 1.0)

    def test_depth_buffer_rejects_occluded_surface(self):
        alpha = np.zeros((11, 11), dtype=np.uint8)
        alpha[1:10, 1:10] = 255
        # Same front-facing triangle at two Y depths.  Camera is at -Y, so
        # y=-0.25 (depth +0.25) must win over y=+0.25 (depth -0.25).
        vertices = np.asarray(
            [
                (-1, -0.25, -1),
                (1, -0.25, -1),
                (0, -0.25, 1),
                (-1, 0.25, -1),
                (1, 0.25, -1),
                (0, 0.25, 1),
            ],
            dtype=np.float64,
        )
        faces = np.asarray(((3, 4, 5), (0, 1, 2)), dtype=np.int64)
        calibration = fit_orthographic_from_alpha(
            alpha, vertices, "front", fit_mode="height"
        )
        depth_buffer = rasterize_depth_buffer(vertices, faces, calibration)
        self.assertAlmostEqual(depth_buffer.depth[5, 5], 0.25)

        pixels = np.asarray(((5, 5), (5, 5)), dtype=np.float64)
        visibility = depth_visibility_confidence(
            pixels,
            np.asarray((0.25, -0.25)),
            depth_buffer,
            absolute_tolerance=1e-6,
            relative_tolerance=0.0,
            softness=0.01,
        )
        self.assertAlmostEqual(visibility[0], 1.0)
        self.assertLess(visibility[1], 1e-10)

        samples = np.asarray(((0, -0.25, 0), (0, 0.25, 0)))
        normals = np.asarray(((0, -1, 0), (0, -1, 0)))
        components = compute_view_confidence(
            samples,
            normals,
            calibration,
            alpha_confidence_map(alpha, feather_pixels=0),
            depth_buffer=depth_buffer,
        )
        self.assertGreater(components.combined[0], 0.99)
        self.assertLess(components.combined[1], 1e-10)

    def test_blend_does_not_fallback_when_every_view_is_invalid(self):
        colors = np.asarray(
            [
                ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
            ]
        )
        weights = np.asarray(((3.0, 0.0), (1.0, 0.0)))
        blended, normalized, valid = blend_view_colors(colors, weights)
        np.testing.assert_allclose(blended[0], (0.75, 0.0, 0.25))
        np.testing.assert_allclose(normalized[:, 0], (0.75, 0.25))
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        np.testing.assert_allclose(blended[1], (0.0, 0.0, 0.0))


class SurfaceColorDiffusionTests(unittest.TestCase):
    def test_propagates_one_frontier_per_hop_and_honors_iteration_limit(self):
        colors = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (np.nan, np.nan, np.nan),
                (np.nan, np.nan, np.nan),
                (np.nan, np.nan, np.nan),
            )
        )
        known = np.asarray((True, False, False, False))
        edges = np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int64)
        normals = np.tile((0.0, 0.0, 1.0), (4, 1))

        result, filled, stats = diffuse_surface_colors(
            colors,
            known,
            edges,
            normals,
            minimum_normal_dot=0.5,
            max_iterations=2,
        )

        np.testing.assert_array_equal(filled, (False, True, True, False))
        np.testing.assert_allclose(result[:3], ((1, 0, 0),) * 3)
        self.assertTrue(np.isnan(result[3]).all())
        self.assertEqual(stats.filled_per_hop, (1, 1))
        self.assertEqual(stats.max_hop, 2)
        self.assertEqual(stats.filled, 2)
        self.assertEqual(stats.remaining, 1)

    def test_normal_barrier_stops_color_at_a_sharp_fold(self):
        colors = np.asarray(((0.2, 0.4, 0.8), (0, 0, 0), (0, 0, 0)))
        known = np.asarray((True, False, False))
        edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
        normals = np.asarray(((0, 0, 1), (0, 0, 1), (1, 0, 0)))

        result, filled, stats = diffuse_surface_colors(
            colors,
            known,
            edges,
            normals,
            minimum_normal_dot=0.5,
            max_iterations=8,
        )

        np.testing.assert_array_equal(filled, (False, True, False))
        np.testing.assert_allclose(result[1], colors[0])
        np.testing.assert_allclose(result[2], colors[2])
        self.assertEqual(stats.usable_edges, 1)
        self.assertEqual(stats.remaining, 1)

    def test_receiver_is_filled_once_by_its_nearest_seed_layer(self):
        colors = np.asarray(((1.0, 0.0, 0.0), (0, 0, 0), (0, 0, 0), (0.0, 0.0, 1.0)))
        known = np.asarray((True, False, False, True))
        edges = np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int64)
        normals = np.tile((0.0, 0.0, 1.0), (4, 1))

        result, filled, stats = diffuse_surface_colors(
            colors, known, edges, normals, max_iterations=8
        )

        np.testing.assert_array_equal(filled, (False, True, True, False))
        np.testing.assert_allclose(result[1], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(result[2], (0.0, 0.0, 1.0))
        self.assertEqual(stats.filled_per_hop, (2,))

    def test_component_without_seed_stays_untouched(self):
        colors = np.asarray(
            (
                (0.9, 0.1, 0.1),
                (0.0, 0.0, 0.0),
                (0.25, 0.5, 0.75),
                (np.nan, np.nan, np.nan),
            )
        )
        original = colors.copy()
        known = np.asarray((True, False, False, False))
        edges = np.asarray(((0, 1), (2, 3)), dtype=np.int64)

        result, filled, stats = diffuse_surface_colors(
            colors,
            known,
            edges,
            None,
            minimum_normal_dot=None,
            max_iterations=8,
        )

        np.testing.assert_array_equal(filled, (False, True, False, False))
        np.testing.assert_allclose(result[1], colors[0])
        np.testing.assert_allclose(result[2], original[2])
        self.assertTrue(np.isnan(result[3]).all())
        self.assertEqual(stats.initially_missing, 3)
        self.assertEqual(stats.filled, 1)
        self.assertEqual(stats.remaining, 2)


class SemanticGuardTests(unittest.TestCase):
    def test_color_classifier_separates_character_palette_families(self):
        colors = np.asarray(
            [
                (10, 10, 12),
                (230, 180, 155),
                (235, 85, 20),
                (35, 80, 200),
                (245, 245, 245),
                (125, 125, 125),
            ],
            dtype=np.uint8,
        )
        labels = classify_anime_colors(colors)
        np.testing.assert_array_equal(
            labels,
            (
                SemanticClass.DARK_OR_INK,
                SemanticClass.SKIN_LIKE,
                SemanticClass.WARM_SATURATED,
                SemanticClass.COOL_SATURATED,
                SemanticClass.LIGHT_NEUTRAL,
                SemanticClass.MID_NEUTRAL,
            ),
        )

    def test_cross_view_consensus_downweights_orange_bleed_on_skin(self):
        colors = np.asarray(
            [
                ((230, 180, 155),),
                ((220, 165, 140),),
                ((245, 190, 160),),
                ((235, 85, 20),),
            ],
            dtype=np.uint8,
        )
        result = apply_semantic_color_guards(colors, np.ones((4, 1), dtype=np.float64))
        self.assertEqual(result.expected_labels[0], int(SemanticClass.SKIN_LIKE))
        self.assertAlmostEqual(result.consensus_strength[0], 0.75)
        self.assertTrue(np.allclose(result.adjusted_weights[:3, 0], 1.0))
        self.assertLess(result.adjusted_weights[3, 0], 0.5)

    def test_explicit_surface_semantics_override_cross_view_vote(self):
        colors = np.asarray(
            [
                ((230, 180, 155),),
                ((220, 165, 140),),
                ((235, 85, 20),),
            ],
            dtype=np.uint8,
        )
        result = apply_semantic_color_guards(
            colors,
            np.ones((3, 1)),
            expected_labels=np.asarray((int(SemanticClass.WARM_SATURATED),)),
        )
        self.assertLess(result.adjusted_weights[0, 0], 0.3)
        self.assertAlmostEqual(result.adjusted_weights[2, 0], 1.0)


class BlenderAdapterTests(unittest.TestCase):
    def test_raycast_adapter_detects_an_occluder_before_target(self):
        expected_object = object()

        class FakeScene:
            def ray_cast(self, depsgraph, origin, direction, *, distance):
                del depsgraph, distance
                origin_array = np.asarray(tuple(origin), dtype=np.float64)
                direction_array = np.asarray(tuple(direction), dtype=np.float64)
                travel = 10.0 if origin_array[0] < 0 else 9.0
                location = origin_array + direction_array * travel
                return (
                    True,
                    tuple(location),
                    (0.0, -1.0, 0.0),
                    0,
                    expected_object,
                    None,
                )

        result = blender_raycast_visibility(
            FakeScene(),
            object(),
            np.asarray(((-1, 0, 0), (1, 0, 0)), dtype=np.float64),
            (0, -1, 0),
            ray_distance=10.0,
            hit_tolerance=1e-4,
            softness=0.01,
            expected_object=expected_object,
        )
        np.testing.assert_array_equal(result.visible, (True, False))
        self.assertAlmostEqual(result.confidence[0], 1.0)
        self.assertLess(result.confidence[1], 1e-10)


if __name__ == "__main__":
    unittest.main()
