"""Visibility-safe arm palette repair applied in the mesh-corner domain.

The same 3D vertex may participate in a side-visible arm face and a
side-occluded garment face.  A vertex-domain override cannot distinguish
those uses and can stamp skin or wristband colors onto the garment behind the
arm.  This module builds a semantic repair plan per vertex, then applies it
per corner only when both the vertex and its polygon center are visible from
the outward side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantics import SemanticClass, classify_anime_colors


@dataclass(frozen=True)
class ArmRepairPlanStats:
    """Summary of semantic candidates in an outer-arm spatial band."""

    spatial_candidates: int
    repair_candidates: int
    protected_surface_vertices: int
    skin_vertices: int
    wrist_vertices: int
    skin_palette: tuple[float, float, float]
    cool_palette: tuple[float, float, float]


@dataclass(frozen=True)
class ArmCornerRepairStats:
    """Summary of the visibility gate applied to corner repair candidates."""

    candidate_corners: int
    side_supported_corners: int
    applied_corners: int
    full_strength_corners: int
    hidden_rejected_corners: int
    applied_outside_visibility: int


def build_arm_repair_plan(
    vertices: np.ndarray,
    sampled_colors: np.ndarray,
    *,
    expected_labels: np.ndarray | None = None,
    width_start_fraction: float = 0.36,
    width_full_fraction: float = 0.43,
    vertical_bottom_fraction: float = 0.44,
    wrist_bottom_fraction: float = 0.535,
    wrist_top_fraction: float = 0.585,
    vertical_top_fraction: float = 0.72,
    cool_value_scale: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, ArmRepairPlanStats]:
    """Build per-vertex palette colors and strengths without mutating colors.

    Trusted skin semantics always use the skin palette, even inside the
    heuristic wrist height. Trusted cool semantics use the wrist palette.
    Other semantic labels use the vertical heuristic. Anatomical ownership
    is enforced later per corner with outward vertex and polygon visibility.
    """

    points = np.asarray(vertices, dtype=np.float64)
    samples = np.asarray(sampled_colors)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if samples.ndim != 3 or samples.shape[1:] != (len(points), 3):
        raise ValueError("sampled_colors must have shape (views, N, 3)")
    if not 0.0 <= width_start_fraction < width_full_fraction <= 0.5:
        raise ValueError(
            "width fractions must satisfy 0 <= start < full <= 0.5"
        )
    if not (
        0.0
        <= vertical_bottom_fraction
        < wrist_bottom_fraction
        < wrist_top_fraction
        < vertical_top_fraction
        <= 1.0
    ):
        raise ValueError("vertical arm fractions must be strictly ordered")
    if not 0.0 < cool_value_scale <= 1.0:
        raise ValueError("cool_value_scale must be between 0 and 1")

    sample_labels = classify_anime_colors(samples)
    skin_samples = samples[
        sample_labels == int(SemanticClass.SKIN_LIKE)
    ]
    cool_samples = samples[
        sample_labels == int(SemanticClass.COOL_SATURATED)
    ]
    skin_palette = (
        np.median(skin_samples, axis=0)
        if len(skin_samples)
        else np.asarray((0.72, 0.48, 0.36), dtype=np.float32)
    )
    cool_palette = (
        np.median(cool_samples, axis=0)
        if len(cool_samples)
        else np.asarray((0.04, 0.10, 0.30), dtype=np.float32)
    )
    cool_palette = np.clip(cool_palette * cool_value_scale, 0.0, 1.0)

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    x_span = max(maximum[0] - minimum[0], 1.0e-8)
    z_span = max(maximum[2] - minimum[2], 1.0e-8)
    x_mid = 0.5 * (minimum[0] + maximum[0])
    lateral = np.abs(points[:, 0] - x_mid) / x_span
    vertical = (points[:, 2] - minimum[2]) / z_span
    strength = np.clip(
        (lateral - width_start_fraction)
        / (width_full_fraction - width_start_fraction),
        0.0,
        1.0,
    ).astype(np.float32)
    spatial = (
        (vertical >= vertical_bottom_fraction)
        & (vertical <= vertical_top_fraction)
        & (strength > 0.0)
    )

    unknown = int(SemanticClass.UNKNOWN)
    skin = int(SemanticClass.SKIN_LIKE)
    cool = int(SemanticClass.COOL_SATURATED)
    if expected_labels is None:
        expected = np.full(len(points), unknown, dtype=np.int16)
    else:
        expected = np.asarray(expected_labels)
        if expected.shape != (len(points),):
            raise ValueError("expected_labels must have shape (N,)")

    strength[~spatial] = 0.0
    heuristic_wrist = (
        (vertical >= wrist_bottom_fraction)
        & (vertical <= wrist_top_fraction)
    )
    wrist = (strength > 0.0) & (
        (expected == cool)
        | ((expected != skin) & heuristic_wrist)
    )
    skin_mask = (strength > 0.0) & ~wrist

    palette_colors = np.zeros((len(points), 3), dtype=np.float32)
    palette_colors[skin_mask] = skin_palette
    palette_colors[wrist] = cool_palette
    stats = ArmRepairPlanStats(
        spatial_candidates=int(np.count_nonzero(spatial)),
        repair_candidates=int(np.count_nonzero(strength > 0.0)),
        protected_surface_vertices=0,
        skin_vertices=int(np.count_nonzero(skin_mask)),
        wrist_vertices=int(np.count_nonzero(wrist)),
        skin_palette=tuple(float(value) for value in skin_palette),
        cool_palette=tuple(float(value) for value in cool_palette),
    )
    return palette_colors, strength, stats


def apply_arm_repair_plan_to_corners(
    vertices: np.ndarray,
    loop_vertices: np.ndarray,
    polygon_for_loop: np.ndarray,
    vertex_visibility: np.ndarray,
    polygon_visibility: np.ndarray,
    fallback_colors: np.ndarray,
    fallback_mix: np.ndarray,
    plan_colors: np.ndarray,
    plan_strength: np.ndarray,
    *,
    left_channel: int = 1,
    right_channel: int = 3,
) -> tuple[np.ndarray, np.ndarray, ArmCornerRepairStats]:
    """Apply an arm plan only to outward-visible vertex/polygon corners."""

    points = np.asarray(vertices)
    loops = np.asarray(loop_vertices, dtype=np.int64)
    polygons = np.asarray(polygon_for_loop, dtype=np.int64)
    vertex_visible = np.asarray(vertex_visibility, dtype=bool)
    polygon_visible = np.asarray(polygon_visibility, dtype=bool)
    colors = np.asarray(fallback_colors, dtype=np.float32).copy()
    mix = np.asarray(fallback_mix, dtype=np.float32).copy()
    palette = np.asarray(plan_colors, dtype=np.float32)
    strength = np.asarray(plan_strength, dtype=np.float32)

    corner_count = len(loops)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if polygons.shape != (corner_count,):
        raise ValueError("polygon_for_loop must match loop_vertices")
    if colors.shape != (corner_count, 3):
        raise ValueError("fallback_colors must have shape (corners, 3)")
    if mix.shape != (corner_count,):
        raise ValueError("fallback_mix must have shape (corners,)")
    if palette.shape != (len(points), 3):
        raise ValueError("plan_colors must have shape (vertices, 3)")
    if strength.shape != (len(points),):
        raise ValueError("plan_strength must have shape (vertices,)")
    if vertex_visible.ndim != 2 or vertex_visible.shape[0] != len(points):
        raise ValueError("vertex_visibility must have shape (vertices, views)")
    if (
        polygon_visible.ndim != 2
        or polygon_visible.shape[0] <= int(polygons.max(initial=-1))
        or polygon_visible.shape[1] != vertex_visible.shape[1]
    ):
        raise ValueError(
            "polygon_visibility must have shape (polygons, views)"
        )
    if not (
        0 <= left_channel < vertex_visible.shape[1]
        and 0 <= right_channel < vertex_visible.shape[1]
    ):
        raise ValueError("side-view channels are out of range")

    x_mid = 0.5 * (
        float(points[:, 0].min()) + float(points[:, 0].max())
    )
    loop_points = points[loops]
    desired = np.where(
        loop_points[:, 0] >= x_mid,
        left_channel,
        right_channel,
    )
    side_supported = (
        vertex_visible[loops, desired]
        & polygon_visible[polygons, desired]
    )
    corner_strength = strength[loops]
    candidate = corner_strength > 0.0
    applied = candidate & side_supported
    applied_strength = np.where(applied, corner_strength, 0.0)
    colors[applied] = palette[loops[applied]]
    mix = np.maximum(mix, applied_strength)

    stats = ArmCornerRepairStats(
        candidate_corners=int(np.count_nonzero(candidate)),
        side_supported_corners=int(
            np.count_nonzero(candidate & side_supported)
        ),
        applied_corners=int(np.count_nonzero(applied)),
        full_strength_corners=int(
            np.count_nonzero(applied & (corner_strength >= 1.0))
        ),
        hidden_rejected_corners=int(
            np.count_nonzero(candidate & ~side_supported)
        ),
        applied_outside_visibility=int(
            np.count_nonzero(applied & ~side_supported)
        ),
    )
    return colors, mix, stats
