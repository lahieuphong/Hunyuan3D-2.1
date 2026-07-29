"""Conservative flat-palette repair for projection-prone character regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantics import SemanticClass, classify_anime_colors


@dataclass(frozen=True)
class ArmPaletteRepairStats:
    """Summary of a feathered outer-arm palette repair."""

    repaired_vertices: int
    full_strength_vertices: int
    skin_vertices: int
    wrist_vertices: int
    skin_palette: tuple[float, float, float]
    cool_palette: tuple[float, float, float]


def apply_arm_palette_repair(
    vertices: np.ndarray,
    sampled_colors: np.ndarray,
    fallback_colors: np.ndarray,
    fallback_mix: np.ndarray,
    *,
    width_start_fraction: float = 0.36,
    width_full_fraction: float = 0.43,
    vertical_bottom_fraction: float = 0.44,
    wrist_bottom_fraction: float = 0.535,
    wrist_top_fraction: float = 0.585,
    vertical_top_fraction: float = 0.72,
    cool_value_scale: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, ArmPaletteRepairStats]:
    """Feather clean skin/blue palette colors onto outer arm surfaces.

    The repair deliberately affects only the outermost lateral band, where
    planar multi-view projection stretches silhouette outlines most severely.
    Existing projected detail remains untouched towards the inner arm.
    """

    points = np.asarray(vertices, dtype=np.float64)
    samples = np.asarray(sampled_colors)
    colors = np.asarray(fallback_colors, dtype=np.float32).copy()
    mix = np.asarray(fallback_mix, dtype=np.float32).copy()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if samples.ndim != 3 or samples.shape[1:] != (len(points), 3):
        raise ValueError("sampled_colors must have shape (views, N, 3)")
    if colors.shape != (len(points), 3):
        raise ValueError("fallback_colors must have shape (N, 3)")
    if mix.shape != (len(points),):
        raise ValueError("fallback_mix must have shape (N,)")
    if not 0 <= width_start_fraction < width_full_fraction <= 0.5:
        raise ValueError("width fractions must satisfy 0 <= start < full <= 0.5")
    if not (
        0
        <= vertical_bottom_fraction
        < wrist_bottom_fraction
        < wrist_top_fraction
        < vertical_top_fraction
        <= 1
    ):
        raise ValueError("vertical arm fractions must be strictly ordered")
    if not 0.0 < cool_value_scale <= 1.0:
        raise ValueError("cool_value_scale must be between 0 and 1")

    labels = classify_anime_colors(samples)
    skin_samples = samples[labels == int(SemanticClass.SKIN_LIKE)]
    cool_samples = samples[labels == int(SemanticClass.COOL_SATURATED)]
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
    lateral_fraction = np.abs(points[:, 0] - x_mid) / x_span
    vertical_fraction = (points[:, 2] - minimum[2]) / z_span
    strength = np.clip(
        (lateral_fraction - width_start_fraction)
        / (width_full_fraction - width_start_fraction),
        0.0,
        1.0,
    ).astype(np.float32)

    in_arm_height = (
        (vertical_fraction >= vertical_bottom_fraction)
        & (vertical_fraction <= vertical_top_fraction)
    )
    wrist = (
        in_arm_height
        & (vertical_fraction >= wrist_bottom_fraction)
        & (vertical_fraction <= wrist_top_fraction)
        & (strength > 0)
    )
    skin = in_arm_height & ~wrist & (strength > 0)
    repaired = skin | wrist
    colors[skin] = skin_palette
    colors[wrist] = cool_palette
    mix[repaired] = np.maximum(mix[repaired], strength[repaired])

    stats = ArmPaletteRepairStats(
        repaired_vertices=int(np.count_nonzero(repaired)),
        full_strength_vertices=int(
            np.count_nonzero(repaired & (strength >= 1.0))
        ),
        skin_vertices=int(np.count_nonzero(skin)),
        wrist_vertices=int(np.count_nonzero(wrist)),
        skin_palette=tuple(float(value) for value in skin_palette),
        cool_palette=tuple(float(value) for value in cool_palette),
    )
    return colors, mix, stats
