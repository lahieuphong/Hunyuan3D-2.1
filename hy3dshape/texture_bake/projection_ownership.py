"""Localized ownership guard for side-view arm pixels on warm garments.

Ray visibility answers whether a reconstructed surface is visible, but it
cannot tell whether the corresponding source-image pixel depicts the same
body part when the input pose and reconstruction are slightly misregistered.
This guard detects the characteristic case where Front/Back sees orange
garment while the stronger side view samples skin, a white hand highlight, or
a blue wristband at the same garment vertex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantics import SemanticClass, classify_anime_colors


@dataclass(frozen=True)
class SideGarmentOwnershipStats:
    """Summary of side-view samples rejected in favor of garment anchors."""

    core_vertices: int
    expanded_vertices: int
    repaired_vertices: int
    rejected_entries: int
    positive_side_vertices: int
    negative_side_vertices: int
    expanded_per_hop: tuple[int, ...]


def apply_side_garment_ownership_guard(
    vertices: np.ndarray,
    sampled_colors: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    *,
    vertical_min_fraction: float = 0.44,
    vertical_max_fraction: float = 0.585,
    lateral_min_fraction: float = 0.10,
    lateral_max_fraction: float = 0.32,
    expansion_vertical_min_fraction: float = 0.465,
    expansion_lateral_max_fraction: float = 0.28,
    minimum_anchor_weight: float = 0.03,
    minimum_side_weight: float = 0.03,
    minimum_side_ratio: float = 1.5,
    expansion_hops: int = 3,
    front_channel: int = 0,
    left_channel: int = 1,
    back_channel: int = 2,
    right_channel: int = 3,
) -> tuple[np.ndarray, np.ndarray, SideGarmentOwnershipStats]:
    """Remove misregistered side-arm samples from a localized garment band.

    A core vertex must have a warm Front/Back anchor and an arm-family side
    sample whose weight dominates that anchor. Three optional graph rings
    absorb outline pixels around the core, but only where the anchor remains
    warm and within tighter geometric bounds. Only the offending side channel
    is zeroed; dark garment folds and the anchor color are preserved.
    """

    points = np.asarray(vertices, dtype=np.float64)
    colors = np.asarray(sampled_colors)
    weight_array = np.asarray(weights, dtype=np.float32)
    edge_array = np.asarray(edges, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if colors.ndim != 3 or colors.shape[1:] != (len(points), 3):
        raise ValueError("sampled_colors must have shape (views, N, 3)")
    if weight_array.shape != (len(points), colors.shape[0]):
        raise ValueError("weights must have shape (N, views)")
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (M, 2)")
    if edge_array.size and (
        int(edge_array.min()) < 0 or int(edge_array.max()) >= len(points)
    ):
        raise ValueError("edges contain an out-of-range vertex")
    channels = (
        front_channel,
        left_channel,
        back_channel,
        right_channel,
    )
    if any(channel < 0 or channel >= colors.shape[0] for channel in channels):
        raise ValueError("view channel is out of range")
    if not (
        0.0 <= vertical_min_fraction < vertical_max_fraction <= 1.0
        and vertical_min_fraction
        <= expansion_vertical_min_fraction
        < vertical_max_fraction
    ):
        raise ValueError("vertical fractions are invalid")
    if not (
        0.0 <= lateral_min_fraction
        < expansion_lateral_max_fraction
        <= lateral_max_fraction
        <= 0.5
    ):
        raise ValueError("lateral fractions are invalid")
    if minimum_anchor_weight < 0.0 or minimum_side_weight < 0.0:
        raise ValueError("minimum weights must not be negative")
    if minimum_side_ratio <= 0.0:
        raise ValueError("minimum_side_ratio must be positive")
    if expansion_hops < 0:
        raise ValueError("expansion_hops must not be negative")
    if not len(points):
        empty = np.zeros(0, dtype=bool)
        stats = SideGarmentOwnershipStats(0, 0, 0, 0, 0, 0, ())
        return weight_array.copy(), empty, stats

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    x_span = max(maximum[0] - minimum[0], 1.0e-8)
    z_span = max(maximum[2] - minimum[2], 1.0e-8)
    x_mid = 0.5 * (minimum[0] + maximum[0])
    y_mid = 0.5 * (minimum[1] + maximum[1])
    lateral = np.abs(points[:, 0] - x_mid) / x_span
    vertical = (points[:, 2] - minimum[2]) / z_span
    positive = points[:, 0] >= x_mid
    anchor_channel = np.where(
        points[:, 1] <= y_mid,
        front_channel,
        back_channel,
    )
    side_channel = np.where(positive, left_channel, right_channel)
    indices = np.arange(len(points))

    labels = classify_anime_colors(colors)
    anchor_label = labels[anchor_channel, indices]
    side_label = labels[side_channel, indices]
    anchor_weight = weight_array[indices, anchor_channel]
    side_weight = weight_array[indices, side_channel]
    warm = int(SemanticClass.WARM_SATURATED)
    side_arm_family = np.isin(
        side_label,
        (
            int(SemanticClass.SKIN_LIKE),
            int(SemanticClass.LIGHT_NEUTRAL),
            int(SemanticClass.COOL_SATURATED),
        ),
    )
    core_region = (
        (vertical >= vertical_min_fraction)
        & (vertical <= vertical_max_fraction)
        & (lateral >= lateral_min_fraction)
        & (lateral <= lateral_max_fraction)
    )
    core = (
        core_region
        & (anchor_label == warm)
        & side_arm_family
        & (anchor_weight >= minimum_anchor_weight)
        & (side_weight >= minimum_side_weight)
        & (side_weight >= minimum_side_ratio * anchor_weight)
    )

    repaired = core.copy()
    frontier = core.copy()
    expanded_per_hop: list[int] = []
    expansion_region = (
        (vertical >= expansion_vertical_min_fraction)
        & (vertical <= vertical_max_fraction)
        & (lateral >= lateral_min_fraction)
        & (lateral <= expansion_lateral_max_fraction)
        & (anchor_label == warm)
        & (anchor_weight >= minimum_anchor_weight)
        & (side_weight >= minimum_side_weight)
    )
    left, right = (
        (edge_array[:, 0], edge_array[:, 1])
        if len(edge_array)
        else (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    )
    for _ in range(expansion_hops):
        if not np.any(frontier) or not len(edge_array):
            break
        next_ring = np.zeros(len(points), dtype=bool)
        left_to_right = frontier[left] & ~repaired[right]
        right_to_left = frontier[right] & ~repaired[left]
        next_ring[right[left_to_right]] = True
        next_ring[left[right_to_left]] = True
        next_ring &= expansion_region & ~repaired
        count = int(np.count_nonzero(next_ring))
        if not count:
            break
        repaired |= next_ring
        frontier = next_ring
        expanded_per_hop.append(count)

    adjusted = weight_array.copy()
    repaired_indices = np.flatnonzero(repaired)
    adjusted[repaired_indices, side_channel[repaired_indices]] = 0.0
    stats = SideGarmentOwnershipStats(
        core_vertices=int(np.count_nonzero(core)),
        expanded_vertices=int(np.count_nonzero(repaired & ~core)),
        repaired_vertices=int(np.count_nonzero(repaired)),
        rejected_entries=int(
            np.count_nonzero(
                weight_array[repaired_indices, side_channel[repaired_indices]]
                > 1.0e-10
            )
        ),
        positive_side_vertices=int(np.count_nonzero(repaired & positive)),
        negative_side_vertices=int(np.count_nonzero(repaired & ~positive)),
        expanded_per_hop=tuple(expanded_per_hop),
    )
    return adjusted, repaired, stats
