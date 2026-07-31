"""Camera frames and robust color consensus for ten-view texture projection.

The production four-view baker stores one weight per cardinal camera in one
RGBA attribute.  Ten cameras no longer fit that representation.  This module
keeps the camera/consensus math independent of Blender so the experimental
ten-view worker can calculate the final per-corner color first and bake that
single color attribute to a normal UV texture.

The consensus pass is deliberately conservative: if several visible cameras
agree and one camera samples a very different color (for example, skin over a
garment), the isolated sample is strongly down-weighted.  A surface visible
from only one camera is retained instead of being discarded.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping

import numpy as np

from .calibration import ViewFrame
from .visibility import smoothstep01


TEN_VIEW_ANGLES: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "front_right": (45.0, 0.0),
    "right": (90.0, 0.0),
    "back_right": (135.0, 0.0),
    "back": (180.0, 0.0),
    "back_left": (225.0, 0.0),
    "left": (270.0, 0.0),
    "front_left": (315.0, 0.0),
    "high_front": (0.0, 30.0),
    "high_back": (180.0, 30.0),
}


def add_regional_ownership_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the ten-view regional projection policy CLI switches.

    Regional ownership is deliberately a native ten-view default: the two
    elevated cameras are useful for the head and shoulders, but should not
    project down the body, and an opposite-side camera should not dominate an
    outer arm. The legacy explicit enable flag remains accepted while the
    disable flag provides a deterministic escape hatch for comparisons.
    """

    ownership = parser.add_mutually_exclusive_group()
    ownership.add_argument(
        "--enable-regional-ownership",
        dest="enable_regional_ownership",
        action="store_true",
        help="enable conservative ten-view camera-region ownership (default)",
    )
    ownership.add_argument(
        "--disable-regional-ownership",
        dest="enable_regional_ownership",
        action="store_false",
        help="disable ten-view camera-region ownership for comparison",
    )
    parser.set_defaults(enable_regional_ownership=True)


HORIZONTAL_VIEW_NAMES = tuple(TEN_VIEW_ANGLES)[:8]
HIGH_VIEW_NAMES = tuple(TEN_VIEW_ANGLES)[8:]


def frame_from_angles(
    name: str,
    yaw_degrees: float,
    elevation_degrees: float,
) -> ViewFrame:
    """Return the Blender Z-up orthographic frame used by the Web UI."""

    yaw = math.radians(float(yaw_degrees))
    elevation = math.radians(float(elevation_degrees))
    horizontal = math.cos(elevation)
    to_camera = np.asarray(
        (
            -math.sin(yaw) * horizontal,
            -math.cos(yaw) * horizontal,
            math.sin(elevation),
        ),
        dtype=np.float64,
    )
    right = np.asarray(
        (math.cos(yaw), -math.sin(yaw), 0.0),
        dtype=np.float64,
    )
    up = np.cross(to_camera, right)
    return ViewFrame(
        name=name,
        right=tuple(right),
        up=tuple(up),
        to_camera=tuple(to_camera),
    )


def ten_view_frames(
    angles: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, ViewFrame]:
    """Build frames in deterministic UI order."""

    selected = angles or TEN_VIEW_ANGLES
    return {
        name: frame_from_angles(name, *selected[name])
        for name in selected
    }


def linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    """Convert scene-linear RGB to sRGB for perceptual consensus distance."""

    values = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    """Convert sRGB values to scene-linear RGB."""

    values = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )


def restore_chroma(
    colors: np.ndarray,
    *,
    amount: float = 0.12,
    neutral_threshold: float = 0.055,
) -> np.ndarray:
    """Undo a small amount of multi-view averaging desaturation.

    Near-neutral pixels are left untouched.  Chromatic pixels are moved away
    from their sRGB luminance by at most ``amount``; this preserves source hue
    and is intentionally much gentler than a global saturation filter.
    """

    if not 0.0 <= amount <= 1.0 or not 0.0 <= neutral_threshold < 1.0:
        raise ValueError("chroma restoration parameters are invalid")
    srgb = linear_to_srgb(colors)
    chroma = srgb.max(axis=-1) - srgb.min(axis=-1)
    strength = smoothstep01(
        (chroma - neutral_threshold) / max(0.30 - neutral_threshold, 1.0e-12)
    )
    luminance = np.sum(
        srgb * np.asarray((0.2126, 0.7152, 0.0722)),
        axis=-1,
        keepdims=True,
    )
    restored = luminance + (srgb - luminance) * (1.0 + amount * strength[..., None])
    return srgb_to_linear(np.clip(restored, 0.0, 1.0)).astype(np.float32)


def robust_consensus_weights(
    colors: np.ndarray,
    weights: np.ndarray,
    *,
    color_sigma: float = 0.28,
    minimum_peer_support: float = 0.20,
    floor: float = 0.035,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Suppress isolated cross-view colors without inventing surface labels.

    ``colors`` has shape ``(samples, views, 3)`` and contains scene-linear
    values.  ``weights`` has shape ``(samples, views)`` and already includes
    angle, alpha-interior and ray-visibility confidence.

    Pairwise agreement is measured in sRGB.  A view's own weight is excluded
    from its support score, so a high-confidence but isolated skin sample
    cannot vote itself onto orange cloth.  Rows with only one usable view are
    left unchanged.
    """

    color_values = np.asarray(colors, dtype=np.float64)
    raw = np.asarray(weights, dtype=np.float64)
    if color_values.ndim != 3 or color_values.shape[2] != 3:
        raise ValueError("colors must have shape (samples, views, 3)")
    if raw.shape != color_values.shape[:2]:
        raise ValueError("weights must match the first two color dimensions")
    if (
        color_sigma <= 0.0
        or not 0.0 <= minimum_peer_support < 1.0
        or not 0.0 <= floor <= 1.0
    ):
        raise ValueError("consensus parameters are outside their valid range")
    if not np.all(np.isfinite(color_values)) or not np.all(np.isfinite(raw)):
        raise ValueError("colors and weights must be finite")

    raw = np.maximum(raw, 0.0)
    usable = raw > 1.0e-12
    usable_count = usable.sum(axis=1)
    srgb = linear_to_srgb(color_values)
    adjusted = raw.copy()
    total = raw.sum(axis=1)
    denominator_scale = 2.0 * color_sigma * color_sigma

    for view_index in range(raw.shape[1]):
        difference = srgb - srgb[:, view_index : view_index + 1, :]
        similarity = np.exp(
            -np.sum(difference * difference, axis=2) / denominator_scale
        )
        peers = raw.copy()
        peers[:, view_index] = 0.0
        peer_total = total - raw[:, view_index]
        peer_support = np.divide(
            np.sum(peers * similarity, axis=1),
            peer_total,
            out=np.ones_like(peer_total),
            where=peer_total > 1.0e-12,
        )
        ramp = (
            (peer_support - minimum_peer_support)
            / max(1.0 - minimum_peer_support, 1.0e-12)
        )
        agreement = floor + (1.0 - floor) * smoothstep01(ramp)
        agreement[usable_count <= 1] = 1.0
        adjusted[:, view_index] *= agreement

    raw_sum = raw.sum(axis=1, keepdims=True)
    adjusted_sum = adjusted.sum(axis=1, keepdims=True)
    normalized = np.divide(
        adjusted,
        adjusted_sum,
        out=np.zeros_like(adjusted),
        where=adjusted_sum > 1.0e-12,
    )
    normalized_raw = np.divide(
        raw,
        raw_sum,
        out=np.zeros_like(raw),
        where=raw_sum > 1.0e-12,
    )
    suppression = normalized_raw - normalized
    report: dict[str, float | int] = {
        "samples": int(len(raw)),
        "views": int(raw.shape[1]),
        "single_view_samples": int(np.count_nonzero(usable_count == 1)),
        "multi_view_samples": int(np.count_nonzero(usable_count > 1)),
        "unresolved_samples": int(np.count_nonzero(usable_count == 0)),
        "strongly_suppressed_entries": int(
            np.count_nonzero((raw > 1.0e-12) & (suppression > 0.20))
        ),
        "mean_usable_views": float(usable_count.mean()) if len(raw) else 0.0,
        "color_sigma": float(color_sigma),
        "minimum_peer_support": float(minimum_peer_support),
        "floor": float(floor),
    }
    return normalized.astype(np.float32), report


def blend_consensus_colors(
    colors: np.ndarray,
    normalized_weights: np.ndarray,
    *,
    winner_mix: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend colors, optionally retaining part of the best source sample."""

    color_values = np.asarray(colors, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if color_values.ndim != 3 or color_values.shape[2] != 3:
        raise ValueError("colors must have shape (samples, views, 3)")
    if weights.shape != color_values.shape[:2]:
        raise ValueError("weights must match the first two color dimensions")
    if not 0.0 <= winner_mix <= 1.0:
        raise ValueError("winner_mix must be between zero and one")
    valid = weights.sum(axis=1) > 1.0e-12
    blended = np.sum(color_values * weights[..., None], axis=1)
    if winner_mix > 0.0 and len(blended):
        winner_index = np.argmax(weights, axis=1)
        winner = color_values[np.arange(len(blended)), winner_index]
        blended[valid] = (
            (1.0 - winner_mix) * blended[valid] + winner_mix * winner[valid]
        )
    blended[~valid] = 0.0
    return blended.astype(np.float32), valid


__all__ = [
    "HIGH_VIEW_NAMES",
    "HORIZONTAL_VIEW_NAMES",
    "TEN_VIEW_ANGLES",
    "add_regional_ownership_arguments",
    "blend_consensus_colors",
    "frame_from_angles",
    "linear_to_srgb",
    "restore_chroma",
    "robust_consensus_weights",
    "srgb_to_linear",
    "ten_view_frames",
]
