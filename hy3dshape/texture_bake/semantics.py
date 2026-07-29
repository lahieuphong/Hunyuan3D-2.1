"""Conservative semantic color guards for stylized character references.

These guards do not attempt full semantic segmentation.  They classify broad
color families (skin-like, saturated warm cloth, cool cloth, neutral, dark)
and down-weight a view whose sampled family conflicts with the cross-view
consensus or an explicitly supplied per-surface expectation.

Thresholds and the compatibility matrix are data, not hard-coded bake policy,
so a character-specific palette can replace the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SemanticClass(IntEnum):
    UNKNOWN = 0
    DARK_OR_INK = 1
    SKIN_LIKE = 2
    WARM_SATURATED = 3
    COOL_SATURATED = 4
    LIGHT_NEUTRAL = 5
    MID_NEUTRAL = 6


@dataclass(frozen=True)
class AnimePaletteThresholds:
    """HSV thresholds for broad, configurable anime color families."""

    dark_value_max: float = 0.25
    neutral_saturation_max: float = 0.18
    light_neutral_value_min: float = 0.78
    skin_hue_min: float = 0.0
    skin_hue_max: float = 0.16
    skin_saturation_min: float = 0.10
    skin_saturation_max: float = 0.58
    skin_value_min: float = 0.45
    warm_hue_min: float = 0.015
    warm_hue_max: float = 0.17
    warm_saturation_min: float = 0.58
    warm_value_min: float = 0.28
    cool_hue_min: float = 0.42
    cool_hue_max: float = 0.80
    cool_saturation_min: float = 0.28
    cool_value_min: float = 0.20


def rgb_to_unit(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB/RGBA values to float64 RGB in ``[0, 1]``."""

    array = np.asarray(rgb)
    if array.ndim < 1 or array.shape[-1] not in (3, 4):
        raise ValueError("rgb must have a final dimension of three or four")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.bool_):
        result = array.astype(np.float64)
    elif np.issubdtype(array.dtype, np.integer):
        result = array.astype(np.float64) / float(np.iinfo(array.dtype).max)
    else:
        result = array.astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("rgb must contain only finite values")
        if float(result.min()) < -1e-6:
            raise ValueError("rgb values must not be negative")
        maximum = float(result.max()) if result.size else 0.0
        if maximum > 1.0 + 1e-6:
            if maximum <= 255.0 + 1e-6:
                result /= 255.0
            else:
                raise ValueError("floating rgb values must use a 0..1 or 0..255 range")
    return np.clip(result, 0.0, 1.0)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB-to-HSV conversion with hue normalized to ``[0, 1)``."""

    unit = rgb_to_unit(rgb)
    red, green, blue = np.moveaxis(unit, -1, 0)
    maximum = unit.max(axis=-1)
    minimum = unit.min(axis=-1)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    chromatic = delta > 1e-12
    red_max = chromatic & (maximum == red)
    green_max = chromatic & ~red_max & (maximum == green)
    blue_max = chromatic & ~red_max & ~green_max
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue /= 6.0

    saturation = np.divide(
        delta,
        maximum,
        out=np.zeros_like(delta),
        where=maximum > 1e-12,
    )
    return np.stack((hue, saturation, maximum), axis=-1)


def classify_anime_colors(
    rgb: np.ndarray,
    *,
    thresholds: AnimePaletteThresholds | None = None,
) -> np.ndarray:
    """Classify broad color families while leaving ambiguous colors unknown."""

    settings = thresholds or AnimePaletteThresholds()
    hsv = rgb_to_hsv(rgb)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    labels = np.full(hue.shape, int(SemanticClass.UNKNOWN), dtype=np.int8)

    dark = value <= settings.dark_value_max
    labels[dark] = int(SemanticClass.DARK_OR_INK)

    unclassified = labels == int(SemanticClass.UNKNOWN)
    light_neutral = (
        unclassified
        & (saturation <= settings.neutral_saturation_max)
        & (value >= settings.light_neutral_value_min)
    )
    labels[light_neutral] = int(SemanticClass.LIGHT_NEUTRAL)

    unclassified = labels == int(SemanticClass.UNKNOWN)
    warm = (
        unclassified
        & (hue >= settings.warm_hue_min)
        & (hue <= settings.warm_hue_max)
        & (saturation >= settings.warm_saturation_min)
        & (value >= settings.warm_value_min)
    )
    labels[warm] = int(SemanticClass.WARM_SATURATED)

    unclassified = labels == int(SemanticClass.UNKNOWN)
    skin = (
        unclassified
        & (hue >= settings.skin_hue_min)
        & (hue <= settings.skin_hue_max)
        & (saturation >= settings.skin_saturation_min)
        & (saturation < settings.skin_saturation_max)
        & (value >= settings.skin_value_min)
    )
    labels[skin] = int(SemanticClass.SKIN_LIKE)

    unclassified = labels == int(SemanticClass.UNKNOWN)
    cool = (
        unclassified
        & (hue >= settings.cool_hue_min)
        & (hue <= settings.cool_hue_max)
        & (saturation >= settings.cool_saturation_min)
        & (value >= settings.cool_value_min)
    )
    labels[cool] = int(SemanticClass.COOL_SATURATED)

    unclassified = labels == int(SemanticClass.UNKNOWN)
    mid_neutral = unclassified & (saturation <= settings.neutral_saturation_max)
    labels[mid_neutral] = int(SemanticClass.MID_NEUTRAL)
    return labels


def default_compatibility_matrix() -> np.ndarray:
    """Return conservative compatibility between sampled and expected classes."""

    count = max(int(value) for value in SemanticClass) + 1
    matrix = np.full((count, count), 0.20, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)

    unknown = int(SemanticClass.UNKNOWN)
    dark = int(SemanticClass.DARK_OR_INK)
    skin = int(SemanticClass.SKIN_LIKE)
    warm = int(SemanticClass.WARM_SATURATED)
    cool = int(SemanticClass.COOL_SATURATED)
    light = int(SemanticClass.LIGHT_NEUTRAL)
    neutral = int(SemanticClass.MID_NEUTRAL)

    # Unknown samples are usable but should not dominate a known consensus.
    matrix[unknown, :] = 0.65
    matrix[:, unknown] = 0.65
    matrix[unknown, unknown] = 1.0

    # Dark pixels can be intentional line art on every region.
    matrix[dark, :] = np.maximum(matrix[dark, :], 0.55)
    matrix[:, dark] = np.maximum(matrix[:, dark], 0.55)
    matrix[dark, dark] = 1.0

    # Skin and orange clothing are near in hue but should still be guarded.
    matrix[skin, warm] = matrix[warm, skin] = 0.25
    matrix[skin, cool] = matrix[cool, skin] = 0.08
    matrix[warm, cool] = matrix[cool, warm] = 0.05
    matrix[light, neutral] = matrix[neutral, light] = 0.80
    matrix[skin, light] = matrix[light, skin] = 0.35
    matrix[warm, light] = matrix[light, warm] = 0.25
    matrix[cool, light] = matrix[light, cool] = 0.25
    matrix[skin, neutral] = matrix[neutral, skin] = 0.35
    matrix[warm, neutral] = matrix[neutral, warm] = 0.35
    matrix[cool, neutral] = matrix[neutral, cool] = 0.35
    return matrix


@dataclass(frozen=True)
class SemanticGuardResult:
    labels: np.ndarray
    expected_labels: np.ndarray
    consensus_strength: np.ndarray
    guard: np.ndarray
    adjusted_weights: np.ndarray


def apply_semantic_color_guards(
    colors: np.ndarray,
    base_weights: np.ndarray,
    *,
    expected_labels: np.ndarray | None = None,
    thresholds: AnimePaletteThresholds | None = None,
    compatibility: np.ndarray | None = None,
    minimum_guard: float = 0.05,
) -> SemanticGuardResult:
    """Down-weight cross-view semantic color outliers.

    Args:
        colors: RGB samples with shape ``(views, samples, 3/4)``.
        base_weights: Existing geometric confidence, shape ``(views, samples)``.
        expected_labels: Optional trusted class per surface sample.  When
            omitted, a base-confidence-weighted cross-view vote is used.
        minimum_guard: Lower bound retained for a conflicting view so line
            details are not destroyed by a brittle classifier.
    """

    color_array = np.asarray(colors)
    weights = np.asarray(base_weights, dtype=np.float64)
    if color_array.ndim != 3 or color_array.shape[2] not in (3, 4):
        raise ValueError("colors must have shape (views, samples, 3/4)")
    if weights.shape != color_array.shape[:2]:
        raise ValueError("base_weights must have shape (views, samples)")
    if not 0.0 <= minimum_guard <= 1.0:
        raise ValueError("minimum_guard must be between 0 and 1")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    labels = classify_anime_colors(color_array, thresholds=thresholds)

    class_count = max(int(value) for value in SemanticClass) + 1
    matrix = (
        default_compatibility_matrix()
        if compatibility is None
        else np.asarray(compatibility, dtype=np.float64)
    )
    if matrix.shape != (class_count, class_count):
        raise ValueError(f"compatibility must have shape {(class_count, class_count)}")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("compatibility values must be finite and between 0 and 1")

    sample_count = color_array.shape[1]
    if expected_labels is None:
        votes = np.zeros((class_count, sample_count), dtype=np.float64)
        for class_id in range(1, class_count):
            votes[class_id] = np.sum(weights * (labels == class_id), axis=0)
        known_total = votes[1:].sum(axis=0)
        expected = np.argmax(votes[1:], axis=0).astype(np.int16) + 1
        expected[known_total <= 1e-12] = int(SemanticClass.UNKNOWN)
        winning_vote = votes[expected, np.arange(sample_count)]
        strength = np.divide(
            winning_vote,
            known_total,
            out=np.zeros_like(known_total),
            where=known_total > 1e-12,
        )
    else:
        expected = np.asarray(expected_labels, dtype=np.int16)
        if expected.shape != (sample_count,):
            raise ValueError("expected_labels must have shape (samples,)")
        if np.any((expected < 0) | (expected >= class_count)):
            raise ValueError("expected_labels contains an unknown class id")
        strength = (expected != int(SemanticClass.UNKNOWN)).astype(np.float64)

    compatibility_for_samples = matrix[labels, expected[None, :]]
    guard = 1.0 - strength[None, :] * (1.0 - compatibility_for_samples)
    guard[:, expected == int(SemanticClass.UNKNOWN)] = 1.0
    guard = np.clip(guard, minimum_guard, 1.0)
    return SemanticGuardResult(
        labels=labels,
        expected_labels=expected,
        consensus_strength=strength,
        guard=guard,
        adjusted_weights=weights * guard,
    )
