"""Adaptive face/hair ownership guards for stylized cyan-haired characters.

The texture baker sees colors, visibility and geometry, but it does not know
that a cyan sample belongs to hair while a peach sample belongs to a face.
This module supplies that deliberately small missing semantic layer without a
learned model.  It is designed for both image pixels and per-surface samples:
all leading dimensions are preserved and only the last RGB/RGBA dimension is
special.

Two details are intentionally conservative:

* near-black pixels are always treated as line art and are never repainted;
* near-white highlights remain neutral unless a caller supplies a geometric
  face/hair prior, and are never repainted by the guard.

The palette is estimated from the actual references, so cyan, turquoise and
blue-cyan hair variants do not need character-specific constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .semantics import rgb_to_hsv, rgb_to_unit


class HeadSemanticClass(IntEnum):
    """Small semantic vocabulary used by the head ownership pass."""

    INVALID = 0
    INK = 1
    FACE = 2
    HAIR = 3
    HIGHLIGHT = 4
    UNKNOWN = 5


@dataclass(frozen=True)
class HeadOwnershipConfig:
    """Thresholds for adaptive anime face/hair classification."""

    ink_value_max: float = 0.24
    highlight_value_min: float = 0.78
    highlight_saturation_max: float = 0.20
    skin_hue_min: float = 0.0
    skin_hue_max: float = 0.17
    skin_saturation_min: float = 0.07
    skin_saturation_max: float = 0.64
    skin_value_min: float = 0.34
    cool_hue_min: float = 0.40
    cool_hue_max: float = 0.68
    cool_saturation_min: float = 0.22
    cool_value_min: float = 0.18
    skin_hue_sigma: float = 0.075
    skin_saturation_sigma: float = 0.22
    skin_value_sigma: float = 0.38
    hair_hue_sigma: float = 0.085
    hair_saturation_sigma: float = 0.30
    hair_value_sigma: float = 0.42
    minimum_color_confidence: float = 0.16
    prior_strength: float = 0.76
    warm_contamination_saturation_min: float = 0.22
    guard_minimum_contamination: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.ink_value_max < self.highlight_value_min <= 1.0:
            raise ValueError("ink/highlight value thresholds are invalid")
        if not 0.0 <= self.highlight_saturation_max <= 1.0:
            raise ValueError("highlight_saturation_max must be between 0 and 1")
        if not 0.0 <= self.skin_hue_min < self.skin_hue_max <= 1.0:
            raise ValueError("skin hue thresholds are invalid")
        if not 0.0 <= self.cool_hue_min < self.cool_hue_max <= 1.0:
            raise ValueError("cool hue thresholds are invalid")
        if not (
            0.0
            <= self.skin_saturation_min
            < self.skin_saturation_max
            <= 1.0
        ):
            raise ValueError("skin saturation thresholds are invalid")
        positive = (
            self.skin_hue_sigma,
            self.skin_saturation_sigma,
            self.skin_value_sigma,
            self.hair_hue_sigma,
            self.hair_saturation_sigma,
            self.hair_value_sigma,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("palette sigmas must be positive")
        bounded = (
            self.minimum_color_confidence,
            self.prior_strength,
            self.warm_contamination_saturation_min,
            self.guard_minimum_contamination,
        )
        if any(value < 0.0 or value > 1.0 for value in bounded):
            raise ValueError("confidence parameters must be between 0 and 1")


@dataclass(frozen=True)
class HeadPalette:
    """Robust skin and hair anchors in unit RGB."""

    skin_rgb: tuple[float, float, float]
    hair_rgb: tuple[float, float, float]
    skin_sample_count: int
    hair_sample_count: int


@dataclass(frozen=True)
class HeadOwnershipResult:
    """Per-sample ownership probabilities and protected-pixel masks."""

    labels: np.ndarray
    face_probability: np.ndarray
    hair_probability: np.ndarray
    ink_mask: np.ndarray
    highlight_mask: np.ndarray
    valid_mask: np.ndarray
    palette: HeadPalette


@dataclass(frozen=True)
class HeadGuardResult:
    """Result of replacing only high-confidence face/hair contamination."""

    corrected: np.ndarray
    replacement_strength: np.ndarray
    face_hair_bleed_mask: np.ndarray
    hair_face_bleed_mask: np.ndarray
    observed: HeadOwnershipResult
    repaired_sample_count: int


@dataclass(frozen=True)
class HeadConsistencyScore:
    """RC-ready semantic score for one candidate texture or render."""

    rc_score: float
    face_compatibility: float
    hair_compatibility: float
    semantic_coverage: float
    face_hair_bleed: float
    hair_face_bleed: float
    face_sample_count: int
    hair_sample_count: int


_FALLBACK_SKIN_RGB = np.asarray((0.82, 0.63, 0.52), dtype=np.float64)
_FALLBACK_HAIR_RGB = np.asarray((0.05, 0.66, 0.70), dtype=np.float64)


def _shape_for_colors(rgb: np.ndarray) -> tuple[int, ...]:
    array = np.asarray(rgb)
    if array.ndim < 1 or array.shape[-1] not in (3, 4):
        raise ValueError("rgb must have a final dimension of three or four")
    return array.shape[:-1]


def _boolean_mask(
    mask: np.ndarray | None,
    shape: tuple[int, ...],
    *,
    default: bool,
    name: str,
) -> np.ndarray:
    if mask is None:
        return np.full(shape, default, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _probability(
    values: np.ndarray | None,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must contain finite values between 0 and 1")
    return result


def _circular_hue_distance(first: np.ndarray, second: float) -> np.ndarray:
    direct = np.abs(first - second)
    return np.minimum(direct, 1.0 - direct)


def _dominant_hue_subset(
    hsv: np.ndarray,
    candidates: np.ndarray,
    *,
    hue_min: float,
    hue_max: float,
) -> np.ndarray:
    """Keep the strongest local hue mode so blue clothing cannot move hair."""

    indices = np.flatnonzero(candidates)
    if len(indices) < 8:
        return candidates
    flat_hsv = hsv.reshape((-1, 3))
    hues = flat_hsv[indices, 0]
    weights = (
        flat_hsv[indices, 1]
        * np.sqrt(np.maximum(flat_hsv[indices, 2], 0.0))
    )
    edges = np.linspace(hue_min, hue_max, 25)
    histogram, _ = np.histogram(hues, bins=edges, weights=weights)
    peak = int(np.argmax(histogram))
    lower = edges[max(0, peak - 1)]
    upper = edges[min(len(edges) - 1, peak + 2)]
    selected = np.zeros(candidates.size, dtype=bool)
    selected[indices[(hues >= lower) & (hues <= upper)]] = True
    return selected.reshape(candidates.shape)


def _median_or_fallback(
    unit_rgb: np.ndarray,
    mask: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, int]:
    samples = unit_rgb[mask]
    if not len(samples):
        return fallback.copy(), 0
    return np.median(samples, axis=0), len(samples)


def estimate_head_palette(
    rgb: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    face_seed_mask: np.ndarray | None = None,
    hair_seed_mask: np.ndarray | None = None,
    config: HeadOwnershipConfig | None = None,
) -> HeadPalette:
    """Estimate robust skin and saturated cool-hair anchors.

    Optional seed masks normally come from geometry or a stable reference
    view.  They restrict palette estimation but are not required.
    """

    settings = config or HeadOwnershipConfig()
    shape = _shape_for_colors(rgb)
    unit = rgb_to_unit(rgb)
    hsv = rgb_to_hsv(unit)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    valid = _boolean_mask(
        valid_mask,
        shape,
        default=True,
        name="valid_mask",
    )
    face_seed = _boolean_mask(
        face_seed_mask,
        shape,
        default=True,
        name="face_seed_mask",
    )
    hair_seed = _boolean_mask(
        hair_seed_mask,
        shape,
        default=True,
        name="hair_seed_mask",
    )
    skin_candidates = (
        valid
        & face_seed
        & (hue >= settings.skin_hue_min)
        & (hue <= settings.skin_hue_max)
        & (saturation >= settings.skin_saturation_min)
        & (saturation <= settings.skin_saturation_max)
        & (value >= settings.skin_value_min)
    )
    cool_candidates = (
        valid
        & hair_seed
        & (hue >= settings.cool_hue_min)
        & (hue <= settings.cool_hue_max)
        & (saturation >= settings.cool_saturation_min)
        & (value >= settings.cool_value_min)
    )
    cool_candidates = _dominant_hue_subset(
        hsv,
        cool_candidates,
        hue_min=settings.cool_hue_min,
        hue_max=settings.cool_hue_max,
    )
    skin_rgb, skin_count = _median_or_fallback(
        unit,
        skin_candidates,
        _FALLBACK_SKIN_RGB,
    )
    hair_rgb, hair_count = _median_or_fallback(
        unit,
        cool_candidates,
        _FALLBACK_HAIR_RGB,
    )
    return HeadPalette(
        skin_rgb=tuple(float(channel) for channel in skin_rgb),
        hair_rgb=tuple(float(channel) for channel in hair_rgb),
        skin_sample_count=int(skin_count),
        hair_sample_count=int(hair_count),
    )


def _validate_palette(palette: HeadPalette) -> tuple[np.ndarray, np.ndarray]:
    skin = np.asarray(palette.skin_rgb, dtype=np.float64)
    hair = np.asarray(palette.hair_rgb, dtype=np.float64)
    if skin.shape != (3,) or hair.shape != (3,):
        raise ValueError("palette colors must contain exactly three channels")
    if (
        not np.all(np.isfinite(skin))
        or not np.all(np.isfinite(hair))
        or np.any((skin < 0.0) | (skin > 1.0))
        or np.any((hair < 0.0) | (hair > 1.0))
    ):
        raise ValueError("palette colors must be finite unit RGB")
    return skin, hair


def _color_scores(
    unit_rgb: np.ndarray,
    palette: HeadPalette,
    settings: HeadOwnershipConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = rgb_to_hsv(unit_rgb)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    skin_rgb, hair_rgb = _validate_palette(palette)
    skin_hsv = rgb_to_hsv(skin_rgb)
    hair_hsv = rgb_to_hsv(hair_rgb)

    skin_hue = np.exp(
        -0.5
        * (
            _circular_hue_distance(hue, float(skin_hsv[0]))
            / settings.skin_hue_sigma
        )
        ** 2
    )
    skin_saturation = np.exp(
        -0.5
        * (
            (saturation - float(skin_hsv[1]))
            / settings.skin_saturation_sigma
        )
        ** 2
    )
    skin_value = np.exp(
        -0.5
        * ((value - float(skin_hsv[2])) / settings.skin_value_sigma) ** 2
    )
    skin_score = skin_hue * np.sqrt(skin_saturation * skin_value)
    skin_range = (
        (hue >= settings.skin_hue_min)
        & (hue <= settings.skin_hue_max)
        & (saturation >= settings.skin_saturation_min)
        & (saturation <= settings.skin_saturation_max)
        & (value >= settings.skin_value_min)
    )
    skin_score *= skin_range

    hair_hue = np.exp(
        -0.5
        * (
            _circular_hue_distance(hue, float(hair_hsv[0]))
            / settings.hair_hue_sigma
        )
        ** 2
    )
    hair_saturation = np.exp(
        -0.5
        * (
            (saturation - float(hair_hsv[1]))
            / settings.hair_saturation_sigma
        )
        ** 2
    )
    hair_value = np.exp(
        -0.5
        * ((value - float(hair_hsv[2])) / settings.hair_value_sigma) ** 2
    )
    hair_score = hair_hue * np.sqrt(hair_saturation * hair_value)
    hair_range = (
        (hue >= settings.cool_hue_min)
        & (hue <= settings.cool_hue_max)
        & (saturation >= settings.cool_saturation_min)
        & (value >= settings.cool_value_min)
    )
    hair_score *= hair_range

    warm = (
        (hue >= settings.skin_hue_min)
        & (hue <= settings.skin_hue_max)
        & (saturation >= settings.warm_contamination_saturation_min)
        & (value >= settings.skin_value_min)
    )
    warm_strength = np.clip(
        (
            saturation - settings.warm_contamination_saturation_min
        )
        / max(1.0 - settings.warm_contamination_saturation_min, 1.0e-8),
        0.0,
        1.0,
    )
    warm_strength *= warm
    return skin_score, hair_score, warm_strength


def classify_head_ownership(
    rgb: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    face_prior: np.ndarray | None = None,
    hair_prior: np.ndarray | None = None,
    palette: HeadPalette | None = None,
    config: HeadOwnershipConfig | None = None,
) -> HeadOwnershipResult:
    """Classify face, cyan hair, protected ink and neutral highlights."""

    settings = config or HeadOwnershipConfig()
    shape = _shape_for_colors(rgb)
    unit = rgb_to_unit(rgb)
    valid = _boolean_mask(
        valid_mask,
        shape,
        default=True,
        name="valid_mask",
    )
    face = _probability(face_prior, shape, name="face_prior")
    hair = _probability(hair_prior, shape, name="hair_prior")
    if (face is None) != (hair is None):
        raise ValueError("face_prior and hair_prior must be supplied together")

    selected_palette = palette or estimate_head_palette(
        unit,
        valid_mask=valid,
        face_seed_mask=None if face is None else face >= hair,
        hair_seed_mask=None if hair is None else hair > face,
        config=settings,
    )
    skin_score, hair_score, _warm = _color_scores(
        unit,
        selected_palette,
        settings,
    )
    hsv = rgb_to_hsv(unit)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    ink = valid & (value <= settings.ink_value_max)
    highlight = (
        valid
        & ~ink
        & (value >= settings.highlight_value_min)
        & (saturation <= settings.highlight_saturation_max)
    )

    if face is not None and hair is not None:
        floor = 1.0 - settings.prior_strength
        skin_score *= floor + settings.prior_strength * face
        hair_score *= floor + settings.prior_strength * hair
    total = skin_score + hair_score
    face_probability = np.divide(
        skin_score,
        total,
        out=np.zeros_like(total),
        where=total > 1.0e-12,
    )
    hair_probability = np.divide(
        hair_score,
        total,
        out=np.zeros_like(total),
        where=total > 1.0e-12,
    )

    labels = np.full(shape, int(HeadSemanticClass.UNKNOWN), dtype=np.int8)
    labels[~valid] = int(HeadSemanticClass.INVALID)
    confident = (
        valid
        & ~ink
        & ~highlight
        & (np.maximum(skin_score, hair_score) >= settings.minimum_color_confidence)
    )
    labels[confident & (skin_score >= hair_score)] = int(
        HeadSemanticClass.FACE
    )
    labels[confident & (hair_score > skin_score)] = int(
        HeadSemanticClass.HAIR
    )
    labels[highlight] = int(HeadSemanticClass.HIGHLIGHT)
    labels[ink] = int(HeadSemanticClass.INK)
    return HeadOwnershipResult(
        labels=labels,
        face_probability=face_probability.astype(np.float32),
        hair_probability=hair_probability.astype(np.float32),
        ink_mask=ink,
        highlight_mask=highlight,
        valid_mask=valid,
        palette=selected_palette,
    )


def _palette_with_original_luminance(
    original: np.ndarray,
    palette_rgb: np.ndarray,
) -> np.ndarray:
    coefficients = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    source_luminance = np.einsum("...c,c->...", original, coefficients)
    palette_luminance = max(float(np.dot(palette_rgb, coefficients)), 1.0e-6)
    scale = source_luminance / palette_luminance
    return np.clip(palette_rgb * scale[..., None], 0.0, 1.0)


def _restore_dtype(unit: np.ndarray, source: np.ndarray) -> np.ndarray:
    if np.issubdtype(source.dtype, np.bool_):
        return unit >= 0.5
    if np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        return np.rint(np.clip(unit, 0.0, 1.0) * maximum).astype(source.dtype)
    source_maximum = float(np.max(source[..., :3])) if source.size else 0.0
    scale = 255.0 if source_maximum > 1.0 + 1.0e-6 else 1.0
    return (np.clip(unit, 0.0, 1.0) * scale).astype(source.dtype)


def apply_head_color_guard(
    rgb: np.ndarray,
    expected_face: np.ndarray,
    expected_hair: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    palette: HeadPalette | None = None,
    strength: float = 1.0,
    config: HeadOwnershipConfig | None = None,
) -> HeadGuardResult:
    """Repair face/hair color crossings while preserving ink and highlights.

    ``expected_face`` and ``expected_hair`` are geometric/topological
    ownership probabilities.  Only the dominant expectation is acted on, so
    an uncertain 50/50 boundary is left untouched.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    settings = config or HeadOwnershipConfig()
    source = np.asarray(rgb)
    shape = _shape_for_colors(source)
    unit = rgb_to_unit(source)
    valid = _boolean_mask(
        valid_mask,
        shape,
        default=True,
        name="valid_mask",
    )
    expected_face_array = _probability(
        expected_face,
        shape,
        name="expected_face",
    )
    expected_hair_array = _probability(
        expected_hair,
        shape,
        name="expected_hair",
    )
    assert expected_face_array is not None
    assert expected_hair_array is not None
    selected_palette = palette or estimate_head_palette(
        unit,
        valid_mask=valid,
        face_seed_mask=expected_face_array > expected_hair_array,
        hair_seed_mask=expected_hair_array > expected_face_array,
        config=settings,
    )
    observed = classify_head_ownership(
        unit,
        valid_mask=valid,
        palette=selected_palette,
        config=settings,
    )
    _skin_score, _hair_score, warm_strength = _color_scores(
        unit,
        selected_palette,
        settings,
    )

    face_expectation = np.clip(
        expected_face_array - expected_hair_array,
        0.0,
        1.0,
    )
    hair_expectation = np.clip(
        expected_hair_array - expected_face_array,
        0.0,
        1.0,
    )
    protected = observed.ink_mask | observed.highlight_mask | ~valid
    face_contamination = (
        face_expectation * observed.hair_probability.astype(np.float64)
    )
    hair_contamination = hair_expectation * np.maximum(
        observed.face_probability.astype(np.float64),
        warm_strength,
    )
    face_contamination[protected] = 0.0
    hair_contamination[protected] = 0.0
    contamination = np.maximum(face_contamination, hair_contamination)
    replacement = strength * np.clip(
        (
            contamination - settings.guard_minimum_contamination
        )
        / max(1.0 - settings.guard_minimum_contamination, 1.0e-8),
        0.0,
        1.0,
    )

    skin_rgb, hair_rgb = _validate_palette(selected_palette)
    face_target = _palette_with_original_luminance(unit, skin_rgb)
    hair_target = _palette_with_original_luminance(unit, hair_rgb)
    target = np.where(
        (face_contamination >= hair_contamination)[..., None],
        face_target,
        hair_target,
    )
    corrected_rgb = (
        unit * (1.0 - replacement[..., None])
        + target * replacement[..., None]
    )
    if source.shape[-1] == 4:
        # rgb_to_unit intentionally drops alpha, so preserve its raw value by
        # restoring RGB first and concatenating the untouched source alpha.
        restored_rgb = _restore_dtype(corrected_rgb, source[..., :3])
        corrected = np.concatenate((restored_rgb, source[..., 3:4]), axis=-1)
    else:
        corrected = _restore_dtype(corrected_rgb, source)

    return HeadGuardResult(
        corrected=corrected,
        replacement_strength=replacement.astype(np.float32),
        face_hair_bleed_mask=(
            face_contamination > settings.guard_minimum_contamination
        ),
        hair_face_bleed_mask=(
            hair_contamination > settings.guard_minimum_contamination
        ),
        observed=observed,
        repaired_sample_count=int(np.count_nonzero(replacement > 0.0)),
    )


def score_head_consistency(
    rgb: np.ndarray,
    expected_face: np.ndarray,
    expected_hair: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    palette: HeadPalette | None = None,
    config: HeadOwnershipConfig | None = None,
) -> HeadConsistencyScore:
    """Score semantic reprojection consistency on a 0..100 RC scale.

    Dark ink and neutral highlights are excluded from bleed denominators.
    They are valid details on both face and hair and should neither inflate
    nor reduce the semantic score.
    """

    settings = config or HeadOwnershipConfig()
    shape = _shape_for_colors(rgb)
    valid = _boolean_mask(
        valid_mask,
        shape,
        default=True,
        name="valid_mask",
    )
    face = _probability(expected_face, shape, name="expected_face")
    hair = _probability(expected_hair, shape, name="expected_hair")
    assert face is not None
    assert hair is not None
    selected_palette = palette or estimate_head_palette(
        rgb,
        valid_mask=valid,
        face_seed_mask=face > hair,
        hair_seed_mask=hair > face,
        config=settings,
    )
    observed = classify_head_ownership(
        rgb,
        valid_mask=valid,
        palette=selected_palette,
        config=settings,
    )
    unit = rgb_to_unit(rgb)
    _skin_score, _hair_score, warm_strength = _color_scores(
        unit,
        selected_palette,
        settings,
    )
    eligible = valid & ~observed.ink_mask & ~observed.highlight_mask
    face_weight = np.clip(face - hair, 0.0, 1.0) * eligible
    hair_weight = np.clip(hair - face, 0.0, 1.0) * eligible

    face_total = float(face_weight.sum())
    hair_total = float(hair_weight.sum())
    face_hair_bleed = (
        float(
            np.sum(
                face_weight
                * observed.hair_probability.astype(np.float64)
            )
            / face_total
        )
        if face_total > 1.0e-12
        else 0.0
    )
    hair_face_bleed = (
        float(
            np.sum(
                hair_weight
                * np.maximum(
                    observed.face_probability.astype(np.float64),
                    warm_strength,
                )
            )
            / hair_total
        )
        if hair_total > 1.0e-12
        else 0.0
    )
    ownership_weight = face_weight + hair_weight
    ownership_total = float(ownership_weight.sum())
    classified = (
        (observed.labels == int(HeadSemanticClass.FACE))
        | (observed.labels == int(HeadSemanticClass.HAIR))
    )
    semantic_coverage = (
        float(np.sum(ownership_weight * classified) / ownership_total)
        if ownership_total > 1.0e-12
        else 0.0
    )
    face_compatibility = 1.0 - face_hair_bleed
    hair_compatibility = 1.0 - hair_face_bleed
    terms: list[tuple[float, float]] = []
    if face_total > 1.0e-12:
        terms.append((face_compatibility, face_total))
    if hair_total > 1.0e-12:
        terms.append((hair_compatibility, hair_total))
    semantic_compatibility = (
        sum(value * weight for value, weight in terms)
        / sum(weight for _value, weight in terms)
        if terms
        else 0.0
    )
    rc_score = 100.0 * np.clip(
        0.95 * semantic_compatibility + 0.05 * semantic_coverage,
        0.0,
        1.0,
    )
    return HeadConsistencyScore(
        rc_score=float(rc_score),
        face_compatibility=float(face_compatibility),
        hair_compatibility=float(hair_compatibility),
        semantic_coverage=float(semantic_coverage),
        face_hair_bleed=float(face_hair_bleed),
        hair_face_bleed=float(hair_face_bleed),
        face_sample_count=int(np.count_nonzero(face_weight > 0.0)),
        hair_sample_count=int(np.count_nonzero(hair_weight > 0.0)),
    )
