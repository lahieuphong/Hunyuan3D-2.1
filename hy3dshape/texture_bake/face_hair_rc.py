"""Deterministic face/hair texture candidate scoring.

``RC`` means *Reprojection/Region Consistency*.  The scorer is deliberately
independent from Blender and from model generation: callers render the same
head cameras for every candidate and pass those aligned RGBA renders together
with the reference views.

The score favours three things:

* semantic purity (skin does not leak into hair and cool hair does not leak
  into the face);
* robust colour agreement with the reference;
* preservation of the face/hair boundary and line detail.

The mean view score is blended with the worst quartile so one badly smeared
side cannot be hidden by several good views.  Hard gates are reported
separately and cap a rejected candidate below a passing candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .semantics import (
    SemanticClass,
    classify_anime_colors,
    default_compatibility_matrix,
)


ImageInput = np.ndarray | str | PathLike[str]
ViewInputs = (
    ImageInput
    | Sequence[ImageInput]
    | Mapping[str, ImageInput]
)


@dataclass(frozen=True)
class FaceHairRCConfig:
    """Thresholds and weights for :func:`score_face_hair_rc`.

    Defaults are conservative enough to reject obvious face/hair swaps while
    allowing small antialiasing and line-art differences.
    """

    alpha_threshold: float = 0.10
    minimum_region_pixels: int = 24
    semantic_weight: float = 0.55
    color_weight: float = 0.30
    detail_weight: float = 0.15
    worst_quartile_weight: float = 0.35
    maximum_worst_quartile_leakage: float = 0.12
    maximum_single_view_leakage: float = 0.35
    minimum_foreground_iou: float = 0.85
    minimum_color_score: float = 45.0
    minimum_detail_score: float = 35.0
    failed_score_cap: float = 49.0

    def validate(self) -> None:
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be between zero and one")
        if self.minimum_region_pixels < 1:
            raise ValueError("minimum_region_pixels must be positive")
        weights = (
            self.semantic_weight,
            self.color_weight,
            self.detail_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("metric weights must be non-negative and non-zero")
        if not 0.0 <= self.worst_quartile_weight <= 1.0:
            raise ValueError("worst_quartile_weight must be between zero and one")
        for name, value in (
            (
                "maximum_worst_quartile_leakage",
                self.maximum_worst_quartile_leakage,
            ),
            ("maximum_single_view_leakage", self.maximum_single_view_leakage),
            ("minimum_foreground_iou", self.minimum_foreground_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name, value in (
            ("minimum_color_score", self.minimum_color_score),
            ("minimum_detail_score", self.minimum_detail_score),
            ("failed_score_cap", self.failed_score_cap),
        ):
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between zero and 100")


@dataclass(frozen=True)
class FaceHairRCView:
    """Metrics for one aligned render/reference view."""

    name: str
    score: float
    semantic_score: float
    color_score: float
    detail_score: float
    foreground_iou: float
    face_leakage: float | None
    hair_leakage: float | None
    face_pixels: int
    hair_pixels: int


@dataclass(frozen=True)
class FaceHairRCResult:
    """Candidate score and the evidence used to accept or reject it."""

    score: float
    raw_score: float
    passed_hard_gates: bool
    hard_gate_failures: tuple[str, ...]
    mean_view_score: float
    worst_quartile_score: float
    semantic_score: float
    color_score: float
    detail_score: float
    foreground_iou: float
    worst_quartile_face_leakage: float | None
    worst_quartile_hair_leakage: float | None
    views: tuple[FaceHairRCView, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable report."""

        return asdict(self)


def _unit_rgba(image: np.ndarray, *, source: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"{source} must be an HxWx4 RGBA image")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{source} must be at least 2x2 pixels")
    if np.issubdtype(array.dtype, np.bool_):
        unit = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.integer):
        unit = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        unit = array.astype(np.float32)
        if not np.all(np.isfinite(unit)):
            raise ValueError(f"{source} contains non-finite values")
        minimum = float(unit.min()) if unit.size else 0.0
        maximum = float(unit.max()) if unit.size else 0.0
        if minimum < -1.0e-6:
            raise ValueError(f"{source} contains negative values")
        if maximum > 1.0 + 1.0e-6:
            if maximum <= 255.0 + 1.0e-6:
                unit /= 255.0
            else:
                raise ValueError(f"{source} must use a 0..1 or 0..255 range")
    return np.clip(unit, 0.0, 1.0)


def _read_rgba(value: ImageInput, *, source: str) -> np.ndarray:
    if isinstance(value, (str, PathLike)):
        path = Path(value)
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"could not read {source}: {path}")
        if loaded.ndim != 3 or loaded.shape[2] != 4:
            raise ValueError(f"{source} must contain four RGBA channels: {path}")
        value = cv2.cvtColor(loaded, cv2.COLOR_BGRA2RGBA)
    return _unit_rgba(np.asarray(value), source=source)


def _as_sequence(value: ViewInputs) -> list[ImageInput]:
    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            return [value]
        if value.ndim == 4:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError("view arrays must have shape HxWx4 or VxHxWx4")
    if isinstance(value, (str, PathLike)):
        return [value]
    return list(value)


def _aligned_views(
    rendered_views: ViewInputs,
    reference_views: ViewInputs,
    view_names: Sequence[str] | None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rendered_mapping = isinstance(rendered_views, Mapping)
    reference_mapping = isinstance(reference_views, Mapping)
    if rendered_mapping != reference_mapping:
        raise ValueError("rendered and reference views must use the same input form")

    if rendered_mapping:
        if view_names is not None:
            raise ValueError("view_names cannot be supplied with mapped views")
        rendered_map = rendered_views
        reference_map = reference_views
        if set(rendered_map) != set(reference_map):
            raise ValueError("rendered and reference mappings must have the same keys")
        names = sorted(rendered_map, key=lambda item: (str(item).casefold(), str(item)))
        pairs = [(str(name), rendered_map[name], reference_map[name]) for name in names]
    else:
        rendered = _as_sequence(rendered_views)
        reference = _as_sequence(reference_views)
        if len(rendered) != len(reference):
            raise ValueError("rendered and reference view counts must match")
        if not rendered:
            raise ValueError("at least one aligned view is required")
        if view_names is None:
            names = [f"view_{index:02d}" for index in range(len(rendered))]
        else:
            names = [str(name) for name in view_names]
            if len(names) != len(rendered):
                raise ValueError("view_names must match the number of views")
            if len(set(names)) != len(names):
                raise ValueError("view_names must be unique")
        pairs = list(zip(names, rendered, reference))

    aligned: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, rendered_value, reference_value in pairs:
        rendered = _read_rgba(rendered_value, source=f"rendered view {name!r}")
        reference = _read_rgba(reference_value, source=f"reference view {name!r}")
        if rendered.shape != reference.shape:
            raise ValueError(
                f"aligned view {name!r} has mismatched shapes "
                f"{rendered.shape} and {reference.shape}"
            )
        aligned.append((name, rendered, reference))
    return aligned


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values, dtype=np.float64))


def _balanced_mean(values: Sequence[float | None], *, fallback: float) -> float:
    available = [float(value) for value in values if value is not None]
    return float(np.mean(available)) if available else float(fallback)


def _worst_quartile(
    values: Sequence[float],
    *,
    high_is_worst: bool = False,
) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot aggregate an empty metric")
    count = max(1, int(np.ceil(array.size * 0.25)))
    ordered = np.sort(array)
    selected = ordered[-count:] if high_is_worst else ordered[:count]
    return float(np.mean(selected))


def _robust_low_score(values: Sequence[float], worst_weight: float) -> float:
    mean = float(np.mean(np.asarray(tuple(values), dtype=np.float64)))
    worst = _worst_quartile(values)
    return (1.0 - worst_weight) * mean + worst_weight * worst


def _lab_color_score(
    rendered_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if not np.any(mask):
        return None
    rendered_lab = cv2.cvtColor(
        np.ascontiguousarray(rendered_rgb.astype(np.float32)),
        cv2.COLOR_RGB2LAB,
    )
    reference_lab = cv2.cvtColor(
        np.ascontiguousarray(reference_rgb.astype(np.float32)),
        cv2.COLOR_RGB2LAB,
    )
    delta = np.linalg.norm(rendered_lab - reference_lab, axis=2)[mask]
    median = float(np.median(delta))
    upper = float(np.percentile(delta, 90.0))
    robust_delta = 0.65 * median + 0.35 * upper
    return float(100.0 * np.exp(-robust_delta / 35.0))


def _gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(
        np.ascontiguousarray(rgb.astype(np.float32)),
        cv2.COLOR_RGB2GRAY,
    )
    x_gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    y_gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(x_gradient, y_gradient)


def _edge_f1(reference_edges: np.ndarray, rendered_edges: np.ndarray) -> float:
    if not np.any(reference_edges):
        return 1.0 if not np.any(rendered_edges) else 0.0
    if not np.any(rendered_edges):
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_dilated = cv2.dilate(reference_edges.astype(np.uint8), kernel) > 0
    rendered_dilated = cv2.dilate(rendered_edges.astype(np.uint8), kernel) > 0
    precision = float(np.mean(reference_dilated[rendered_edges]))
    recall = float(np.mean(rendered_dilated[reference_edges]))
    if precision + recall <= 1.0e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _detail_score(
    rendered_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    face_mask: np.ndarray,
    hair_mask: np.ndarray,
    foreground: np.ndarray,
) -> float:
    height, width = foreground.shape
    radius = max(1, min(4, int(round(min(height, width) * 0.006))))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    regions = face_mask | hair_mask
    if np.any(regions):
        detail_mask = cv2.dilate(regions.astype(np.uint8), kernel) > 0
        detail_mask &= foreground
    else:
        detail_mask = foreground
    if not np.any(detail_mask):
        return 0.0

    reference_gradient = _gradient_magnitude(reference_rgb)
    rendered_gradient = _gradient_magnitude(rendered_rgb)
    reference_scale = float(np.percentile(reference_gradient[detail_mask], 90.0))
    reference_scale = max(reference_scale, 1.0e-4)
    reference_normalized = np.clip(reference_gradient / reference_scale, 0.0, 1.0)
    rendered_normalized = np.clip(rendered_gradient / reference_scale, 0.0, 1.0)
    gradient_similarity = 1.0 - float(
        np.mean(
            np.abs(reference_normalized[detail_mask] - rendered_normalized[detail_mask]),
            dtype=np.float64,
        )
    )

    reference_edges = (reference_normalized >= 0.24) & detail_mask
    rendered_edges = (rendered_normalized >= 0.24) & detail_mask
    edge_f1 = _edge_f1(reference_edges, rendered_edges)

    boundary = np.zeros_like(foreground)
    if np.any(face_mask) and np.any(hair_mask):
        face_near = cv2.dilate(face_mask.astype(np.uint8), kernel) > 0
        hair_near = cv2.dilate(hair_mask.astype(np.uint8), kernel) > 0
        boundary = face_near & hair_near & foreground
    boundary_reference_edges = reference_edges & boundary
    if np.any(boundary_reference_edges):
        rendered_dilated = cv2.dilate(rendered_edges.astype(np.uint8), kernel) > 0
        boundary_recall = float(np.mean(rendered_dilated[boundary_reference_edges]))
        result = (
            0.45 * edge_f1
            + 0.35 * max(0.0, gradient_similarity)
            + 0.20 * boundary_recall
        )
    else:
        result = 0.56 * edge_f1 + 0.44 * max(0.0, gradient_similarity)
    return float(np.clip(result * 100.0, 0.0, 100.0))


def _score_view(
    name: str,
    rendered: np.ndarray,
    reference: np.ndarray,
    config: FaceHairRCConfig,
) -> FaceHairRCView:
    rendered_rgb = rendered[..., :3]
    reference_rgb = reference[..., :3]
    rendered_foreground = rendered[..., 3] >= config.alpha_threshold
    reference_foreground = reference[..., 3] >= config.alpha_threshold
    intersection = np.count_nonzero(rendered_foreground & reference_foreground)
    union = np.count_nonzero(rendered_foreground | reference_foreground)
    foreground_iou = float(intersection / union) if union else 1.0

    reference_labels = classify_anime_colors(reference_rgb)
    rendered_labels = classify_anime_colors(rendered_rgb)
    face_mask = (
        reference_foreground
        & (reference_labels == int(SemanticClass.SKIN_LIKE))
    )
    hair_mask = (
        reference_foreground
        & (reference_labels == int(SemanticClass.COOL_SATURATED))
    )
    face_pixels = int(np.count_nonzero(face_mask))
    hair_pixels = int(np.count_nonzero(hair_mask))
    if face_pixels < config.minimum_region_pixels:
        face_mask[:] = False
        face_pixels = 0
    if hair_pixels < config.minimum_region_pixels:
        hair_mask[:] = False
        hair_pixels = 0

    face_leakage = _mean_or_none(
        np.isin(
            rendered_labels[face_mask],
            (
                int(SemanticClass.WARM_SATURATED),
                int(SemanticClass.COOL_SATURATED),
            ),
        )
    )
    hair_leakage = _mean_or_none(
        np.isin(
            rendered_labels[hair_mask],
            (
                int(SemanticClass.SKIN_LIKE),
                int(SemanticClass.WARM_SATURATED),
            ),
        )
    )

    compatibility = default_compatibility_matrix()
    face_semantic = (
        _mean_or_none(
            compatibility[
                rendered_labels[face_mask],
                int(SemanticClass.SKIN_LIKE),
            ]
        )
        if face_pixels
        else None
    )
    hair_semantic = (
        _mean_or_none(
            compatibility[
                rendered_labels[hair_mask],
                int(SemanticClass.COOL_SATURATED),
            ]
        )
        if hair_pixels
        else None
    )
    semantic_score = 100.0 * _balanced_mean(
        (face_semantic, hair_semantic),
        fallback=0.0,
    )

    face_color = _lab_color_score(rendered_rgb, reference_rgb, face_mask)
    hair_color = _lab_color_score(rendered_rgb, reference_rgb, hair_mask)
    color_score = _balanced_mean((face_color, hair_color), fallback=0.0)
    detail_score = _detail_score(
        rendered_rgb,
        reference_rgb,
        face_mask,
        hair_mask,
        reference_foreground,
    )
    # A candidate with missing pixels must not receive full detail credit.
    detail_score *= 0.75 + 0.25 * foreground_iou

    weight_sum = (
        config.semantic_weight + config.color_weight + config.detail_weight
    )
    score = (
        config.semantic_weight * semantic_score
        + config.color_weight * color_score
        + config.detail_weight * detail_score
    ) / weight_sum
    return FaceHairRCView(
        name=name,
        score=float(np.clip(score, 0.0, 100.0)),
        semantic_score=float(np.clip(semantic_score, 0.0, 100.0)),
        color_score=float(np.clip(color_score, 0.0, 100.0)),
        detail_score=float(np.clip(detail_score, 0.0, 100.0)),
        foreground_iou=foreground_iou,
        face_leakage=face_leakage,
        hair_leakage=hair_leakage,
        face_pixels=face_pixels,
        hair_pixels=hair_pixels,
    )


def score_face_hair_rc(
    rendered_views: ViewInputs,
    reference_views: ViewInputs,
    *,
    view_names: Sequence[str] | None = None,
    config: FaceHairRCConfig | None = None,
) -> FaceHairRCResult:
    """Score aligned RGBA candidate renders against head reference views.

    Inputs may be RGBA arrays, image file paths, sequences of either, or
    mappings from view name to image.  Mapped views are aligned by key;
    sequences are aligned by index.  Shapes must already match because silent
    resizing would make reprojection errors look artificially better.
    """

    settings = config or FaceHairRCConfig()
    settings.validate()
    aligned = _aligned_views(rendered_views, reference_views, view_names)
    views = tuple(
        _score_view(name, rendered, reference, settings)
        for name, rendered, reference in aligned
    )

    view_scores = [view.score for view in views]
    mean_view_score = float(np.mean(view_scores))
    worst_quartile_score = _worst_quartile(view_scores)
    raw_score = (
        (1.0 - settings.worst_quartile_weight) * mean_view_score
        + settings.worst_quartile_weight * worst_quartile_score
    )
    semantic_score = _robust_low_score(
        [view.semantic_score for view in views],
        settings.worst_quartile_weight,
    )
    color_score = _robust_low_score(
        [view.color_score for view in views],
        settings.worst_quartile_weight,
    )
    detail_score = _robust_low_score(
        [view.detail_score for view in views],
        settings.worst_quartile_weight,
    )
    foreground_iou = _robust_low_score(
        [view.foreground_iou for view in views],
        settings.worst_quartile_weight,
    )

    face_leakages = [
        view.face_leakage for view in views if view.face_leakage is not None
    ]
    hair_leakages = [
        view.hair_leakage for view in views if view.hair_leakage is not None
    ]
    worst_face = (
        _worst_quartile(face_leakages, high_is_worst=True)
        if face_leakages
        else None
    )
    worst_hair = (
        _worst_quartile(hair_leakages, high_is_worst=True)
        if hair_leakages
        else None
    )

    failures: list[str] = []
    if not face_leakages:
        failures.append("missing_face_reference_region")
    if not hair_leakages:
        failures.append("missing_hair_reference_region")
    if foreground_iou < settings.minimum_foreground_iou:
        failures.append("foreground_iou_below_minimum")
    if (
        worst_face is not None
        and worst_face > settings.maximum_worst_quartile_leakage
    ):
        failures.append("face_leakage_above_limit")
    if (
        worst_hair is not None
        and worst_hair > settings.maximum_worst_quartile_leakage
    ):
        failures.append("hair_leakage_above_limit")
    maximum_leakage = max(face_leakages + hair_leakages, default=0.0)
    if maximum_leakage > settings.maximum_single_view_leakage:
        failures.append("single_view_leakage_above_limit")
    if color_score < settings.minimum_color_score:
        failures.append("color_consistency_below_minimum")
    if detail_score < settings.minimum_detail_score:
        failures.append("boundary_detail_below_minimum")

    passed = not failures
    score = raw_score if passed else min(raw_score, settings.failed_score_cap)
    return FaceHairRCResult(
        score=float(np.clip(score, 0.0, 100.0)),
        raw_score=float(np.clip(raw_score, 0.0, 100.0)),
        passed_hard_gates=passed,
        hard_gate_failures=tuple(failures),
        mean_view_score=mean_view_score,
        worst_quartile_score=worst_quartile_score,
        semantic_score=semantic_score,
        color_score=color_score,
        detail_score=detail_score,
        foreground_iou=foreground_iou,
        worst_quartile_face_leakage=worst_face,
        worst_quartile_hair_leakage=worst_hair,
        views=views,
    )


def rank_face_hair_candidates(
    candidates: Mapping[str, FaceHairRCResult]
    | Sequence[tuple[str, FaceHairRCResult]],
) -> tuple[str, ...]:
    """Return candidate ids in deterministic best-first order.

    Hard-gate acceptance always remains the first priority.  Within the same
    gate class, candidates first have to contain assessable face/hair regions
    and viable boundary detail.  That appearance tier prevents an untextured
    or washed candidate from looking artificially clean merely because it has
    no colours that can leak.  Among viable textured candidates, robust head
    quality is compared using a head-clarity composite made from 80% robust
    worst-quartile quality and 20% boundary detail.  The raw worst quartile,
    worst individual view, and boundary detail then break close calls before
    semantic leakage.  This keeps a slightly cleaner but visibly washed face
    from winning solely through deceptively low leakage, while leakage remains
    the primary tie-breaker between candidates with equally robust head
    quality.  Aggregate scores and the
    case-insensitive candidate id are deterministic final tie-breakers.
    """

    items = list(candidates.items()) if isinstance(candidates, Mapping) else list(candidates)
    if not items:
        raise ValueError("at least one candidate result is required")
    ids = [str(candidate_id) for candidate_id, _ in items]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    for _, result in items:
        if not isinstance(result, FaceHairRCResult):
            raise TypeError("candidate values must be FaceHairRCResult instances")

    def appearance_viability_tier(result: FaceHairRCResult) -> int:
        failures = set(result.hard_gate_failures)
        missing_region = bool(
            failures
            & {
                "missing_face_reference_region",
                "missing_hair_reference_region",
            }
        )
        detail_failed = "boundary_detail_below_minimum" in failures
        color_failed = "color_consistency_below_minimum" in failures
        if missing_region:
            return 3
        if detail_failed and color_failed:
            # This is the characteristic zero-leakage-but-uncoloured fallback:
            # both its colour agreement and boundary evidence are absent.
            return 2
        if detail_failed:
            return 1
        return 0

    def worst_view_score(result: FaceHairRCResult) -> float:
        scores = [
            float(view.score)
            for view in result.views
            if np.isfinite(view.score) and 0.0 <= float(view.score) <= 100.0
        ]
        # Missing/invalid view evidence cannot outrank a candidate assessed
        # across every available head view.
        return min(scores, default=float("-inf"))

    def robust_head_clarity(result: FaceHairRCResult) -> float:
        """Blend bad-angle robustness with preserved facial line detail."""

        return (
            0.80 * float(result.worst_quartile_score)
            + 0.20 * float(result.detail_score)
        )

    def finite_leakage(value: float | None) -> float | None:
        if value is None:
            return None
        leakage = float(value)
        if not np.isfinite(leakage) or not 0.0 <= leakage <= 1.0:
            return float("inf")
        return leakage

    def leakage_priority(result: FaceHairRCResult) -> tuple[float, ...]:
        category_maxima: list[float] = []
        for attribute in ("face_leakage", "hair_leakage"):
            values = [
                finite
                for view in result.views
                if (finite := finite_leakage(getattr(view, attribute)))
                is not None
            ]
            # A completely unassessed face or hair category must not look
            # cleaner than an assessed category.  The scorer also records a
            # hard-gate failure for this case.
            category_maxima.append(max(values, default=float("inf")))

        quartiles = [
            finite_leakage(result.worst_quartile_face_leakage),
            finite_leakage(result.worst_quartile_hair_leakage),
        ]
        quartile_values = [
            value if value is not None else float("inf")
            for value in quartiles
        ]
        # Compare the worse category first, then the other category.  This is
        # symmetric between face and hair and avoids privileging either one.
        return (
            max(category_maxima),
            min(category_maxima),
            max(quartile_values),
            min(quartile_values),
        )

    ranked = sorted(
        ((str(candidate_id), result) for candidate_id, result in items),
        key=lambda item: (
            0 if item[1].passed_hard_gates else 1,
            appearance_viability_tier(item[1]),
            -robust_head_clarity(item[1]),
            -item[1].worst_quartile_score,
            -worst_view_score(item[1]),
            -item[1].detail_score,
            *leakage_priority(item[1]),
            -item[1].score,
            -item[1].raw_score,
            -item[1].mean_view_score,
            item[0].casefold(),
            item[0],
        ),
    )
    return tuple(candidate_id for candidate_id, _ in ranked)


def select_best_face_hair_candidate(
    candidates: Mapping[str, FaceHairRCResult]
    | Sequence[tuple[str, FaceHairRCResult]],
) -> tuple[str, FaceHairRCResult]:
    """Return the deterministic best candidate id and its RC report."""

    if isinstance(candidates, Mapping):
        candidate_map = {str(key): value for key, value in candidates.items()}
    else:
        candidate_map = {str(key): value for key, value in candidates}
    ranking = rank_face_hair_candidates(candidate_map)
    winner = ranking[0]
    return winner, candidate_map[winner]


__all__ = [
    "FaceHairRCConfig",
    "FaceHairRCResult",
    "FaceHairRCView",
    "rank_face_hair_candidates",
    "score_face_hair_rc",
    "select_best_face_hair_candidate",
]
