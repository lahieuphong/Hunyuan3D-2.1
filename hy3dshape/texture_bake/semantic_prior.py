"""Conservative surface-neighbour semantic prior for projected colors."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .semantics import (
    SemanticClass,
    apply_semantic_color_guards,
    classify_anime_colors,
)
from .surface_semantics import (
    SurfaceLabelDiffusionStats,
    diffuse_surface_labels,
)


@dataclass(frozen=True)
class SurfaceSemanticPriorResult:
    """Weights and graph constrained by a surface-local palette prior."""

    adjusted_weights: np.ndarray
    surface_labels: np.ndarray
    diffusion_edges: np.ndarray
    report: dict[str, object]


def apply_surface_semantic_prior(
    sampled_colors: np.ndarray,
    base_weights: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray,
    *,
    prior_iterations: int = 12,
    prior_normal_dot: float = 0.55,
    hard_reject_threshold: float = 0.45,
    minimum_seed_view_weight: float = 0.08,
    minimum_seed_agreeing_views: int = 2,
    minimum_seed_vote_fraction: float = 2.0 / 3.0,
    minimum_vote_fraction: float = 0.6,
) -> SurfaceSemanticPriorResult:
    """Reject singleton projected colors that conflict with their surface.

    Multi-view consensus creates trusted semantic seeds. Labels spread only
    through agreeing, similarly-oriented neighbours. Strongly incompatible
    projected samples are then set to zero so the caller can use a local
    surface-color fill instead of renormalizing a known bad singleton view.
    """

    colors = np.asarray(sampled_colors)
    weights = np.asarray(base_weights, dtype=np.float64)
    if colors.ndim != 3 or colors.shape[2] not in (3, 4):
        raise ValueError("sampled_colors must have shape (views, samples, 3/4)")
    if weights.shape != (colors.shape[1], colors.shape[0]):
        raise ValueError("base_weights must have shape (samples, views)")
    if not 0.0 <= hard_reject_threshold <= 1.0:
        raise ValueError("hard_reject_threshold must be between 0 and 1")
    if not 0.0 <= minimum_seed_view_weight <= 1.0:
        raise ValueError(
            "minimum_seed_view_weight must be between 0 and 1"
        )
    if minimum_seed_agreeing_views < 2:
        raise ValueError("minimum_seed_agreeing_views must be at least two")
    if not 0.0 < minimum_seed_vote_fraction <= 1.0:
        raise ValueError(
            "minimum_seed_vote_fraction must be between zero and one"
        )

    initial = apply_semantic_color_guards(
        colors,
        weights.T,
        minimum_guard=0.0,
    )
    unknown = int(SemanticClass.UNKNOWN)
    sampled_labels = classify_anime_colors(colors).T
    active_views = (
        weights >= minimum_seed_view_weight - 1.0e-7
    )
    semantic_active = active_views & (sampled_labels != unknown)
    view_counts = np.count_nonzero(active_views, axis=1)
    semantic_view_counts = np.count_nonzero(semantic_active, axis=1)
    class_count = max(int(value) for value in SemanticClass) + 1
    vote_counts = np.zeros(
        (len(weights), class_count),
        dtype=np.int16,
    )
    for semantic in SemanticClass:
        label = int(semantic)
        if label == unknown:
            continue
        vote_counts[:, label] = np.count_nonzero(
            semantic_active & (sampled_labels == label),
            axis=1,
        )
    winner_labels = np.argmax(vote_counts, axis=1).astype(np.int16)
    winner_counts = vote_counts[
        np.arange(len(weights)),
        winner_labels,
    ]
    seed_vote_fraction = np.divide(
        winner_counts,
        semantic_view_counts,
        out=np.zeros(len(weights), dtype=np.float64),
        where=semantic_view_counts > 0,
    )
    trusted = (
        (winner_counts >= minimum_seed_agreeing_views)
        & (seed_vote_fraction >= minimum_seed_vote_fraction)
    )
    seed_labels = winner_labels
    seed_labels[~trusted] = unknown

    if np.any(trusted):
        surface_labels, inferred, prior_stats = diffuse_surface_labels(
            seed_labels,
            trusted,
            edges,
            normals,
            unknown_label=int(SemanticClass.UNKNOWN),
            minimum_normal_dot=prior_normal_dot,
            minimum_vote_fraction=minimum_vote_fraction,
            max_iterations=prior_iterations,
        )
    else:
        surface_labels = seed_labels
        inferred = np.zeros(len(seed_labels), dtype=bool)
        prior_stats = SurfaceLabelDiffusionStats(
            initially_known=0,
            initially_missing=len(seed_labels),
            filled=0,
            remaining=len(seed_labels),
            iterations_run=0,
            filled_per_hop=(),
            usable_edges=0,
        )

    prior_known = trusted | inferred
    surface_labels[~prior_known] = int(SemanticClass.UNKNOWN)
    guarded = apply_semantic_color_guards(
        colors,
        weights.T,
        expected_labels=surface_labels,
        minimum_guard=0.0,
    )
    hard_incompatible = (
        prior_known[None, :]
        & (guarded.guard < hard_reject_threshold)
    )
    adjusted = guarded.adjusted_weights.T.astype(np.float32)
    adjusted[hard_incompatible.T] = 0.0

    edge_array = np.asarray(edges, dtype=np.int64)
    if len(edge_array):
        left, right = edge_array[:, 0], edge_array[:, 1]
        semantic_barrier = (
            prior_known[left]
            & prior_known[right]
            & (surface_labels[left] != surface_labels[right])
        )
        diffusion_edges = edge_array[~semantic_barrier]
    else:
        semantic_barrier = np.zeros(0, dtype=bool)
        diffusion_edges = edge_array

    expected_counts: dict[str, int] = {}
    for semantic in SemanticClass:
        count = int(
            np.count_nonzero(surface_labels == int(semantic))
        )
        if count:
            expected_counts[semantic.name.lower()] = count
    changed = np.max(np.abs(adjusted - weights), axis=1) > 0.05
    rejected_valid = hard_incompatible & (weights.T > 1.0e-10)
    report: dict[str, object] = {
        "enabled": True,
        "changed_vertices": int(np.count_nonzero(changed)),
        "mean_consensus_strength": float(
            initial.consensus_strength.mean()
        ),
        "expected_classes": expected_counts,
        "surface_prior": asdict(prior_stats),
        "trusted_seed_vertices": int(np.count_nonzero(trusted)),
        "inferred_prior_vertices": int(np.count_nonzero(inferred)),
        "hard_rejected_entries": int(np.count_nonzero(rejected_valid)),
        "hard_rejected_vertices": int(
            np.count_nonzero(np.any(rejected_valid, axis=0))
        ),
        "singleton_rejected_vertices": int(
            np.count_nonzero(
                (view_counts == 1)
                & np.any(rejected_valid, axis=0)
            )
        ),
        "semantic_barrier_edges": int(
            np.count_nonzero(semantic_barrier)
        ),
        "mean_guard_by_channel": [
            float(guarded.guard[index].mean())
            for index in range(colors.shape[0])
        ],
    }
    return SurfaceSemanticPriorResult(
        adjusted_weights=adjusted,
        surface_labels=surface_labels,
        diffusion_edges=diffusion_edges,
        report=report,
    )
