"""Surface-local semantic priors for multi-view texture projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceLabelDiffusionStats:
    """Summary of conservative nearest-layer label propagation."""

    initially_known: int
    initially_missing: int
    filled: int
    remaining: int
    iterations_run: int
    filled_per_hop: tuple[int, ...]
    usable_edges: int


def diffuse_surface_labels(
    labels: np.ndarray,
    known_mask: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray | None,
    *,
    unknown_label: int = 0,
    minimum_normal_dot: float | None = 0.5,
    minimum_vote_fraction: float = 0.6,
    max_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, SurfaceLabelDiffusionStats]:
    """Infer missing labels from the nearest agreeing surface neighbours.

    Only the previous BFS layer donates to the next one. Tied or weak votes
    stay unresolved, and a normal-dot barrier prevents labels crossing sharp
    folds. This makes the helper suitable as a conservative prior rather than
    a segmentation algorithm.
    """

    source = np.asarray(labels)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    result = source.astype(np.int16, copy=True)
    vertex_count = len(result)

    seeds = np.asarray(known_mask, dtype=bool)
    if seeds.shape != (vertex_count,):
        raise ValueError("known_mask must have shape (N,)")
    if np.any(result[seeds] == unknown_label):
        raise ValueError("known labels must not equal unknown_label")
    if not 0.5 <= minimum_vote_fraction <= 1.0:
        raise ValueError("minimum_vote_fraction must be between 0.5 and 1")
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")

    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (M, 2)")
    if edge_array.size and (
        int(edge_array.min()) < 0
        or int(edge_array.max()) >= vertex_count
    ):
        raise ValueError("edges contain an out-of-range vertex index")
    if len(edge_array):
        edge_array = np.sort(edge_array, axis=1)
        edge_array = edge_array[edge_array[:, 0] != edge_array[:, 1]]
        edge_array = np.unique(edge_array, axis=0)

    if minimum_normal_dot is None:
        usable = np.ones(len(edge_array), dtype=bool)
    else:
        if not -1.0 <= minimum_normal_dot <= 1.0:
            raise ValueError("minimum_normal_dot must be between -1 and 1")
        if normals is None:
            raise ValueError("normals are required with a normal barrier")
        normal_array = np.asarray(normals, dtype=np.float64)
        if normal_array.shape != (vertex_count, 3):
            raise ValueError("normals must have shape (N, 3)")
        lengths = np.linalg.norm(normal_array, axis=1)
        valid = lengths > 1.0e-12
        unit = np.zeros_like(normal_array)
        unit[valid] = normal_array[valid] / lengths[valid, None]
        if len(edge_array):
            left, right = edge_array[:, 0], edge_array[:, 1]
            dots = np.einsum("ij,ij->i", unit[left], unit[right])
            usable = (
                valid[left]
                & valid[right]
                & (dots >= minimum_normal_dot)
            )
        else:
            usable = np.zeros(0, dtype=bool)

    active_edges = edge_array[usable]
    class_ids = np.unique(result[seeds])
    resolved = seeds.copy()
    frontier = seeds.copy()
    filled_mask = np.zeros(vertex_count, dtype=bool)
    filled_per_hop: list[int] = []

    for _hop in range(max_iterations):
        if not np.any(frontier) or not len(active_edges):
            break
        votes = np.zeros((vertex_count, len(class_ids)), dtype=np.uint16)
        left, right = active_edges[:, 0], active_edges[:, 1]
        for column, class_id in enumerate(class_ids):
            left_donor = (
                frontier[left]
                & (result[left] == class_id)
                & ~resolved[right]
            )
            if np.any(left_donor):
                np.add.at(votes[:, column], right[left_donor], 1)

            right_donor = (
                frontier[right]
                & (result[right] == class_id)
                & ~resolved[left]
            )
            if np.any(right_donor):
                np.add.at(votes[:, column], left[right_donor], 1)

        totals = votes.sum(axis=1)
        winner_columns = votes.argmax(axis=1)
        winning_votes = votes[
            np.arange(vertex_count),
            winner_columns,
        ]
        vote_fraction = np.divide(
            winning_votes,
            totals,
            out=np.zeros(vertex_count, dtype=np.float64),
            where=totals > 0,
        )
        receivers = (
            ~resolved
            & (totals > 0)
            & (vote_fraction >= minimum_vote_fraction)
        )
        count = int(np.count_nonzero(receivers))
        if not count:
            break
        result[receivers] = class_ids[winner_columns[receivers]]
        resolved[receivers] = True
        filled_mask[receivers] = True
        frontier = receivers
        filled_per_hop.append(count)

    initially_known = int(np.count_nonzero(seeds))
    filled = int(np.count_nonzero(filled_mask))
    stats = SurfaceLabelDiffusionStats(
        initially_known=initially_known,
        initially_missing=vertex_count - initially_known,
        filled=filled,
        remaining=vertex_count - initially_known - filled,
        iterations_run=len(filled_per_hop),
        filled_per_hop=tuple(filled_per_hop),
        usable_edges=int(np.count_nonzero(usable)),
    )
    return result, filled_mask, stats
