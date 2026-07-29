"""Topology-only cleanup for small holes in trusted polygon masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolygonMaskExpansionStats:
    """Summary of enclosed-polygon mask expansion."""

    initially_marked: int
    added: int
    final_marked: int
    iterations_run: int
    added_per_iteration: tuple[int, ...]


def expand_enclosed_polygon_mask(
    mask: np.ndarray,
    loop_vertices: np.ndarray,
    polygon_loop_totals: np.ndarray,
    polygon_centers: np.ndarray,
    *,
    minimum_z: float,
    max_iterations: int = 3,
) -> tuple[np.ndarray, PolygonMaskExpansionStats]:
    """Fill high-region polygons whose every vertex touches the trusted mask.

    Requiring every corner to touch an already marked polygon fills isolated
    holes while avoiding ordinary two-vertex boundaries such as a hairline.
    """

    result = np.asarray(mask, dtype=bool).copy()
    loops = np.asarray(loop_vertices, dtype=np.int64)
    totals = np.asarray(polygon_loop_totals, dtype=np.int64)
    centers = np.asarray(polygon_centers, dtype=np.float64)
    polygon_count = len(result)
    if totals.shape != (polygon_count,):
        raise ValueError("polygon_loop_totals must have shape (P,)")
    if centers.shape != (polygon_count, 3):
        raise ValueError("polygon_centers must have shape (P, 3)")
    if int(totals.sum()) != len(loops):
        raise ValueError("polygon loop totals must sum to loop_vertices length")
    if loops.size and int(loops.min()) < 0:
        raise ValueError("loop vertex indices must be non-negative")
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")

    polygon_for_loop = np.repeat(
        np.arange(polygon_count, dtype=np.int64),
        totals,
    )
    vertex_count = int(loops.max()) + 1 if len(loops) else 0
    added_per_iteration: list[int] = []
    initially_marked = int(np.count_nonzero(result))

    for _ in range(max_iterations):
        vertex_touches_mask = np.zeros(vertex_count, dtype=bool)
        np.logical_or.at(
            vertex_touches_mask,
            loops,
            result[polygon_for_loop],
        )
        touched_corner_count = np.zeros(polygon_count, dtype=np.int16)
        np.add.at(
            touched_corner_count,
            polygon_for_loop,
            vertex_touches_mask[loops].astype(np.int16),
        )
        enclosed = (
            ~result
            & (centers[:, 2] >= minimum_z)
            & (touched_corner_count == totals)
        )
        count = int(np.count_nonzero(enclosed))
        if not count:
            break
        result[enclosed] = True
        added_per_iteration.append(count)

    added = int(np.count_nonzero(result)) - initially_marked
    return result, PolygonMaskExpansionStats(
        initially_marked=initially_marked,
        added=added,
        final_marked=int(np.count_nonzero(result)),
        iterations_run=len(added_per_iteration),
        added_per_iteration=tuple(added_per_iteration),
    )
