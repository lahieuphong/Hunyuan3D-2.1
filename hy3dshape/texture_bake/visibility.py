"""Visibility and confidence primitives for multi-view texture baking.

The functions in this module operate on NumPy arrays and can be tested without
Blender.  Blender callers can either supply a rendered depth buffer or use the
ray-cast adapter in :mod:`hy3dshape.texture_bake.blender_visibility`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .calibration import (
    OrthographicCalibration,
    Projection,
    ViewFrame,
    alpha_to_unit,
)


def _points_array(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite values")
    return result


def _unit_rows(
    vectors: np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(vectors, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("vectors must contain only finite values")
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > 1e-12
    normalized = np.zeros_like(result)
    normalized[valid] = result[valid] / lengths[valid, None]
    return normalized, valid


def smoothstep01(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def angular_confidence(
    normals: np.ndarray | Sequence[Sequence[float]],
    frame_or_direction: ViewFrame | Sequence[float],
    *,
    minimum_cosine: float = 0.05,
    full_confidence_cosine: float = 0.65,
    exponent: float = 1.5,
) -> np.ndarray:
    """Score how directly each surface normal faces an orthographic camera."""

    if not -1.0 <= minimum_cosine < full_confidence_cosine <= 1.0:
        raise ValueError("cosine thresholds must satisfy -1 <= minimum < full <= 1")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    unit_normals, valid = _unit_rows(normals)
    direction = (
        frame_or_direction.to_camera_array
        if isinstance(frame_or_direction, ViewFrame)
        else np.asarray(frame_or_direction, dtype=np.float64)
    )
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("camera direction must contain three finite values")
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-12:
        raise ValueError("camera direction must not be zero")
    direction /= direction_length

    cosines = unit_normals @ direction
    ramp = (cosines - minimum_cosine) / (full_confidence_cosine - minimum_cosine)
    result = smoothstep01(ramp) ** exponent
    result[~valid] = 0.0
    return result


def alpha_confidence_map(
    alpha: np.ndarray,
    *,
    threshold: float = 0.5,
    feather_pixels: float = 8.0,
    multiply_soft_alpha: bool = True,
) -> np.ndarray:
    """Build confidence that fades towards the inside of a silhouette edge.

    Pixels outside the alpha silhouette always receive zero confidence.  The
    interior distance guard prevents nearest-background padding from becoming
    a trusted texture source.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if feather_pixels < 0:
        raise ValueError("feather_pixels must not be negative")
    unit_alpha = alpha_to_unit(alpha)
    inside = unit_alpha >= threshold
    if feather_pixels == 0:
        confidence = inside.astype(np.float64)
    else:
        try:
            from scipy.ndimage import distance_transform_edt
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "alpha edge confidence requires scipy.ndimage"
            ) from error
        distance = distance_transform_edt(inside)
        # Pixel centers on the first interior row are roughly half a pixel from
        # the continuous silhouette boundary.
        confidence = smoothstep01((distance - 0.5) / feather_pixels)
        confidence[~inside] = 0.0
    if multiply_soft_alpha:
        confidence *= unit_alpha
    return confidence.astype(np.float32)


def bilinear_sample(
    image: np.ndarray,
    pixels: np.ndarray,
    *,
    outside_value: float = 0.0,
) -> np.ndarray:
    """Sample a ``H x W`` or ``H x W x C`` image at floating pixel centers."""

    source = np.asarray(image)
    if source.ndim not in (2, 3) or not source.size:
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    coordinates = np.asarray(pixels, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")

    height, width = source.shape[:2]
    channels = () if source.ndim == 2 else (source.shape[2],)
    output = np.full((len(coordinates),) + channels, outside_value, dtype=np.float64)
    finite = np.all(np.isfinite(coordinates), axis=1)
    valid = (
        finite
        & (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] <= width - 1)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= height - 1)
    )
    if not np.any(valid):
        return output

    selected = coordinates[valid]
    x0 = np.floor(selected[:, 0]).astype(np.int64)
    y0 = np.floor(selected[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = selected[:, 0] - x0
    dy = selected[:, 1] - y0

    if source.ndim == 3:
        dx = dx[:, None]
        dy = dy[:, None]
    top = source[y0, x0] * (1.0 - dx) + source[y0, x1] * dx
    bottom = source[y1, x0] * (1.0 - dx) + source[y1, x1] * dx
    output[valid] = top * (1.0 - dy) + bottom * dy
    return output


def nearest_sample(
    image: np.ndarray,
    pixels: np.ndarray,
    *,
    outside_value: float = 0.0,
) -> np.ndarray:
    """Nearest-neighbour sampling for masks and discontinuous depth buffers."""

    source = np.asarray(image)
    if source.ndim not in (2, 3) or not source.size:
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    coordinates = np.asarray(pixels, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")

    height, width = source.shape[:2]
    channels = () if source.ndim == 2 else (source.shape[2],)
    output = np.full((len(coordinates),) + channels, outside_value, dtype=np.float64)
    finite = np.all(np.isfinite(coordinates), axis=1)
    rounded = np.zeros_like(coordinates, dtype=np.int64)
    rounded[finite] = np.rint(coordinates[finite]).astype(np.int64)
    valid = (
        finite
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    output[valid] = source[rounded[valid, 1], rounded[valid, 0]]
    return output


@dataclass(frozen=True)
class DepthBuffer:
    """Near-is-larger orthographic depth with ``-inf`` for uncovered pixels."""

    depth: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.depth, dtype=np.float64)
        if array.ndim != 2 or not array.size:
            raise ValueError("depth must be a non-empty 2-D array")
        if np.any(np.isnan(array)):
            raise ValueError("depth must not contain NaN values")
        object.__setattr__(self, "depth", array)

    @property
    def coverage(self) -> np.ndarray:
        return np.isfinite(self.depth)

    @property
    def image_size(self) -> tuple[int, int]:
        return (int(self.depth.shape[1]), int(self.depth.shape[0]))


def rasterize_depth_buffer(
    vertices: np.ndarray,
    faces: np.ndarray,
    calibration: OrthographicCalibration,
    *,
    cull_backfaces: bool = True,
) -> DepthBuffer:
    """Reference CPU triangle rasterizer for tests and modest resolutions.

    Production Blender baking should generally use a Blender-rendered depth
    pass or ray casting.  This implementation is intentionally straightforward
    and deterministic, making calibration/visibility behaviour reproducible in
    non-Blender tests.
    """

    vertex_array = _points_array(vertices)
    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if face_array.size and (
        int(face_array.min()) < 0 or int(face_array.max()) >= len(vertex_array)
    ):
        raise ValueError("faces contain an out-of-range vertex index")

    projection = calibration.project(vertex_array)
    projected = projection.pixels
    vertex_depth = projection.depth
    width, height = calibration.image_width, calibration.image_height
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    to_camera = calibration.frame.to_camera_array

    for face in face_array:
        triangle_3d = vertex_array[face]
        if cull_backfaces:
            normal = np.cross(
                triangle_3d[1] - triangle_3d[0],
                triangle_3d[2] - triangle_3d[0],
            )
            if float(np.dot(normal, to_camera)) <= 1e-12:
                continue

        triangle = projected[face]
        if not np.all(np.isfinite(triangle)):
            continue
        x_min = max(0, int(np.ceil(float(triangle[:, 0].min()))))
        x_max = min(width - 1, int(np.floor(float(triangle[:, 0].max()))))
        y_min = max(0, int(np.ceil(float(triangle[:, 1].min()))))
        y_max = min(height - 1, int(np.floor(float(triangle[:, 1].max()))))
        if x_min > x_max or y_min > y_max:
            continue

        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) <= 1e-12:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float64),
            np.arange(y_min, y_max + 1, dtype=np.float64),
        )
        weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (weight0 >= -1e-8) & (weight1 >= -1e-8) & (weight2 >= -1e-8)
        if not np.any(inside):
            continue

        candidate = (
            weight0 * vertex_depth[face[0]]
            + weight1 * vertex_depth[face[1]]
            + weight2 * vertex_depth[face[2]]
        )
        candidate[~inside] = -np.inf
        region = depth_buffer[y_min : y_max + 1, x_min : x_max + 1]
        np.maximum(region, candidate, out=region)

    return DepthBuffer(depth_buffer)


def depth_visibility_confidence(
    pixels: np.ndarray,
    sample_depth: np.ndarray,
    depth_buffer: DepthBuffer,
    *,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
    softness: float = 1e-3,
) -> np.ndarray:
    """Compare projected samples against the nearest visible depth.

    A sample at or in front of the reference depth receives confidence one.
    Samples behind it fade exponentially after the configured tolerance.
    """

    if absolute_tolerance < 0 or relative_tolerance < 0 or softness < 0:
        raise ValueError("depth tolerances and softness must not be negative")
    depths = np.asarray(sample_depth, dtype=np.float64)
    if depths.ndim != 1:
        raise ValueError("sample_depth must have shape (N,)")
    coordinates = np.asarray(pixels, dtype=np.float64)
    if coordinates.shape != (len(depths), 2):
        raise ValueError("pixels and sample_depth lengths must match")

    reference = nearest_sample(depth_buffer.depth, coordinates, outside_value=-np.inf)
    covered = np.isfinite(reference) & np.isfinite(depths)
    scale = np.maximum.reduce((np.abs(reference), np.abs(depths), np.ones_like(depths)))
    tolerance = absolute_tolerance + relative_tolerance * scale
    behind = reference - depths - tolerance
    result = np.zeros_like(depths)
    if softness == 0:
        result[covered] = (behind[covered] <= 0.0).astype(np.float64)
    else:
        result[covered] = np.exp(-np.maximum(behind[covered], 0.0) / softness)
    return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class ViewConfidenceConfig:
    minimum_cosine: float = 0.05
    full_confidence_cosine: float = 0.65
    angular_exponent: float = 1.5
    absolute_depth_tolerance: float = 1e-4
    relative_depth_tolerance: float = 1e-4
    depth_softness: float = 1e-3


@dataclass(frozen=True)
class ViewConfidence:
    projection: Projection
    angular: np.ndarray
    silhouette: np.ndarray
    visibility: np.ndarray
    combined: np.ndarray


def compute_view_confidence(
    points: np.ndarray,
    normals: np.ndarray,
    calibration: OrthographicCalibration,
    silhouette_confidence: np.ndarray,
    *,
    depth_buffer: DepthBuffer | None = None,
    config: ViewConfidenceConfig | None = None,
) -> ViewConfidence:
    """Combine view angle, silhouette-edge safety and depth visibility."""

    point_array = _points_array(points)
    normal_array = np.asarray(normals, dtype=np.float64)
    if normal_array.shape != point_array.shape:
        raise ValueError("normals must have the same shape as points")
    confidence_map = np.asarray(silhouette_confidence, dtype=np.float64)
    expected_shape = (calibration.image_height, calibration.image_width)
    if confidence_map.shape != expected_shape:
        raise ValueError(
            f"silhouette confidence must have shape {expected_shape}, "
            f"got {confidence_map.shape}"
        )
    settings = config or ViewConfidenceConfig()
    projection = calibration.project(point_array)
    angular = angular_confidence(
        normal_array,
        calibration.frame,
        minimum_cosine=settings.minimum_cosine,
        full_confidence_cosine=settings.full_confidence_cosine,
        exponent=settings.angular_exponent,
    )
    silhouette = bilinear_sample(confidence_map, projection.pixels, outside_value=0.0)
    silhouette = np.clip(silhouette, 0.0, 1.0)
    if depth_buffer is None:
        visibility = projection.inside_image.astype(np.float64)
    else:
        if depth_buffer.image_size != (
            calibration.image_width,
            calibration.image_height,
        ):
            raise ValueError("depth buffer and calibration image sizes do not match")
        visibility = depth_visibility_confidence(
            projection.pixels,
            projection.depth,
            depth_buffer,
            absolute_tolerance=settings.absolute_depth_tolerance,
            relative_tolerance=settings.relative_depth_tolerance,
            softness=settings.depth_softness,
        )
    combined = angular * silhouette * visibility
    combined[~projection.inside_image] = 0.0
    return ViewConfidence(
        projection=projection,
        angular=angular,
        silhouette=silhouette,
        visibility=visibility,
        combined=combined,
    )


def normalize_view_weights(
    weights: np.ndarray,
    *,
    axis: int = 0,
    exponent: float = 1.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Normalize non-negative confidence weights without inventing fallbacks."""

    if exponent <= 0 or epsilon <= 0:
        raise ValueError("exponent and epsilon must be positive")
    values = np.asarray(weights, dtype=np.float64)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    if exponent != 1.0:
        values = values**exponent
    totals = values.sum(axis=axis, keepdims=True)
    return np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > epsilon,
    )


def blend_view_colors(
    colors: np.ndarray,
    weights: np.ndarray,
    *,
    weight_exponent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend ``(views, samples, channels)`` colors using confidence weights."""

    color_array = np.asarray(colors, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if color_array.ndim != 3:
        raise ValueError("colors must have shape (views, samples, channels)")
    if weight_array.shape != color_array.shape[:2]:
        raise ValueError("weights must have shape (views, samples)")
    finite_color = np.all(np.isfinite(color_array), axis=2)
    guarded_weights = np.where(finite_color, weight_array, 0.0)
    normalized = normalize_view_weights(guarded_weights, exponent=weight_exponent)
    safe_colors = np.where(np.isfinite(color_array), color_array, 0.0)
    blended = np.sum(safe_colors * normalized[..., None], axis=0)
    valid = guarded_weights.sum(axis=0) > 1e-12
    return blended, normalized, valid


@dataclass(frozen=True)
class SurfaceColorDiffusionStats:
    """Summary of one layer-by-layer surface color diffusion pass."""

    initially_known: int
    initially_missing: int
    filled: int
    remaining: int
    iterations_run: int
    max_hop: int
    filled_per_hop: tuple[int, ...]
    usable_edges: int


def diffuse_surface_colors(
    colors: np.ndarray,
    known_mask: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray | None,
    *,
    minimum_normal_dot: float | None = 0.5,
    max_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, SurfaceColorDiffusionStats]:
    """Fill unknown surface colors from the nearest resolved graph layer.

    Each iteration uses only receivers filled by the preceding layer as new
    donors. A receiver is written at most once, so later and more distant
    colors cannot overwrite a nearer result. Components without an initial
    seed remain untouched and are counted in ``stats.remaining``.

    Args:
        colors: ``(N, 3)`` RGB values. Initially known rows must be finite;
            unknown rows may contain placeholders or NaNs.
        known_mask: Boolean seed mask with shape ``(N,)``.
        edges: Undirected mesh edges with shape ``(M, 2)``.
        normals: Optional ``(N, 3)`` surface normals. Pass normals together
            with ``minimum_normal_dot`` to stop propagation across sharp folds.
        minimum_normal_dot: Inclusive dot-product threshold in ``[-1, 1]``.
            Set to ``None`` to disable the normal barrier. If a threshold is
            supplied, normals are required and zero-length normals reject their
            incident edges.
        max_iterations: Maximum graph hops from the nearest seed.

    Returns:
        ``(result_colors, filled_mask, stats)``. ``filled_mask`` marks only
        vertices that were unknown initially and received a propagated color.
    """

    color_array = np.asarray(colors)
    if color_array.ndim != 2 or color_array.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3)")
    result = color_array.astype(np.float64, copy=True)
    vertex_count = len(result)

    seeds = np.asarray(known_mask, dtype=bool)
    if seeds.shape != (vertex_count,):
        raise ValueError("known_mask must have shape (N,)")
    if not np.all(np.isfinite(result[seeds])):
        raise ValueError("initially known colors must contain only finite values")
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")

    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (M, 2)")
    if edge_array.size and (
        int(edge_array.min()) < 0 or int(edge_array.max()) >= vertex_count
    ):
        raise ValueError("edges contain an out-of-range vertex index")
    if len(edge_array):
        # Treat the graph as undirected and prevent duplicate input edges from
        # biasing the donor average.
        edge_array = np.sort(edge_array, axis=1)
        edge_array = edge_array[edge_array[:, 0] != edge_array[:, 1]]
        edge_array = np.unique(edge_array, axis=0)

    if minimum_normal_dot is None:
        usable_edges = np.ones(len(edge_array), dtype=bool)
    else:
        if not -1.0 <= minimum_normal_dot <= 1.0:
            raise ValueError("minimum_normal_dot must be between -1 and 1")
        if normals is None:
            raise ValueError("normals are required when minimum_normal_dot is enabled")
        unit_normals, valid_normals = _unit_rows(normals)
        if len(unit_normals) != vertex_count:
            raise ValueError("normals must have shape (N, 3)")
        if len(edge_array):
            left, right = edge_array[:, 0], edge_array[:, 1]
            normal_dot = np.einsum("ij,ij->i", unit_normals[left], unit_normals[right])
            usable_edges = (
                valid_normals[left]
                & valid_normals[right]
                & (normal_dot >= minimum_normal_dot)
            )
        else:
            usable_edges = np.zeros(0, dtype=bool)

    active_edges = edge_array[usable_edges]
    resolved = seeds.copy()
    frontier = seeds.copy()
    filled_mask = np.zeros(vertex_count, dtype=bool)
    filled_per_hop: list[int] = []

    for _hop in range(1, max_iterations + 1):
        if not np.any(frontier) or not len(active_edges):
            break
        accumulator = np.zeros((vertex_count, 3), dtype=np.float64)
        donor_count = np.zeros(vertex_count, dtype=np.int32)
        left, right = active_edges[:, 0], active_edges[:, 1]

        left_to_right = frontier[left] & ~resolved[right]
        if np.any(left_to_right):
            receivers = right[left_to_right]
            np.add.at(accumulator, receivers, result[left[left_to_right]])
            np.add.at(donor_count, receivers, 1)

        right_to_left = frontier[right] & ~resolved[left]
        if np.any(right_to_left):
            receivers = left[right_to_left]
            np.add.at(accumulator, receivers, result[right[right_to_left]])
            np.add.at(donor_count, receivers, 1)

        receivers = (~resolved) & (donor_count > 0)
        receiver_count = int(np.count_nonzero(receivers))
        if not receiver_count:
            break
        result[receivers] = accumulator[receivers] / donor_count[receivers, None]
        resolved[receivers] = True
        filled_mask[receivers] = True
        frontier = receivers
        filled_per_hop.append(receiver_count)

    filled = int(np.count_nonzero(filled_mask))
    initially_known = int(np.count_nonzero(seeds))
    remaining = int(vertex_count - initially_known - filled)
    stats = SurfaceColorDiffusionStats(
        initially_known=initially_known,
        initially_missing=vertex_count - initially_known,
        filled=filled,
        remaining=remaining,
        iterations_run=len(filled_per_hop),
        max_hop=len(filled_per_hop),
        filled_per_hop=tuple(filled_per_hop),
        usable_edges=int(np.count_nonzero(usable_edges)),
    )
    return result, filled_mask, stats
