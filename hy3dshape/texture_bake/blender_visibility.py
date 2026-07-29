"""Small Blender adapter for orthographic ray-cast visibility.

The module intentionally avoids importing ``bpy`` at import time.  It can
therefore be imported by normal Python tests, while the public function accepts
the standard ``bpy.types.Scene.ray_cast`` interface inside Blender.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class BlenderRaycastVisibility:
    visible: np.ndarray
    confidence: np.ndarray
    hit_error: np.ndarray
    object_match: np.ndarray


def _blender_vector(values: np.ndarray) -> Any:
    """Use mathutils.Vector in Blender and a tuple in ordinary Python."""

    try:
        from mathutils import Vector
    except ModuleNotFoundError:
        return tuple(float(value) for value in values)
    return Vector(tuple(float(value) for value in values))


def _same_object(hit_object: Any, expected_object: Any | None) -> bool:
    if expected_object is None:
        return True
    if hit_object is expected_object:
        return True
    return getattr(hit_object, "original", None) is expected_object


def blender_raycast_visibility(
    scene: Any,
    depsgraph: Any,
    world_points: np.ndarray,
    to_camera: Sequence[float],
    *,
    ray_distance: float,
    hit_tolerance: float = 1e-3,
    softness: float = 1e-3,
    expected_object: Any | None = None,
) -> BlenderRaycastVisibility:
    """Ray-cast parallel orthographic rays from the camera side of each point.

    ``world_points`` must lie on the evaluated surface.  A point is visible
    when the first ray hit belongs to ``expected_object`` (when supplied) and
    lands within ``hit_tolerance`` of that point.
    """

    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("world_points must contain only finite values")
    direction = np.asarray(to_camera, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("to_camera must contain three finite values")
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-12:
        raise ValueError("to_camera must not be zero")
    direction /= direction_length
    if ray_distance <= 0 or hit_tolerance < 0 or softness < 0:
        raise ValueError(
            "ray_distance must be positive; tolerance and softness non-negative"
        )

    visible = np.zeros(len(points), dtype=bool)
    confidence = np.zeros(len(points), dtype=np.float64)
    hit_error = np.full(len(points), np.inf, dtype=np.float64)
    object_match = np.zeros(len(points), dtype=bool)
    ray_direction = -direction

    for index, point in enumerate(points):
        origin = point + direction * ray_distance
        result = scene.ray_cast(
            depsgraph,
            _blender_vector(origin),
            _blender_vector(ray_direction),
            distance=ray_distance + hit_tolerance,
        )
        if not result or not bool(result[0]):
            continue
        location = np.asarray(result[1], dtype=np.float64)
        if location.shape != (3,):
            location = np.asarray(tuple(result[1]), dtype=np.float64)
        error = float(np.linalg.norm(location - point))
        matches = _same_object(result[4], expected_object)
        hit_error[index] = error
        object_match[index] = matches
        visible[index] = matches and error <= hit_tolerance
        if not matches:
            continue
        excess = max(0.0, error - hit_tolerance)
        if softness == 0:
            confidence[index] = float(excess == 0.0)
        else:
            confidence[index] = float(np.exp(-excess / softness))

    return BlenderRaycastVisibility(
        visible=visible,
        confidence=confidence,
        hit_error=hit_error,
        object_match=object_match,
    )
