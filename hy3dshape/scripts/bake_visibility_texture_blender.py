"""Bake four canonical images onto a mesh with visibility-aware blending.

This Blender worker deliberately lives beside the legacy planar projector
instead of replacing it.  It:

1. keeps the selected geometry unchanged;
2. creates one projection UV layer for each canonical view;
3. ray-tests vertices against the mesh to reject occluded views;
4. stores normalized per-corner view weights;
5. bakes the blended result to a fresh, non-overlapping UV atlas; and
6. exports a new GLB without touching the source asset.

Run with Blender's Python:

  blender --background --python bake_visibility_texture_blender.py -- ...
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hy3dshape.texture_bake.hair_topology import (  # noqa: E402
    expand_enclosed_polygon_mask,
)
from hy3dshape.texture_bake.arm_corner_repair import (  # noqa: E402
    apply_arm_repair_plan_to_corners,
    build_arm_repair_plan,
)
from hy3dshape.texture_bake.projection_ownership import (  # noqa: E402
    apply_side_garment_ownership_guard,
)
from hy3dshape.texture_bake.semantic_prior import (  # noqa: E402
    apply_surface_semantic_prior,
)
from hy3dshape.texture_bake.visibility import diffuse_surface_colors  # noqa: E402


VIEW_ORDER = ("front", "left", "back", "right")
VIEW_DIRECTIONS = {
    "front": np.array((0.0, -1.0, 0.0), dtype=np.float32),
    "left": np.array((1.0, 0.0, 0.0), dtype=np.float32),
    "back": np.array((0.0, 1.0, 0.0), dtype=np.float32),
    "right": np.array((-1.0, 0.0, 0.0), dtype=np.float32),
}
VIEW_CHANNEL = {view: index for index, view in enumerate(VIEW_ORDER)}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--hair-mask", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--texture-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--bake-margin", type=int, default=16)
    parser.add_argument("--normal-exponent", type=float, default=4.0)
    parser.add_argument("--minimum-facing", type=float, default=0.02)
    parser.add_argument("--ray-epsilon-scale", type=float, default=2.0e-5)
    parser.add_argument("--silhouette-inset", type=int, default=3)
    parser.add_argument("--silhouette-feather", type=int, default=5)
    parser.add_argument("--propagation-iterations", type=int, default=48)
    parser.add_argument("--head-bottom", type=float, default=0.47)
    parser.add_argument("--hair-inference-bottom", type=float, default=0.62)
    parser.add_argument("--hair-evidence-threshold", type=float, default=0.68)
    parser.add_argument("--hair-topology-fill-iterations", type=int, default=3)
    parser.add_argument("--semantic-prior-iterations", type=int, default=64)
    parser.add_argument("--semantic-prior-normal-dot", type=float, default=0.55)
    parser.add_argument("--semantic-hard-reject", type=float, default=0.45)
    parser.add_argument("--arm-lock-width-fraction", type=float, default=0.30)
    parser.add_argument("--arm-lock-bottom-fraction", type=float, default=0.36)
    parser.add_argument("--arm-lock-top-fraction", type=float, default=0.80)
    parser.add_argument("--enable-arm-view-lock", action="store_true")
    parser.add_argument("--arm-palette-width-start", type=float, default=0.36)
    parser.add_argument("--arm-palette-width-full", type=float, default=0.43)
    parser.add_argument("--arm-palette-bottom", type=float, default=0.44)
    parser.add_argument("--arm-palette-wrist-bottom", type=float, default=0.535)
    parser.add_argument("--arm-palette-wrist-top", type=float, default=0.585)
    parser.add_argument("--arm-palette-top", type=float, default=0.72)
    parser.add_argument("--arm-palette-cool-scale", type=float, default=0.40)
    parser.add_argument("--skip-arm-palette-repair", action="store_true")
    parser.add_argument("--garment-vertical-min", type=float, default=0.44)
    parser.add_argument("--garment-vertical-max", type=float, default=0.585)
    parser.add_argument("--garment-lateral-min", type=float, default=0.10)
    parser.add_argument("--garment-lateral-max", type=float, default=0.32)
    parser.add_argument("--garment-expansion-vertical-min", type=float, default=0.465)
    parser.add_argument("--garment-expansion-lateral-max", type=float, default=0.28)
    parser.add_argument("--garment-anchor-weight", type=float, default=0.03)
    parser.add_argument("--garment-side-weight", type=float, default=0.03)
    parser.add_argument("--garment-side-ratio", type=float, default=1.5)
    parser.add_argument("--garment-expansion-hops", type=int, default=3)
    parser.add_argument(
        "--enable-garment-projection-guard",
        action="store_true",
    )
    parser.add_argument("--surface-fill-iterations", type=int, default=32)
    parser.add_argument("--surface-fill-normal-dot", type=float, default=0.45)
    parser.add_argument("--unwrap-angle", type=float, default=66.0)
    parser.add_argument("--unwrap-margin", type=float, default=0.0015)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--skip-visibility", action="store_true")
    parser.add_argument("--allow-occluded-propagation", action="store_true")
    parser.add_argument("--skip-hair-guard", action="store_true")
    parser.add_argument("--skip-semantic-guard", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_joined_mesh(path: Path) -> bpy.types.Object:
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh object was imported from {path}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj.name = "GohanVisibilityBaked"
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.update()
    return obj


def mesh_arrays(
    mesh: bpy.types.Mesh,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    edge_vertices = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    polygon_centers = np.empty(len(mesh.polygons) * 3, dtype=np.float32)
    polygon_normals = np.empty(len(mesh.polygons) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", vertices)
    mesh.vertices.foreach_get("normal", normals)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    mesh.edges.foreach_get("vertices", edge_vertices)
    mesh.polygons.foreach_get("center", polygon_centers)
    mesh.polygons.foreach_get("normal", polygon_normals)
    return (
        vertices.reshape(-1, 3),
        normals.reshape(-1, 3),
        loop_vertices,
        edge_vertices.reshape(-1, 2),
        polygon_centers.reshape(-1, 3),
        polygon_normals.reshape(-1, 3),
    )


def bounds(vertices: np.ndarray) -> dict[str, float]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return {
        "x_min": float(minimum[0]),
        "x_max": float(maximum[0]),
        "y_min": float(minimum[1]),
        "y_max": float(maximum[1]),
        "z_min": float(minimum[2]),
        "z_max": float(maximum[2]),
    }


def projected_coordinates(
    view: str,
    coordinates: np.ndarray,
    limits: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    x_span = max(limits["x_max"] - limits["x_min"], 1.0e-8)
    y_span = max(limits["y_max"] - limits["y_min"], 1.0e-8)
    z_span = max(limits["z_max"] - limits["z_min"], 1.0e-8)
    vertical = np.clip(
        (coordinates[:, 2] - limits["z_min"]) / z_span, 0.0, 1.0
    )
    if view == "front":
        horizontal = np.clip(
            (coordinates[:, 0] - limits["x_min"]) / x_span, 0.0, 1.0
        )
    elif view == "back":
        horizontal = 1.0 - np.clip(
            (coordinates[:, 0] - limits["x_min"]) / x_span, 0.0, 1.0
        )
    elif view == "left":
        horizontal = np.clip(
            (coordinates[:, 1] - limits["y_min"]) / y_span, 0.0, 1.0
        )
    else:
        horizontal = 1.0 - np.clip(
            (coordinates[:, 1] - limits["y_min"]) / y_span, 0.0, 1.0
        )
    return horizontal, vertical


def atlas_uvs(
    view: str,
    coordinates: np.ndarray,
    limits: dict[str, float],
    metadata: dict[str, object],
) -> np.ndarray:
    horizontal, vertical = projected_coordinates(view, coordinates, limits)
    view_meta = metadata["views"][view]
    tile_x, tile_y = view_meta["tile"]
    x_min, y_min, x_max, y_max = view_meta["bbox"]
    tile_size = float(metadata["tile_size"])
    atlas_size = float(metadata["atlas_size"])
    source_x = x_min + horizontal * (x_max - x_min)
    source_y = y_max - vertical * (y_max - y_min)
    atlas_x = tile_x * tile_size + source_x
    atlas_y = tile_y * tile_size + source_y
    return np.column_stack(
        (atlas_x / atlas_size, 1.0 - atlas_y / atlas_size)
    ).astype(np.float32)


def load_blender_image_array(path: Path) -> tuple[bpy.types.Image, np.ndarray]:
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    width, height = (int(value) for value in image.size)
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return image, pixels.reshape(height, width, 4)


def alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(alpha > 0.5)
    if not len(rows):
        raise ValueError("Input view has an empty alpha mask")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def alpha_interior_confidence(
    alpha: np.ndarray,
    inset_pixels: int,
    feather_pixels: int,
) -> np.ndarray:
    """Reject opaque silhouette outlines before projecting source colors."""

    if inset_pixels < 0 or feather_pixels < 0:
        raise ValueError("silhouette inset and feather must be non-negative")
    unit_alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    current = unit_alpha > 0.5
    if inset_pixels == 0 and feather_pixels == 0:
        return unit_alpha * current.astype(np.float32)

    confidence = np.zeros_like(unit_alpha, dtype=np.float32)
    total_steps = inset_pixels + max(feather_pixels, 1)
    for step in range(1, total_steps + 1):
        eroded = np.zeros_like(current)
        eroded[1:-1, 1:-1] = np.logical_and.reduce(
            (
                current[:-2, :-2],
                current[:-2, 1:-1],
                current[:-2, 2:],
                current[1:-1, :-2],
                current[1:-1, 1:-1],
                current[1:-1, 2:],
                current[2:, :-2],
                current[2:, 1:-1],
                current[2:, 2:],
            )
        )
        current = eroded
        if step <= inset_pixels:
            continue
        value = (
            1.0
            if feather_pixels == 0
            else min((step - inset_pixels) / feather_pixels, 1.0)
        )
        confidence[current] = value
    return confidence * unit_alpha


def load_source_alpha(
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for view in VIEW_ORDER:
        view_metadata = metadata["views"][view]
        alpha_mask = view_metadata.get("alpha_mask")
        path = Path(
            alpha_mask
            if isinstance(alpha_mask, str)
            else view_metadata["source"]
        )
        image, array = load_blender_image_array(path)
        alpha = array[..., 0] if isinstance(alpha_mask, str) else array[..., 3]
        confidence = alpha_interior_confidence(
            alpha,
            args.silhouette_inset,
            args.silhouette_feather,
        )
        result[view] = {
            "path": str(path.resolve()),
            "image": image,
            "alpha": alpha,
            "confidence": confidence,
            "bbox": alpha_bbox(alpha),
            "size": [int(image.size[0]), int(image.size[1])],
        }
    return result


def sample_source_alpha(
    view: str,
    coordinates: np.ndarray,
    limits: dict[str, float],
    source: dict[str, object],
) -> np.ndarray:
    horizontal, vertical = projected_coordinates(view, coordinates, limits)
    x_min, y_min, x_max, y_max = source["bbox"]
    alpha = source["confidence"]
    x = np.rint(x_min + horizontal * (x_max - x_min)).astype(np.int32)
    y = np.rint(y_min + vertical * (y_max - y_min)).astype(np.int32)
    x = np.clip(x, 0, alpha.shape[1] - 1)
    y = np.clip(y, 0, alpha.shape[0] - 1)
    return alpha[y, x].astype(np.float32)


def sample_atlas_channel(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(np.rint(uv[:, 0] * (width - 1)), 0, width - 1).astype(
        np.int32
    )
    y = np.clip(np.rint(uv[:, 1] * (height - 1)), 0, height - 1).astype(
        np.int32
    )
    return image[y, x, 0].astype(np.float32)


def sample_atlas_rgb(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = np.array((width - 1, height - 1), dtype=np.float32)
    pixels = np.rint(uv * scale).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return image[pixels[:, 1], pixels[:, 0], :3].astype(np.float32)


def create_projection_uv_layers(
    mesh: bpy.types.Mesh,
    vertices: np.ndarray,
    loop_vertices: np.ndarray,
    limits: dict[str, float],
    metadata: dict[str, object],
) -> dict[str, np.ndarray]:
    projected: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        uv_per_vertex = atlas_uvs(view, vertices, limits, metadata)
        loop_uv = uv_per_vertex[loop_vertices]
        layer_name = f"Projection_{view.capitalize()}"
        existing = mesh.uv_layers.get(layer_name)
        uv_layer = existing or mesh.uv_layers.new(name=layer_name)
        uv_layer.data.foreach_set("uv", loop_uv.ravel())
        projected[view] = uv_per_vertex
    mesh.update()
    return projected


def configure_cycles(device: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    if device != "GPU":
        return
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return
    preferences = addon.preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            for candidate in preferences.devices:
                candidate.use = candidate.type != "CPU"
            break
        except Exception:
            continue


def ray_visibility(
    obj: bpy.types.Object,
    coordinates: np.ndarray,
    candidate: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
    maximum_distance: float,
) -> np.ndarray:
    visible = np.zeros(len(coordinates), dtype=bool)
    if not np.any(candidate):
        return visible
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(obj, depsgraph, epsilon=0.0)
    ray_direction = Vector(tuple(float(value) for value in direction))
    for index in np.flatnonzero(candidate):
        point = Vector(tuple(float(value) for value in coordinates[index]))
        origin = point + ray_direction * epsilon
        hit, _, _, _ = bvh.ray_cast(origin, ray_direction, maximum_distance)
        visible[index] = hit is None
    return visible


def propagate_missing_weights(
    weights: np.ndarray,
    edges: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, int, int]:
    result = weights.copy()
    known = result.sum(axis=1) > 1.0e-10
    initially_missing = int(np.count_nonzero(~known))
    for _ in range(max(0, iterations)):
        if np.all(known):
            break
        accumulator = np.zeros_like(result)
        count = np.zeros(len(result), dtype=np.float32)
        left, right = edges[:, 0], edges[:, 1]
        right_known = known[right]
        left_known = known[left]
        np.add.at(accumulator, left[right_known], result[right[right_known]])
        np.add.at(count, left[right_known], 1.0)
        np.add.at(accumulator, right[left_known], result[left[left_known]])
        np.add.at(count, right[left_known], 1.0)
        fillable = (~known) & (count > 0.0)
        if not np.any(fillable):
            break
        result[fillable] = accumulator[fillable] / count[fillable, None]
        known[fillable] = True
    remaining = int(np.count_nonzero(~known))
    return result, initially_missing, remaining


def fallback_weights(
    coordinates: np.ndarray,
    normals: np.ndarray,
    limits: dict[str, float],
    source_alpha: dict[str, dict[str, object]],
    exponent: float,
) -> np.ndarray:
    fallback = np.zeros((len(coordinates), 4), dtype=np.float32)
    for view in VIEW_ORDER:
        channel = VIEW_CHANNEL[view]
        facing = np.maximum(normals @ VIEW_DIRECTIONS[view], 0.0)
        alpha = sample_source_alpha(
            view, coordinates, limits, source_alpha[view]
        )
        fallback[:, channel] = np.power(facing, exponent) * alpha

    empty = fallback.sum(axis=1) <= 1.0e-10
    if np.any(empty):
        middle_depth = 0.5 * (limits["y_min"] + limits["y_max"])
        front = empty & (coordinates[:, 1] <= middle_depth)
        back = empty & ~front
        fallback[front, VIEW_CHANNEL["front"]] = 1.0
        fallback[back, VIEW_CHANNEL["back"]] = 1.0
    fallback /= np.maximum(fallback.sum(axis=1, keepdims=True), 1.0e-10)
    return fallback


def compute_vertex_weights(
    obj: bpy.types.Object,
    vertices: np.ndarray,
    normals: np.ndarray,
    edges: np.ndarray,
    limits: dict[str, float],
    source_alpha: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    start = time.perf_counter()
    raw = np.zeros((len(vertices), 4), dtype=np.float32)
    alpha_matrix = np.zeros((len(vertices), 4), dtype=np.float32)
    visibility_matrix = np.zeros((len(vertices), 4), dtype=bool)
    view_stats: dict[str, object] = {}
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    epsilon = diagonal * args.ray_epsilon_scale

    for view in VIEW_ORDER:
        channel = VIEW_CHANNEL[view]
        facing = np.maximum(normals @ VIEW_DIRECTIONS[view], 0.0)
        alpha = sample_source_alpha(view, vertices, limits, source_alpha[view])
        alpha_matrix[:, channel] = alpha
        candidate = (facing >= args.minimum_facing) & (alpha > 0.5)
        if args.skip_visibility:
            visible = candidate
        else:
            visible = ray_visibility(
                obj,
                vertices,
                candidate,
                VIEW_DIRECTIONS[view],
                epsilon,
                diagonal * 2.5,
            )
        visibility_matrix[:, channel] = visible
        raw[:, channel] = (
            np.power(facing, args.normal_exponent)
            * alpha
            * visible.astype(np.float32)
        )
        view_stats[view] = {
            "alpha_candidates": int(np.count_nonzero(candidate)),
            "visible_vertices": int(np.count_nonzero(visible)),
            "mean_facing_visible": (
                float(facing[visible].mean()) if np.any(visible) else 0.0
            ),
        }

    arm_lock_report: dict[str, object] = {"enabled": False}
    if args.enable_arm_view_lock:
        x_mid = 0.5 * (limits["x_min"] + limits["x_max"])
        x_span = max(limits["x_max"] - limits["x_min"], 1.0e-8)
        z_span = max(limits["z_max"] - limits["z_min"], 1.0e-8)
        vertical = (vertices[:, 2] - limits["z_min"]) / z_span
        arm_region = (
            (np.abs(vertices[:, 0] - x_mid)
             >= x_span * args.arm_lock_width_fraction)
            & (vertical >= args.arm_lock_bottom_fraction)
            & (vertical <= args.arm_lock_top_fraction)
        )
        arm_indices = np.flatnonzero(arm_region)
        desired_channels = np.where(
            vertices[arm_indices, 0] >= x_mid,
            VIEW_CHANNEL["left"],
            VIEW_CHANNEL["right"],
        )
        desired_weights = raw[arm_indices, desired_channels].copy()
        raw[arm_indices] = 0.0
        raw[arm_indices, desired_channels] = desired_weights
        retained = desired_weights > 1.0e-10
        arm_lock_report = {
            "enabled": True,
            "region_vertices": int(len(arm_indices)),
            "retained_side_vertices": int(np.count_nonzero(retained)),
            "surface_fill_vertices": int(np.count_nonzero(~retained)),
            "width_fraction": args.arm_lock_width_fraction,
            "bottom_fraction": args.arm_lock_bottom_fraction,
            "top_fraction": args.arm_lock_top_fraction,
        }

    original_trust = np.clip(raw.sum(axis=1), 0.0, 1.0).astype(np.float32)
    strict_occlusion = not args.allow_occluded_propagation
    initially_missing = int(np.count_nonzero(raw.sum(axis=1) <= 1.0e-10))

    if strict_occlusion:
        propagated = raw.copy()
        remaining = initially_missing
        occluded_propagated_entries = 0
    else:
        propagated, initially_missing, remaining = propagate_missing_weights(
            raw, edges, args.propagation_iterations
        )
        occluded_propagated_entries = int(
            np.count_nonzero((propagated > 1.0e-10) & ~visibility_matrix)
        )
        propagated *= alpha_matrix
        fallback = fallback_weights(
            vertices,
            normals,
            limits,
            source_alpha,
            args.normal_exponent,
        )
        legacy_missing = propagated.sum(axis=1) <= 1.0e-10
        propagated[legacy_missing] = fallback[legacy_missing]

    missing = propagated.sum(axis=1) <= 1.0e-10
    row_sums = propagated.sum(axis=1, keepdims=True)
    resolved = ~missing
    propagated[resolved] /= np.maximum(
        row_sums[resolved],
        1.0e-10,
    )
    weights_outside_visibility = int(
        np.count_nonzero((propagated > 1.0e-10) & ~visibility_matrix)
    ) if strict_occlusion else 0
    report = {
        "view_stats": view_stats,
        "arm_view_lock": arm_lock_report,
        "initially_uncovered_vertices": initially_missing,
        "remaining_after_propagation": remaining,
        "fallback_vertices": int(np.count_nonzero(missing)),
        "unresolved_no_visible_view": int(np.count_nonzero(missing)),
        "strict_occlusion": strict_occlusion,
        "occluded_propagated_entries_rejected": occluded_propagated_entries,
        "weights_outside_visibility": weights_outside_visibility,
        "mean_original_trust": float(original_trust.mean()),
        "seconds": round(time.perf_counter() - start, 3),
    }
    return propagated, original_trust, missing, visibility_matrix, report


def compute_polygon_visibility(
    obj: bpy.types.Object,
    centers: np.ndarray,
    normals: np.ndarray,
    limits: dict[str, float],
    source_alpha: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    start = time.perf_counter()
    result = np.zeros((len(centers), 4), dtype=bool)
    view_stats: dict[str, object] = {}
    diagonal = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    epsilon = diagonal * args.ray_epsilon_scale

    for view in VIEW_ORDER:
        channel = VIEW_CHANNEL[view]
        facing = np.maximum(normals @ VIEW_DIRECTIONS[view], 0.0)
        confidence = sample_source_alpha(
            view, centers, limits, source_alpha[view]
        )
        candidate = (
            (facing >= args.minimum_facing)
            & (confidence > 0.5)
        )
        if args.skip_visibility:
            visible = candidate
        else:
            visible = ray_visibility(
                obj,
                centers,
                candidate,
                VIEW_DIRECTIONS[view],
                epsilon,
                diagonal * 2.5,
            )
        result[:, channel] = visible
        view_stats[view] = {
            "alpha_candidates": int(np.count_nonzero(candidate)),
            "visible_polygons": int(np.count_nonzero(visible)),
        }

    report = {
        "view_stats": view_stats,
        "seconds": round(time.perf_counter() - start, 3),
    }
    return result, report


def apply_palette_guards(
    vertices: np.ndarray,
    normals: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    visibility_matrix: np.ndarray,
    limits: dict[str, float],
    metadata: dict[str, object],
    atlas: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    sampled = np.empty((4, len(vertices), 3), dtype=np.float32)
    for view in VIEW_ORDER:
        channel = VIEW_CHANNEL[view]
        uv = atlas_uvs(view, vertices, limits, metadata)
        sampled[channel] = sample_atlas_rgb(atlas, uv)

    if not args.enable_garment_projection_guard:
        guarded_input_weights = weights.copy()
        garment_guard_report: dict[str, object] = {"enabled": False}
    else:
        (
            guarded_input_weights,
            _garment_repair_vertices,
            garment_guard_stats,
        ) = apply_side_garment_ownership_guard(
            vertices,
            sampled,
            weights,
            edges,
            vertical_min_fraction=args.garment_vertical_min,
            vertical_max_fraction=args.garment_vertical_max,
            lateral_min_fraction=args.garment_lateral_min,
            lateral_max_fraction=args.garment_lateral_max,
            expansion_vertical_min_fraction=args.garment_expansion_vertical_min,
            expansion_lateral_max_fraction=args.garment_expansion_lateral_max,
            minimum_anchor_weight=args.garment_anchor_weight,
            minimum_side_weight=args.garment_side_weight,
            minimum_side_ratio=args.garment_side_ratio,
            expansion_hops=args.garment_expansion_hops,
            front_channel=VIEW_CHANNEL["front"],
            left_channel=VIEW_CHANNEL["left"],
            back_channel=VIEW_CHANNEL["back"],
            right_channel=VIEW_CHANNEL["right"],
        )
        garment_guard_report = {
            "enabled": True,
            **asdict(garment_guard_stats),
        }

    if args.skip_semantic_guard:
        adjusted = guarded_input_weights
        surface_edges = edges
        semantic_report: dict[str, object] = {"enabled": False}
        surface_labels = np.zeros(len(vertices), dtype=np.int16)
    else:
        prior = apply_surface_semantic_prior(
            sampled,
            guarded_input_weights,
            edges,
            normals,
            prior_iterations=args.semantic_prior_iterations,
            prior_normal_dot=args.semantic_prior_normal_dot,
            hard_reject_threshold=args.semantic_hard_reject,
        )
        adjusted = prior.adjusted_weights
        surface_edges = prior.diffusion_edges
        semantic_report = prior.report
        surface_labels = prior.surface_labels

    semantic_report["garment_projection_guard"] = (
        garment_guard_report
    )
    if not args.allow_occluded_propagation:
        adjusted *= visibility_matrix.astype(np.float32)
    row_sums = adjusted.sum(axis=1, keepdims=True)
    known = row_sums[:, 0] > 1.0e-10
    adjusted[known] /= np.maximum(
        row_sums[known],
        1.0e-10,
    )

    safe_colors = np.sum(
        sampled * adjusted.T[:, :, None],
        axis=0,
    )
    diffusion_input = safe_colors.copy()
    diffusion_input[~known] = np.nan
    fallback_colors, filled_mask, diffusion_stats = diffuse_surface_colors(
        diffusion_input,
        known,
        surface_edges,
        normals,
        minimum_normal_dot=args.surface_fill_normal_dot,
        max_iterations=args.surface_fill_iterations,
    )
    remaining = (~known) & ~filled_mask
    fallback_colors = fallback_colors.astype(np.float32)
    fallback_colors[remaining] = (0.035, 0.04, 0.055)
    fallback_mix = (~known).astype(np.float32)
    if args.skip_arm_palette_repair:
        arm_plan_colors = np.zeros((len(vertices), 3), dtype=np.float32)
        arm_plan_strength = np.zeros(len(vertices), dtype=np.float32)
        semantic_report["arm_palette_repair"] = {"enabled": False}
    else:
        (
            arm_plan_colors,
            arm_plan_strength,
            arm_palette_stats,
        ) = build_arm_repair_plan(
            vertices,
            sampled,
            expected_labels=surface_labels,
            width_start_fraction=args.arm_palette_width_start,
            width_full_fraction=args.arm_palette_width_full,
            vertical_bottom_fraction=args.arm_palette_bottom,
            wrist_bottom_fraction=args.arm_palette_wrist_bottom,
            wrist_top_fraction=args.arm_palette_wrist_top,
            vertical_top_fraction=args.arm_palette_top,
            cool_value_scale=args.arm_palette_cool_scale,
        )
        semantic_report["arm_palette_repair"] = {
            "enabled": True,
            **asdict(arm_palette_stats),
        }
    weights_outside_visibility = int(
        np.count_nonzero((adjusted > 1.0e-10) & ~visibility_matrix)
    )
    semantic_report["surface_color_diffusion"] = asdict(diffusion_stats)
    semantic_report["controlled_neutral_vertices"] = int(
        np.count_nonzero(remaining)
    )
    semantic_report["weights_outside_visibility"] = (
        weights_outside_visibility
    )
    return (
        adjusted,
        fallback_colors,
        fallback_mix,
        arm_plan_colors,
        arm_plan_strength,
        semantic_report,
    )


def polygon_hair_flags(obj: bpy.types.Object) -> np.ndarray:
    hair_materials = {
        index
        for index, material in enumerate(obj.data.materials)
        if material is not None and "hair" in material.name.lower()
    }
    if not hair_materials:
        return np.zeros(len(obj.data.polygons), dtype=bool)
    return np.fromiter(
        (
            polygon.material_index in hair_materials
            for polygon in obj.data.polygons
        ),
        dtype=bool,
        count=len(obj.data.polygons),
    )


def create_corner_weight_attributes(
    obj: bpy.types.Object,
    vertices: np.ndarray,
    loop_vertices: np.ndarray,
    polygon_centers: np.ndarray,
    vertex_weights: np.ndarray,
    vertex_trust: np.ndarray,
    visibility_matrix: np.ndarray,
    polygon_visibility: np.ndarray,
    vertex_fallback_colors: np.ndarray,
    vertex_fallback_mix: np.ndarray,
    arm_plan_colors: np.ndarray,
    arm_plan_strength: np.ndarray,
    limits: dict[str, float],
    metadata: dict[str, object],
    hair_mask: np.ndarray | None,
    hair_flags: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    mesh = obj.data
    polygon_for_loop = np.repeat(
        np.arange(len(mesh.polygons), dtype=np.int32),
        np.fromiter(
            (polygon.loop_total for polygon in mesh.polygons),
            dtype=np.int32,
            count=len(mesh.polygons),
        ),
    )
    corner_weights = vertex_weights[loop_vertices].copy()
    vertex_corner_visibility = visibility_matrix[loop_vertices]
    polygon_corner_visibility = polygon_visibility[polygon_for_loop]
    corner_visibility = (
        vertex_corner_visibility & polygon_corner_visibility
    )
    center_rejected_entries = int(
        np.count_nonzero(
            (corner_weights > 1.0e-10)
            & vertex_corner_visibility
            & ~polygon_corner_visibility
        )
    )
    surface_fallback_colors = vertex_fallback_colors[loop_vertices].copy()
    surface_fallback_mix = vertex_fallback_mix[loop_vertices].copy()
    if not args.allow_occluded_propagation:
        corner_weights *= corner_visibility.astype(np.float32)
    (
        surface_fallback_colors,
        surface_fallback_mix,
        arm_corner_stats,
    ) = apply_arm_repair_plan_to_corners(
        vertices,
        loop_vertices,
        polygon_for_loop,
        visibility_matrix,
        polygon_visibility,
        surface_fallback_colors,
        surface_fallback_mix,
        arm_plan_colors,
        arm_plan_strength,
        left_channel=VIEW_CHANNEL["left"],
        right_channel=VIEW_CHANNEL["right"],
    )

    hair_fallback = np.zeros(len(corner_weights), dtype=np.float32)
    hair_guard_stats: dict[str, object] = {"enabled": False}

    if (
        hair_mask is not None
        and not args.skip_hair_guard
        and np.any(hair_flags)
    ):
        polygon_hair_confidence = np.zeros(
            (len(mesh.polygons), 4), dtype=np.float32
        )
        for view in VIEW_ORDER:
            channel = VIEW_CHANNEL[view]
            uv = atlas_uvs(view, polygon_centers, limits, metadata)
            polygon_hair_confidence[:, channel] = sample_atlas_channel(
                hair_mask, uv
            )

        polygon_view_weights = np.zeros_like(polygon_hair_confidence)
        np.add.at(
            polygon_view_weights,
            polygon_for_loop,
            corner_weights,
        )
        polygon_weight_sum = polygon_view_weights.sum(axis=1, keepdims=True)
        polygon_view_weights /= np.maximum(
            polygon_weight_sum, 1.0e-10
        )
        hair_evidence = np.sum(
            polygon_view_weights * polygon_hair_confidence, axis=1
        )
        inferred_hair = (
            (~hair_flags)
            & (polygon_centers[:, 2] > args.hair_inference_bottom)
            & (hair_evidence >= args.hair_evidence_threshold)
        )
        effective_hair = hair_flags | inferred_hair

        gate = np.ones_like(polygon_hair_confidence)
        gate[effective_hair] = polygon_hair_confidence[effective_hair]
        head_non_hair = (~effective_hair) & (
            polygon_centers[:, 2] > args.head_bottom
        )
        gate[head_non_hair] = 1.0 - polygon_hair_confidence[head_non_hair]
        loop_gate = gate[polygon_for_loop]
        guarded = corner_weights * loop_gate
        guarded_sum = guarded.sum(axis=1)
        usable = guarded_sum > 1.0e-10
        corner_weights[usable] = guarded[usable] / guarded_sum[usable, None]

        # When visibility and semantics disagree, choose any semantically
        # valid source view. Hair absent from all four masks falls back to a
        # controlled near-black value instead of padded white/skin pixels.
        loop_hair = effective_hair[polygon_for_loop]
        loop_head_non_hair = head_non_hair[polygon_for_loop]
        semantic_alternative = loop_gate * corner_visibility
        alternative_sum = semantic_alternative.sum(axis=1)
        semantic_missing = (~usable) & (alternative_sum > 1.0e-10)
        corner_weights[semantic_missing] = (
            semantic_alternative[semantic_missing]
            / alternative_sum[semantic_missing, None]
        )
        # The source character has flat black anime hair. Projecting the
        # bright silhouette outline onto the sides of individual 3D spikes
        # stretches that outline into broad white bands. Keep all polygons
        # that came from the trusted hair material uniformly dark and let
        # the final PBR material provide the three-dimensional highlights.
        solid_hair = loop_hair
        hair_fallback[solid_hair] = 1.0

        unresolved_head = (~usable) & loop_head_non_hair & (
            alternative_sum <= 1.0e-10
        )
        surface_fallback_mix[unresolved_head] = 1.0
        hair_guard_stats = {
            "enabled": True,
            "source_hair_polygons": int(np.count_nonzero(hair_flags)),
            "inferred_hair_polygons": int(np.count_nonzero(inferred_hair)),
            "hair_polygons": int(np.count_nonzero(effective_hair)),
            "head_non_hair_polygons": int(np.count_nonzero(head_non_hair)),
            "guarded_corners": int(np.count_nonzero(usable)),
            "semantic_recovery_corners": int(
                np.count_nonzero(semantic_missing)
            ),
            "solid_hair_corners": int(
                np.count_nonzero(solid_hair)
            ),
            "fallback_corners": int(np.count_nonzero(~usable)),
        }

    if not args.allow_occluded_propagation:
        corner_weights *= corner_visibility.astype(np.float32)
    corner_sums = corner_weights.sum(axis=1, keepdims=True)
    corner_unresolved = corner_sums[:, 0] <= 1.0e-10
    surface_fallback_mix[corner_unresolved] = 1.0
    corner_weights /= np.maximum(
        corner_sums,
        1.0e-10,
    )
    weights_outside_visibility = int(
        np.count_nonzero((corner_weights > 1.0e-10) & ~corner_visibility)
    )
    hair_guard_stats["surface_fallback_corners"] = int(
        np.count_nonzero(surface_fallback_mix > 0.5)
    )
    hair_guard_stats["weights_outside_visibility"] = weights_outside_visibility
    hair_guard_stats["polygon_center_rejected_weight_entries"] = (
        center_rejected_entries
    )
    hair_guard_stats["arm_palette_corner_guard"] = asdict(
        arm_corner_stats
    )
    existing = mesh.color_attributes.get("ViewWeights")
    if existing is not None:
        mesh.color_attributes.remove(existing)
    weight_attribute = mesh.color_attributes.new(
        name="ViewWeights", type="FLOAT_COLOR", domain="CORNER"
    )
    weight_attribute.data.foreach_set("color", corner_weights.ravel())

    corner_trust = vertex_trust[loop_vertices]
    trust_rgba = np.column_stack(
        (corner_trust, corner_trust, corner_trust, np.ones_like(corner_trust))
    ).astype(np.float32)
    existing_trust = mesh.color_attributes.get("BakeTrust")
    if existing_trust is not None:
        mesh.color_attributes.remove(existing_trust)
    trust_attribute = mesh.color_attributes.new(
        name="BakeTrust", type="FLOAT_COLOR", domain="CORNER"
    )
    trust_attribute.data.foreach_set("color", trust_rgba.ravel())

    fallback_rgba = np.column_stack(
        (
            hair_fallback,
            hair_fallback,
            hair_fallback,
            np.ones_like(hair_fallback),
        )
    ).astype(np.float32)
    existing_fallback = mesh.color_attributes.get("HairFallback")
    if existing_fallback is not None:
        mesh.color_attributes.remove(existing_fallback)
    fallback_attribute = mesh.color_attributes.new(
        name="HairFallback", type="FLOAT_COLOR", domain="CORNER"
    )
    fallback_attribute.data.foreach_set("color", fallback_rgba.ravel())

    surface_color_rgba = np.column_stack(
        (
            surface_fallback_colors,
            np.ones(len(surface_fallback_colors), dtype=np.float32),
        )
    ).astype(np.float32)
    existing_surface_color = mesh.color_attributes.get(
        "SurfaceFallbackColor"
    )
    if existing_surface_color is not None:
        mesh.color_attributes.remove(existing_surface_color)
    surface_color_attribute = mesh.color_attributes.new(
        name="SurfaceFallbackColor", type="FLOAT_COLOR", domain="CORNER"
    )
    surface_color_attribute.data.foreach_set(
        "color", surface_color_rgba.ravel()
    )

    surface_mix_rgba = np.column_stack(
        (
            surface_fallback_mix,
            surface_fallback_mix,
            surface_fallback_mix,
            np.ones_like(surface_fallback_mix),
        )
    ).astype(np.float32)
    existing_surface_mix = mesh.color_attributes.get("SurfaceFallbackMix")
    if existing_surface_mix is not None:
        mesh.color_attributes.remove(existing_surface_mix)
    surface_mix_attribute = mesh.color_attributes.new(
        name="SurfaceFallbackMix", type="FLOAT_COLOR", domain="CORNER"
    )
    surface_mix_attribute.data.foreach_set("color", surface_mix_rgba.ravel())
    mesh.update()
    return hair_guard_stats


def create_baked_uv(
    obj: bpy.types.Object,
    angle_degrees: float,
    island_margin: float,
) -> float:
    start = time.perf_counter()
    mesh = obj.data
    existing = mesh.uv_layers.get("BakedUV")
    if existing is not None:
        mesh.uv_layers.remove(existing)
    baked_uv = mesh.uv_layers.new(name="BakedUV")
    mesh.uv_layers.active = baked_uv
    baked_uv.active_render = True

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(angle_degrees),
        island_margin=island_margin,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh.uv_layers.active = mesh.uv_layers["BakedUV"]
    mesh.uv_layers["BakedUV"].active_render = True
    mesh.update()
    return round(time.perf_counter() - start, 3)


def weight_output_nodes(
    nodes: bpy.types.Nodes,
) -> tuple[dict[str, bpy.types.NodeSocket], bpy.types.Node]:
    weights = nodes.new("ShaderNodeVertexColor")
    weights.name = "VisibilityViewWeights"
    weights.layer_name = "ViewWeights"
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.mode = "RGB"
    nodes.id_data.links.new(weights.outputs["Color"], separate.inputs["Color"])
    sockets = {
        "front": separate.outputs["Red"],
        "left": separate.outputs["Green"],
        "back": separate.outputs["Blue"],
        "right": weights.outputs["Alpha"],
    }
    return sockets, weights


def build_bake_material(
    obj: bpy.types.Object,
    atlas_image: bpy.types.Image,
    target_image: bpy.types.Image,
) -> tuple[bpy.types.Material, bpy.types.Node]:
    material = bpy.data.materials.new("VisibilityBakeWorker")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    weight_sockets, _ = weight_output_nodes(nodes)
    weighted_colors: list[bpy.types.NodeSocket] = []

    for view in VIEW_ORDER:
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = f"Projection_{view.capitalize()}"
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = atlas_image
        texture.interpolation = "Linear"
        texture.extension = "CLIP"
        multiply = nodes.new("ShaderNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs["Fac"].default_value = 1.0
        links.new(uv_node.outputs["UV"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], multiply.inputs[1])
        links.new(weight_sockets[view], multiply.inputs[2])
        weighted_colors.append(multiply.outputs["Color"])

    current = weighted_colors[0]
    for next_color in weighted_colors[1:]:
        add = nodes.new("ShaderNodeMixRGB")
        add.blend_type = "ADD"
        add.inputs["Fac"].default_value = 1.0
        links.new(current, add.inputs[1])
        links.new(next_color, add.inputs[2])
        current = add.outputs["Color"]

    surface_color = nodes.new("ShaderNodeVertexColor")
    surface_color.layer_name = "SurfaceFallbackColor"
    surface_mix_attribute = nodes.new("ShaderNodeVertexColor")
    surface_mix_attribute.layer_name = "SurfaceFallbackMix"
    surface_mix = nodes.new("ShaderNodeMixRGB")
    surface_mix.blend_type = "MIX"
    links.new(surface_mix_attribute.outputs["Color"], surface_mix.inputs["Fac"])
    links.new(current, surface_mix.inputs[1])
    links.new(surface_color.outputs["Color"], surface_mix.inputs[2])
    current = surface_mix.outputs["Color"]

    hair_fallback = nodes.new("ShaderNodeVertexColor")
    hair_fallback.layer_name = "HairFallback"
    solid_hair = nodes.new("ShaderNodeMixRGB")
    solid_hair.blend_type = "MIX"
    solid_hair.inputs[2].default_value = (0.004, 0.005, 0.008, 1.0)
    links.new(hair_fallback.outputs["Color"], solid_hair.inputs["Fac"])
    links.new(current, solid_hair.inputs[1])
    links.new(solid_hair.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    target = nodes.new("ShaderNodeTexImage")
    target.name = "VisibilityBakeTarget"
    target.image = target_image
    target.select = True
    nodes.active = target

    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return material, target


def create_target_image(name: str, size: int) -> bpy.types.Image:
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.new(
        name=name,
        width=size,
        height=size,
        alpha=True,
        float_buffer=False,
    )
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    image.colorspace_settings.name = "sRGB"
    return image


def bake_texture(
    obj: bpy.types.Object,
    image: bpy.types.Image,
    destination: Path,
    margin: int,
    device: str,
) -> float:
    start = time.perf_counter()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.device = device
    scene.render.bake.use_clear = True
    scene.render.bake.margin = margin
    scene.render.bake.target = "IMAGE_TEXTURES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(destination.resolve())
    image.file_format = "PNG"
    image.save()
    return round(time.perf_counter() - start, 3)


def build_final_material(
    obj: bpy.types.Object,
    baked_image: bpy.types.Image,
    hair_flags: np.ndarray,
) -> bpy.types.Material:
    material = bpy.data.materials.new("GohanVisibilityBaked4K")
    material.use_nodes = True
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "BakedUV"
    texture.image = baked_image
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    principled.inputs["Roughness"].default_value = 0.82
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.18
    principled.inputs["Emission Strength"].default_value = 0.08
    links.new(uv_map.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(texture.outputs["Color"], principled.inputs["Emission Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    hair_material = bpy.data.materials.new("GohanHairSolid")
    hair_material.use_nodes = True
    hair_material.diffuse_color = (0.004, 0.005, 0.008, 1.0)
    hair_nodes = hair_material.node_tree.nodes
    hair_links = hair_material.node_tree.links
    hair_nodes.clear()
    hair_output = hair_nodes.new("ShaderNodeOutputMaterial")
    hair_principled = hair_nodes.new("ShaderNodeBsdfPrincipled")
    hair_principled.inputs["Base Color"].default_value = (0.004, 0.005, 0.008, 1.0)
    hair_principled.inputs["Roughness"].default_value = 0.88
    hair_principled.inputs["Metallic"].default_value = 0.0
    hair_principled.inputs["Specular IOR Level"].default_value = 0.16
    hair_principled.inputs["Emission Color"].default_value = (0.002, 0.003, 0.006, 1.0)
    hair_principled.inputs["Emission Strength"].default_value = 0.025
    hair_links.new(hair_principled.outputs["BSDF"], hair_output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.data.materials.append(hair_material)
    for index, polygon in enumerate(obj.data.polygons):
        polygon.material_index = 1 if hair_flags[index] else 0
    return material


def remove_worker_attributes(mesh: bpy.types.Mesh) -> None:
    for name in (
        "ViewWeights",
        "BakeTrust",
        "HairFallback",
        "SurfaceFallbackColor",
        "SurfaceFallbackMix",
    ):
        attribute = mesh.color_attributes.get(name)
        if attribute is not None:
            mesh.color_attributes.remove(attribute)
    for view in VIEW_ORDER:
        layer = mesh.uv_layers.get(f"Projection_{view.capitalize()}")
        if layer is not None:
            mesh.uv_layers.remove(layer)
    mesh.uv_layers.active = mesh.uv_layers.get("BakedUV")
    if mesh.uv_layers.get("BakedUV") is not None:
        mesh.uv_layers["BakedUV"].active_render = True
    mesh.update()


def export_glb(obj: bpy.types.Object, destination: Path) -> float:
    start = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(destination.resolve()),
        export_format="GLB",
        use_selection=True,
        export_normals=True,
        export_texcoords=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    return round(time.perf_counter() - start, 3)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    clear_scene()
    configure_cycles(args.device)
    obj = import_joined_mesh(args.mesh)
    mesh = obj.data
    hair_flags = polygon_hair_flags(obj)
    (
        vertices,
        normals,
        loop_vertices,
        edges,
        polygon_centers,
        polygon_normals,
    ) = mesh_arrays(mesh)
    polygon_loop_totals = np.fromiter(
        (polygon.loop_total for polygon in mesh.polygons),
        dtype=np.int32,
        count=len(mesh.polygons),
    )
    hair_flags, hair_topology_stats = expand_enclosed_polygon_mask(
        hair_flags,
        loop_vertices,
        polygon_loop_totals,
        polygon_centers,
        minimum_z=args.hair_inference_bottom,
        max_iterations=args.hair_topology_fill_iterations,
    )
    limits = bounds(vertices)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    source_alpha = load_source_alpha(metadata, args)
    atlas_image, atlas_array = load_blender_image_array(args.atlas)

    hair_mask_path = args.hair_mask
    if hair_mask_path is None:
        value = metadata.get("hair_mask")
        hair_mask_path = Path(value) if isinstance(value, str) else None
    hair_mask = None
    if hair_mask_path is not None and hair_mask_path.is_file():
        _, hair_mask = load_blender_image_array(hair_mask_path)

    create_projection_uv_layers(
        mesh, vertices, loop_vertices, limits, metadata
    )
    (
        vertex_weights,
        vertex_trust,
        unresolved_vertices,
        visibility_matrix,
        weight_report,
    ) = compute_vertex_weights(
        obj,
        vertices,
        normals,
        edges,
        limits,
        source_alpha,
        args,
    )
    (
        polygon_visibility,
        polygon_visibility_report,
    ) = compute_polygon_visibility(
        obj,
        polygon_centers,
        polygon_normals,
        limits,
        source_alpha,
        args,
    )
    (
        vertex_weights,
        vertex_fallback_colors,
        vertex_fallback_mix,
        arm_plan_colors,
        arm_plan_strength,
        semantic_guard_report,
    ) = apply_palette_guards(
        vertices,
        normals,
        edges,
        vertex_weights,
        visibility_matrix,
        limits,
        metadata,
        atlas_array,
        args,
    )
    hair_guard_report = create_corner_weight_attributes(
        obj,
        vertices,
        loop_vertices,
        polygon_centers,
        vertex_weights,
        vertex_trust,
        visibility_matrix,
        polygon_visibility,
        vertex_fallback_colors,
        vertex_fallback_mix,
        arm_plan_colors,
        arm_plan_strength,
        limits,
        metadata,
        hair_mask,
        hair_flags,
        args,
    )
    hair_guard_report["topology_fill"] = asdict(
        hair_topology_stats
    )
    unwrap_seconds = create_baked_uv(
        obj, args.unwrap_angle, args.unwrap_margin
    )
    target_image = create_target_image(
        "GohanVisibilityBakedAtlas", args.texture_size
    )
    build_bake_material(obj, atlas_image, target_image)
    bake_seconds = bake_texture(
        obj,
        target_image,
        args.texture_output,
        args.bake_margin,
        args.device,
    )
    build_final_material(obj, target_image, hair_flags)
    remove_worker_attributes(mesh)
    export_seconds = export_glb(obj, args.output)

    report = {
        "source_mesh": str(args.mesh.resolve()),
        "source_atlas": str(args.atlas.resolve()),
        "metadata": str(args.metadata.resolve()),
        "output_glb": str(args.output.resolve()),
        "output_texture": str(args.texture_output.resolve()),
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "loops": len(mesh.loops),
        "texture_size": args.texture_size,
        "normal_exponent": args.normal_exponent,
        "visibility_enabled": not args.skip_visibility,
        "semantic_guard": semantic_guard_report,
        "hair_guard": hair_guard_report,
        "polygon_visibility": polygon_visibility_report,
        "weights": weight_report,
        "timings": {
            "unwrap_seconds": unwrap_seconds,
            "bake_seconds": bake_seconds,
            "export_seconds": export_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "source_views": {
            view: {
                "path": source_alpha[view]["path"],
                "size": source_alpha[view]["size"],
                "alpha_bbox_blender_origin": source_alpha[view]["bbox"],
            }
            for view in VIEW_ORDER
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
