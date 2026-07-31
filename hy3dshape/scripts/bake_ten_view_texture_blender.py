"""Bake a strict visibility-aware texture from eight yaw and two high views.

Unlike the four-view worker, this script does not encode camera weights in one
RGBA attribute.  It ray-tests all ten cameras, rejects polygon-center
occlusion, applies cross-view color consensus, stores the resulting final
per-corner color, and then bakes that color to a conventional UV texture.

The source mesh is never modified.  Use a candidate output filename while
auditing quality, for example::

  blender --background --factory-startup \
    --python bake_ten_view_texture_blender.py -- \
    --mesh white_mesh.glb --front input_front.png ... \
    --output candidate_ten_view.glb \
    --texture-output candidate_ten_view.png \
    --report candidate_ten_view.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPOSITORY_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from bake_visibility_texture_blender import (  # noqa: E402
    alpha_interior_confidence,
    bake_texture,
    clear_scene,
    configure_cycles,
    create_baked_uv,
    create_target_image,
    export_glb,
    import_joined_mesh,
    mesh_arrays,
)
from hy3dshape.texture_bake.calibration import (  # noqa: E402
    OrthographicCalibration,
    fit_orthographic_from_alpha,
)
from hy3dshape.texture_bake.semantic_prior import (  # noqa: E402
    apply_surface_semantic_prior,
)
from hy3dshape.texture_bake.ten_view_consensus import (  # noqa: E402
    HIGH_VIEW_NAMES,
    HORIZONTAL_VIEW_NAMES,
    TEN_VIEW_ANGLES,
    add_regional_ownership_arguments,
    blend_consensus_colors,
    linear_to_srgb,
    restore_chroma,
    robust_consensus_weights,
    ten_view_frames,
)
from hy3dshape.texture_bake.visibility import (  # noqa: E402
    bilinear_sample,
    diffuse_surface_colors,
)


VIEW_ORDER = tuple(TEN_VIEW_ANGLES)
VIEW_PRIOR = {
    name: (0.68 if name in HIGH_VIEW_NAMES else 1.0)
    for name in VIEW_ORDER
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    for name in VIEW_ORDER:
        dashed = f"--{name.replace('_', '-')}"
        underscored = f"--{name}"
        parser.add_argument(dashed, underscored, dest=name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--texture-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--bake-margin", type=int, default=12)
    parser.add_argument("--normal-exponent", type=float, default=5.0)
    parser.add_argument("--minimum-facing", type=float, default=0.035)
    parser.add_argument("--ray-epsilon-scale", type=float, default=2.0e-5)
    parser.add_argument("--silhouette-inset", type=int, default=3)
    parser.add_argument("--silhouette-feather", type=int, default=6)
    parser.add_argument("--color-sigma", type=float, default=0.28)
    parser.add_argument("--minimum-peer-support", type=float, default=0.20)
    parser.add_argument("--consensus-floor", type=float, default=0.035)
    parser.add_argument("--winner-color-mix", type=float, default=0.0)
    parser.add_argument("--chroma-restore", type=float, default=0.0)
    add_regional_ownership_arguments(parser)
    parser.add_argument("--surface-fill-iterations", type=int, default=56)
    parser.add_argument("--surface-fill-normal-dot", type=float, default=0.70)
    parser.add_argument("--semantic-prior-iterations", type=int, default=28)
    parser.add_argument("--semantic-prior-normal-dot", type=float, default=0.72)
    parser.add_argument("--semantic-hard-reject", type=float, default=0.45)
    parser.add_argument("--unwrap-angle", type=float, default=66.0)
    parser.add_argument("--unwrap-margin", type=float, default=0.0015)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    mesh = args.mesh.resolve(strict=True)
    output = args.output.resolve()
    texture_output = args.texture_output.resolve()
    report = args.report.resolve()
    if mesh.suffix.lower() != ".glb":
        raise ValueError("source mesh must be a GLB")
    if output == mesh:
        raise ValueError("candidate output must not overwrite the source mesh")
    if len({output, texture_output, report}) != 3:
        raise ValueError("output, texture output and report must be different files")
    for name in VIEW_ORDER:
        path = Path(getattr(args, name)).resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(f"{name} image does not exist: {path}")
    if (
        args.texture_size < 256
        or args.bake_margin < 0
        or args.normal_exponent <= 0.0
        or args.minimum_facing < 0.0
        or args.ray_epsilon_scale <= 0.0
        or args.chunk_size < 1
    ):
        raise ValueError("numeric bake arguments are outside their valid range")


def load_rgba_top_down(path: Path) -> np.ndarray:
    """Load Blender pixels and flip them to conventional top-left image rows."""

    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    width, height = (int(value) for value in image.size)
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    result = pixels.reshape(height, width, 4)[::-1].copy()
    bpy.data.images.remove(image)
    return result


def prepare_sources(
    args: argparse.Namespace,
    vertices: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, OrthographicCalibration],
    dict[str, object],
]:
    frames = ten_view_frames()
    images: dict[str, np.ndarray] = {}
    confidence: dict[str, np.ndarray] = {}
    provisional: dict[str, OrthographicCalibration] = {}
    source_report: dict[str, object] = {}

    for name in VIEW_ORDER:
        path = Path(getattr(args, name))
        rgba = load_rgba_top_down(path)
        alpha = rgba[..., 3]
        rows, columns = np.nonzero(alpha >= 0.5)
        if not len(rows):
            raise ValueError(f"{name} image has an empty alpha silhouette")
        images[name] = rgba
        confidence[name] = alpha_interior_confidence(
            alpha,
            args.silhouette_inset,
            args.silhouette_feather,
        )
        provisional[name] = fit_orthographic_from_alpha(
            alpha,
            vertices,
            frames[name],
            fit_mode="height",
        )
        source_report[name] = {
            "path": str(path.resolve()),
            "size": [int(rgba.shape[1]), int(rgba.shape[0])],
            "alpha_bbox_top_left": [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()),
                int(rows.max()),
            ],
        }

    shared_scale = float(
        np.median(
            [
                provisional[name].pixels_per_unit_v
                for name in HORIZONTAL_VIEW_NAMES
            ]
        )
    )
    calibrations = {
        name: fit_orthographic_from_alpha(
            images[name][..., 3],
            vertices,
            frames[name],
            fit_mode="height",
            pixels_per_unit=shared_scale,
        )
        for name in VIEW_ORDER
    }
    source_report["shared_pixels_per_unit"] = shared_scale
    return images, confidence, calibrations, source_report


def ray_visibility(
    bvh: BVHTree,
    coordinates: np.ndarray,
    candidate: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
    maximum_distance: float,
) -> np.ndarray:
    visible = np.zeros(len(coordinates), dtype=bool)
    if not np.any(candidate):
        return visible
    ray_direction = Vector(tuple(float(value) for value in direction))
    for index in np.flatnonzero(candidate):
        point = Vector(tuple(float(value) for value in coordinates[index]))
        origin = point + ray_direction * epsilon
        hit, _, _, _ = bvh.ray_cast(origin, ray_direction, maximum_distance)
        visible[index] = hit is None
    return visible


def apply_regional_ownership(
    vertices: np.ndarray,
    normals: np.ndarray,
    polygon_centers: np.ndarray,
    polygon_normals: np.ndarray,
    vertex_weights: np.ndarray,
    polygon_weights: np.ndarray,
    calibrations: dict[str, OrthographicCalibration],
) -> dict[str, int | float]:
    """Constrain high cameras and opposite-side arm projections."""

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0e-8)
    vertex_height = (vertices[:, 2] - minimum[2]) / span[2]
    polygon_height = (polygon_centers[:, 2] - minimum[2]) / span[2]
    high_channels = [VIEW_ORDER.index(name) for name in HIGH_VIEW_NAMES]
    high_vertex_allowed = (
        (vertex_height >= 0.58)
        & (normals[:, 2] >= -0.10)
    )
    high_polygon_allowed = (
        (polygon_height >= 0.58)
        & (polygon_normals[:, 2] >= -0.10)
    )
    high_vertex_rejected = 0
    high_polygon_rejected = 0
    for channel in high_channels:
        high_vertex_rejected += int(
            np.count_nonzero(
                (vertex_weights[:, channel] > 1.0e-12)
                & ~high_vertex_allowed
            )
        )
        high_polygon_rejected += int(
            np.count_nonzero(
                (polygon_weights[:, channel] > 1.0e-12)
                & ~high_polygon_allowed
            )
        )
        vertex_weights[~high_vertex_allowed, channel] = 0.0
        polygon_weights[~high_polygon_allowed, channel] = 0.0

    x_mid = 0.5 * (minimum[0] + maximum[0])
    arm_vertex = (
        (np.abs(vertices[:, 0] - x_mid) >= 0.30 * span[0])
        & (vertex_height >= 0.41)
        & (vertex_height <= 0.72)
    )
    arm_polygon = (
        (np.abs(polygon_centers[:, 0] - x_mid) >= 0.30 * span[0])
        & (polygon_height >= 0.41)
        & (polygon_height <= 0.72)
    )
    arm_entries_reduced = 0
    for channel, name in enumerate(VIEW_ORDER):
        camera_x = calibrations[name].frame.to_camera_array[0]
        vertex_same_side = (
            np.sign(vertices[:, 0] - x_mid) * camera_x >= 0.20
        )
        polygon_same_side = (
            np.sign(polygon_centers[:, 0] - x_mid) * camera_x >= 0.20
        )
        vertex_reduce = arm_vertex & ~vertex_same_side
        polygon_reduce = arm_polygon & ~polygon_same_side
        arm_entries_reduced += int(
            np.count_nonzero(
                (vertex_weights[:, channel] > 1.0e-12) & vertex_reduce
            )
        )
        vertex_weights[vertex_reduce, channel] *= 0.06
        polygon_weights[polygon_reduce, channel] *= 0.06
    return {
        "enabled": True,
        "high_vertex_entries_rejected": high_vertex_rejected,
        "high_polygon_entries_rejected": high_polygon_rejected,
        "outer_arm_vertices": int(np.count_nonzero(arm_vertex)),
        "outer_arm_polygons": int(np.count_nonzero(arm_polygon)),
        "outer_arm_entries_reduced": arm_entries_reduced,
        "opposite_side_retention": 0.06,
    }


def sample_views(
    obj: bpy.types.Object,
    vertices: np.ndarray,
    normals: np.ndarray,
    polygon_centers: np.ndarray,
    polygon_normals: np.ndarray,
    images: dict[str, np.ndarray],
    confidence_maps: dict[str, np.ndarray],
    calibrations: dict[str, OrthographicCalibration],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    vertex_colors = np.zeros(
        (len(vertices), len(VIEW_ORDER), 3),
        dtype=np.float32,
    )
    vertex_weights = np.zeros(
        (len(vertices), len(VIEW_ORDER)),
        dtype=np.float32,
    )
    polygon_weights = np.zeros(
        (len(polygon_centers), len(VIEW_ORDER)),
        dtype=np.float32,
    )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(obj, depsgraph, epsilon=0.0)
    diagonal = float(
        np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    )
    epsilon = diagonal * args.ray_epsilon_scale
    maximum_distance = diagonal * 2.5
    view_report: dict[str, object] = {}

    for channel, name in enumerate(VIEW_ORDER):
        calibration = calibrations[name]
        direction = calibration.frame.to_camera_array
        vertex_projection = calibration.project(vertices)
        vertex_alpha = bilinear_sample(
            confidence_maps[name],
            vertex_projection.pixels,
            outside_value=0.0,
        )
        vertex_facing = np.maximum(normals @ direction, 0.0)
        vertex_candidate = (
            vertex_projection.inside_image
            & (vertex_facing >= args.minimum_facing)
            & (vertex_alpha > 0.02)
        )
        vertex_visible = ray_visibility(
            bvh,
            vertices,
            vertex_candidate,
            direction,
            epsilon,
            maximum_distance,
        )
        vertex_weight = (
            np.power(vertex_facing, args.normal_exponent)
            * vertex_alpha
            * vertex_visible.astype(np.float64)
            * VIEW_PRIOR[name]
        )
        sampled = bilinear_sample(
            images[name][..., :3],
            vertex_projection.pixels,
            outside_value=0.0,
        )
        vertex_colors[:, channel] = sampled.astype(np.float32)
        vertex_weights[:, channel] = vertex_weight.astype(np.float32)

        polygon_projection = calibration.project(polygon_centers)
        polygon_alpha = bilinear_sample(
            confidence_maps[name],
            polygon_projection.pixels,
            outside_value=0.0,
        )
        polygon_facing = np.maximum(polygon_normals @ direction, 0.0)
        polygon_candidate = (
            polygon_projection.inside_image
            & (polygon_facing >= args.minimum_facing)
            & (polygon_alpha > 0.02)
        )
        polygon_visible = ray_visibility(
            bvh,
            polygon_centers,
            polygon_candidate,
            direction,
            epsilon,
            maximum_distance,
        )
        polygon_weights[:, channel] = (
            np.sqrt(np.maximum(polygon_facing, 0.0))
            * polygon_alpha
            * polygon_visible.astype(np.float64)
        ).astype(np.float32)
        view_report[name] = {
            "vertex_alpha_candidates": int(np.count_nonzero(vertex_candidate)),
            "visible_vertices": int(np.count_nonzero(vertex_visible)),
            "visible_polygons": int(np.count_nonzero(polygon_visible)),
            "mean_facing_visible": (
                float(vertex_facing[vertex_visible].mean())
                if np.any(vertex_visible)
                else 0.0
            ),
            "prior": VIEW_PRIOR[name],
        }

    if args.enable_regional_ownership:
        view_report["regional_ownership"] = apply_regional_ownership(
            vertices,
            normals,
            polygon_centers,
            polygon_normals,
            vertex_weights,
            polygon_weights,
            calibrations,
        )
    else:
        view_report["regional_ownership"] = {"enabled": False}
    return vertex_colors, vertex_weights, polygon_weights, view_report


def consensus_options(args: argparse.Namespace) -> dict[str, float]:
    return {
        "color_sigma": args.color_sigma,
        "minimum_peer_support": args.minimum_peer_support,
        "floor": args.consensus_floor,
    }


def aggregate_consensus_reports(
    reports: list[dict[str, float | int]],
) -> dict[str, float | int]:
    if not reports:
        return {}
    sample_count = sum(int(report["samples"]) for report in reports)
    result: dict[str, float | int] = {}
    for key in (
        "samples",
        "single_view_samples",
        "multi_view_samples",
        "unresolved_samples",
        "strongly_suppressed_entries",
    ):
        result[key] = sum(int(report[key]) for report in reports)
    result["views"] = int(reports[0]["views"])
    result["mean_usable_views"] = (
        sum(
            float(report["mean_usable_views"]) * int(report["samples"])
            for report in reports
        )
        / max(sample_count, 1)
    )
    for key in ("color_sigma", "minimum_peer_support", "floor"):
        result[key] = float(reports[0][key])
    return result


def build_corner_colors(
    mesh: bpy.types.Mesh,
    vertices: np.ndarray,
    normals: np.ndarray,
    loop_vertices: np.ndarray,
    edges: np.ndarray,
    vertex_colors: np.ndarray,
    vertex_weights: np.ndarray,
    polygon_weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    options = consensus_options(args)
    semantic_prior = apply_surface_semantic_prior(
        np.moveaxis(linear_to_srgb(vertex_colors), 1, 0),
        vertex_weights,
        edges,
        normals,
        prior_iterations=args.semantic_prior_iterations,
        prior_normal_dot=args.semantic_prior_normal_dot,
        hard_reject_threshold=args.semantic_hard_reject,
    )
    guarded_vertex_weights = semantic_prior.adjusted_weights
    normalized_vertex, vertex_consensus_report = robust_consensus_weights(
        vertex_colors,
        guarded_vertex_weights,
        **options,
    )
    blended_vertex, vertex_valid = blend_consensus_colors(
        vertex_colors,
        normalized_vertex,
        winner_mix=args.winner_color_mix,
    )
    (
        diffused_vertex,
        filled_vertex,
        diffusion_report,
    ) = diffuse_surface_colors(
        blended_vertex,
        vertex_valid,
        semantic_prior.diffusion_edges,
        normals,
        minimum_normal_dot=args.surface_fill_normal_dot,
        max_iterations=args.surface_fill_iterations,
    )
    resolved_vertex = vertex_valid | filled_vertex
    if np.any(vertex_valid):
        neutral = np.median(blended_vertex[vertex_valid], axis=0)
    else:
        neutral = np.asarray((0.18, 0.18, 0.18), dtype=np.float32)
    diffused_vertex[~resolved_vertex] = neutral

    polygon_for_loop = np.repeat(
        np.arange(len(mesh.polygons), dtype=np.int32),
        np.fromiter(
            (polygon.loop_total for polygon in mesh.polygons),
            dtype=np.int32,
            count=len(mesh.polygons),
        ),
    )
    corner_colors = np.empty((len(loop_vertices), 3), dtype=np.float32)
    chunk_reports: list[dict[str, float | int]] = []
    strict_resolved = 0
    for start in range(0, len(loop_vertices), args.chunk_size):
        stop = min(start + args.chunk_size, len(loop_vertices))
        vertex_index = loop_vertices[start:stop]
        polygon_index = polygon_for_loop[start:stop]
        raw = (
            guarded_vertex_weights[vertex_index]
            * polygon_weights[polygon_index]
        )
        normalized, report = robust_consensus_weights(
            vertex_colors[vertex_index],
            raw,
            **options,
        )
        blended, valid = blend_consensus_colors(
            vertex_colors[vertex_index],
            normalized,
            winner_mix=args.winner_color_mix,
        )
        blended[~valid] = diffused_vertex[vertex_index[~valid]]
        corner_colors[start:stop] = blended
        strict_resolved += int(np.count_nonzero(valid))
        chunk_reports.append(report)

    corner_colors = restore_chroma(corner_colors, amount=args.chroma_restore)
    return np.clip(corner_colors, 0.0, 1.0), {
        "semantic_prior": semantic_prior.report,
        "vertex_consensus": vertex_consensus_report,
        "corner_consensus": aggregate_consensus_reports(chunk_reports),
        "strict_resolved_corners": strict_resolved,
        "surface_fallback_corners": int(len(loop_vertices) - strict_resolved),
        "diffusion": asdict(diffusion_report),
        "remaining_neutral_vertices": int(np.count_nonzero(~resolved_vertex)),
    }


def install_projected_color(
    mesh: bpy.types.Mesh,
    corner_colors: np.ndarray,
) -> str:
    name = "TenViewProjectedColor"
    existing = mesh.color_attributes.get(name)
    if existing is not None:
        mesh.color_attributes.remove(existing)
    attribute = mesh.color_attributes.new(
        name=name,
        type="FLOAT_COLOR",
        domain="CORNER",
    )
    rgba = np.column_stack(
        (
            corner_colors,
            np.ones(len(corner_colors), dtype=np.float32),
        )
    ).astype(np.float32)
    attribute.data.foreach_set("color", rgba.ravel())
    mesh.update()
    return name


def build_color_bake_material(
    obj: bpy.types.Object,
    color_attribute: str,
    target_image: bpy.types.Image,
) -> bpy.types.Material:
    material = bpy.data.materials.new("TenViewColorBakeWorker")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = color_attribute
    target = nodes.new("ShaderNodeTexImage")
    target.name = "TenViewBakeTarget"
    target.image = target_image
    target.select = True
    nodes.active = target
    links.new(vertex_color.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return material


def build_final_material(
    obj: bpy.types.Object,
    baked_image: bpy.types.Image,
) -> bpy.types.Material:
    material = bpy.data.materials.new("TenViewVisibilityBaked")
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
    principled.inputs["Emission Strength"].default_value = 0.025
    links.new(uv_map.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(texture.outputs["Color"], principled.inputs["Emission Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    return material


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    clear_scene()
    configure_cycles(args.device)
    obj = import_joined_mesh(args.mesh)
    obj.name = "TenViewVisibilityBaked"
    mesh = obj.data
    (
        vertices,
        normals,
        loop_vertices,
        edges,
        polygon_centers,
        polygon_normals,
    ) = mesh_arrays(mesh)
    images, confidence, calibrations, source_report = prepare_sources(
        args,
        vertices,
    )
    sample_started = time.perf_counter()
    (
        vertex_colors,
        vertex_weights,
        polygon_weights,
        view_report,
    ) = sample_views(
        obj,
        vertices,
        normals,
        polygon_centers,
        polygon_normals,
        images,
        confidence,
        calibrations,
        args,
    )
    sample_seconds = round(time.perf_counter() - sample_started, 3)
    consensus_started = time.perf_counter()
    corner_colors, consensus_report = build_corner_colors(
        mesh,
        vertices,
        normals,
        loop_vertices,
        edges,
        vertex_colors,
        vertex_weights,
        polygon_weights,
        args,
    )
    consensus_seconds = round(time.perf_counter() - consensus_started, 3)
    color_attribute = install_projected_color(mesh, corner_colors)
    unwrap_seconds = create_baked_uv(
        obj,
        args.unwrap_angle,
        args.unwrap_margin,
    )
    target_image = create_target_image(
        "TenViewVisibilityAtlas",
        args.texture_size,
    )
    build_color_bake_material(obj, color_attribute, target_image)
    bake_seconds = bake_texture(
        obj,
        target_image,
        args.texture_output,
        args.bake_margin,
        args.device,
    )
    build_final_material(obj, target_image)
    attribute = mesh.color_attributes.get(color_attribute)
    if attribute is not None:
        mesh.color_attributes.remove(attribute)
    mesh.uv_layers.active = mesh.uv_layers.get("BakedUV")
    if mesh.uv_layers.get("BakedUV") is not None:
        mesh.uv_layers["BakedUV"].active_render = True
    mesh.update()
    export_seconds = export_glb(obj, args.output)
    report = {
        "method": "strict-visibility-ten-view-consensus",
        "source_mesh": str(args.mesh.resolve()),
        "output_glb": str(args.output.resolve()),
        "output_texture": str(args.texture_output.resolve()),
        "vertices": len(vertices),
        "polygons": len(polygon_centers),
        "loops": len(loop_vertices),
        "views": list(VIEW_ORDER),
        "view_stats": view_report,
        "consensus": consensus_report,
        "calibrations": {
            name: calibrations[name].to_dict()
            for name in VIEW_ORDER
        },
        "sources": source_report,
        "settings": {
            "texture_size": args.texture_size,
            "normal_exponent": args.normal_exponent,
            "minimum_facing": args.minimum_facing,
            "silhouette_inset": args.silhouette_inset,
            "silhouette_feather": args.silhouette_feather,
            "color_sigma": args.color_sigma,
            "minimum_peer_support": args.minimum_peer_support,
            "consensus_floor": args.consensus_floor,
            "winner_color_mix": args.winner_color_mix,
            "chroma_restore": args.chroma_restore,
            "regional_ownership": args.enable_regional_ownership,
            "surface_fill_iterations": args.surface_fill_iterations,
            "surface_fill_normal_dot": args.surface_fill_normal_dot,
            "semantic_prior_iterations": args.semantic_prior_iterations,
            "semantic_prior_normal_dot": args.semantic_prior_normal_dot,
            "semantic_hard_reject": args.semantic_hard_reject,
        },
        "timings": {
            "sampling_seconds": sample_seconds,
            "consensus_seconds": consensus_seconds,
            "unwrap_seconds": unwrap_seconds,
            "bake_seconds": bake_seconds,
            "export_seconds": export_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
