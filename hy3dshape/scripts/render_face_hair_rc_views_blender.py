"""Render deterministic unlit views aligned to the supplied references.

This worker is intentionally separate from candidate scoring.  Blender owns
the camera and visibility work; ``face_hair_rc.py`` remains a normal
NumPy/OpenCV module that can be unit tested without Blender.

The orthographic camera uses the exact calibration convention used by the
ten-view texture baker.  Output can be downscaled for fast RC evaluation while
retaining the same framing as the original reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPOSITORY_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from bake_visibility_texture_blender import (  # noqa: E402
    clear_scene,
    import_joined_mesh,
    mesh_arrays,
)
from hy3dshape.texture_bake.calibration import (  # noqa: E402
    OrthographicCalibration,
    fit_orthographic_from_alpha,
)
from hy3dshape.texture_bake.ten_view_consensus import (  # noqa: E402
    HORIZONTAL_VIEW_NAMES,
    TEN_VIEW_ANGLES,
    ten_view_frames,
)


VIEW_ORDER = tuple(TEN_VIEW_ANGLES)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    for name in VIEW_ORDER:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            f"--{name}",
            dest=name,
            required=True,
            type=Path,
        )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--resolution-scale", type=float, default=0.5)
    return parser.parse_args(argv)


def load_alpha_top_down(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    width, height = (int(value) for value in image.size)
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    alpha = pixels.reshape(height, width, 4)[::-1, :, 3].copy()
    bpy.data.images.remove(image)
    return alpha, (width, height)


def prepare_calibrations(
    args: argparse.Namespace,
    vertices: np.ndarray,
) -> tuple[dict[str, OrthographicCalibration], dict[str, tuple[int, int]]]:
    frames = ten_view_frames()
    alpha: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    provisional: dict[str, OrthographicCalibration] = {}
    for name in VIEW_ORDER:
        path = Path(getattr(args, name)).resolve(strict=True)
        alpha[name], sizes[name] = load_alpha_top_down(path)
        provisional[name] = fit_orthographic_from_alpha(
            alpha[name],
            vertices,
            frames[name],
            fit_mode="height",
        )
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
            alpha[name],
            vertices,
            frames[name],
            fit_mode="height",
            pixels_per_unit=shared_scale,
        )
        for name in VIEW_ORDER
    }
    return calibrations, sizes


def make_material_unlit(material: bpy.types.Material) -> None:
    if not material.use_nodes or material.node_tree is None:
        return
    image = next(
        (
            node.image
            for node in material.node_tree.nodes
            if node.bl_idname == "ShaderNodeTexImage" and node.image is not None
        ),
        None,
    )
    if image is None:
        return
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])


def point_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (
        target - camera.location
    ).to_track_quat("-Z", "Y").to_euler()


def camera_target(
    calibration: OrthographicCalibration,
    vertices: np.ndarray,
) -> Vector:
    image_center_x = (calibration.image_width - 1) * 0.5
    image_center_y = (calibration.image_height - 1) * 0.5
    target_u = (
        calibration.world_center_u
        + (image_center_x - calibration.pixel_center_x)
        / calibration.pixels_per_unit_u
    )
    target_v = (
        calibration.world_center_v
        + (calibration.pixel_center_y - image_center_y)
        / calibration.pixels_per_unit_v
    )
    depth = float(
        np.median(vertices @ calibration.frame.to_camera_array)
    )
    target = (
        calibration.frame.right_array * target_u
        + calibration.frame.up_array * target_v
        + calibration.frame.to_camera_array * depth
    )
    return Vector(tuple(float(value) for value in target))


def configure_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    return scene


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.resolution_scale <= 1.0:
        raise ValueError("resolution-scale must be between 0.1 and 1")
    clear_scene()
    obj = import_joined_mesh(args.mesh.resolve(strict=True))
    (
        vertices,
        _,
        _,
        _,
        _,
        _,
    ) = mesh_arrays(obj.data)
    calibrations, sizes = prepare_calibrations(args, vertices)
    for material in bpy.data.materials:
        make_material_unlit(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True

    scene = configure_scene()
    camera_data = bpy.data.cameras.new("FaceHairRCCamera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("FaceHairRCCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    diagonal = float(
        np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, object] = {}

    for name in VIEW_ORDER:
        calibration = calibrations[name]
        width, height = sizes[name]
        scene.render.resolution_x = max(
            64,
            int(round(width * args.resolution_scale)),
        )
        scene.render.resolution_y = max(
            64,
            int(round(height * args.resolution_scale)),
        )
        camera_data.ortho_scale = (
            calibration.image_height / calibration.pixels_per_unit_v
        )
        target = camera_target(calibration, vertices)
        camera.location = (
            target
            + Vector(calibration.frame.to_camera) * (2.0 * diagonal)
        )
        point_at(camera, target)
        output = (args.output_dir / f"{name}.png").resolve()
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        rendered[name] = {
            "file": str(output),
            "size": [
                scene.render.resolution_x,
                scene.render.resolution_y,
            ],
            "calibration": calibration.to_dict(),
        }
        print(f"Rendered RC view {name}: {output}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "method": "reference-aligned-unlit-face-hair-rc",
                "mesh": str(args.mesh.resolve()),
                "resolution_scale": args.resolution_scale,
                "views": rendered,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


main()
