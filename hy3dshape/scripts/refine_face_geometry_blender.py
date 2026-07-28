"""Create controlled smoothing/subdivision variants of a textured GLB.

This worker is intended for final face-quality sweeps.  It preserves the
projected UVs/materials and limits smoothing to the front eye/face region so
hair spikes and clothing remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("subsurf", "smooth_eye", "smooth_face"),
    )
    parser.add_argument("--factor", type=float)
    parser.add_argument("--iterations", type=int)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_joined_mesh(path: Path) -> bpy.types.Object:
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh object imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def textured_vertex_indices(obj: bpy.types.Object) -> set[int]:
    material_names = [material.name if material else "" for material in obj.data.materials]
    textured_indices = {
        index for index, name in enumerate(material_names) if "Hair" not in name
    }
    result: set[int] = set()
    for polygon in obj.data.polygons:
        if polygon.material_index in textured_indices:
            result.update(polygon.vertices)
    return result


def add_face_vertex_group(
    obj: bpy.types.Object,
    mode: str,
) -> bpy.types.VertexGroup:
    coordinates = [vertex.co for vertex in obj.data.vertices]
    middle_depth = 0.5 * (
        min(coordinate.y for coordinate in coordinates)
        + max(coordinate.y for coordinate in coordinates)
    )
    if mode == "smooth_eye":
        z_min, z_max, x_limit = 0.655, 0.795, 0.205
    else:
        z_min, z_max, x_limit = 0.505, 0.815, 0.225

    allowed = textured_vertex_indices(obj)
    group = obj.vertex_groups.new(name=f"{mode}_region")
    selected = 0
    for vertex in obj.data.vertices:
        coordinate = vertex.co
        if (
            vertex.index not in allowed
            or not z_min < coordinate.z < z_max
            or abs(coordinate.x) >= x_limit
            or coordinate.y >= middle_depth - 0.015
        ):
            continue
        vertical = (coordinate.z - z_min) / (z_max - z_min)
        vertical_weight = max(0.0, min(1.0, 4.0 * vertical * (1.0 - vertical)))
        horizontal_weight = max(0.0, 1.0 - (abs(coordinate.x) / x_limit) ** 4)
        depth_weight = max(
            0.0,
            min(1.0, (middle_depth - coordinate.y - 0.015) / 0.075),
        )
        weight = vertical_weight * horizontal_weight * depth_weight
        if weight > 0.01:
            group.add([vertex.index], weight, "REPLACE")
            selected += 1
    if not selected:
        raise RuntimeError(f"No vertices selected for {mode}")
    print(f"Selected {selected} weighted vertices for {mode}")
    return group


def refine(obj: bpy.types.Object, args: argparse.Namespace) -> None:
    bpy.context.view_layer.objects.active = obj
    if args.mode == "subsurf":
        modifier = obj.modifiers.new("FaceQualitySubdivision", "SUBSURF")
        modifier.subdivision_type = "CATMULL_CLARK"
        modifier.levels = 1
        modifier.render_levels = 1
    else:
        group = add_face_vertex_group(obj, args.mode)
        modifier = obj.modifiers.new("FaceQualitySmooth", "SMOOTH")
        modifier.vertex_group = group.name
        modifier.factor = args.factor if args.factor is not None else (
            0.30 if args.mode == "smooth_eye" else 0.16
        )
        modifier.iterations = args.iterations if args.iterations is not None else (
            3 if args.mode == "smooth_eye" else 2
        )
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.update()


def export_glb(obj: bpy.types.Object, destination: Path) -> None:
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


def main() -> None:
    args = parse_args()
    clear_scene()
    obj = import_joined_mesh(args.mesh)
    refine(obj, args)
    export_glb(obj, args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
