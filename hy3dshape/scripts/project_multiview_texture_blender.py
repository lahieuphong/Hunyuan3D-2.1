"""Project four canonical views onto a dense Hunyuan mesh in Blender.

Run this file with Blender's bundled Python:
  blender --background --python project_multiview_texture_blender.py -- ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--head-center", type=float, default=0.72)
    parser.add_argument("--head-span", type=float, default=0.72)
    parser.add_argument("--hair-top-offset", type=float, default=0.06)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_mesh(path: Path) -> bpy.types.Object:
    bpy.ops.import_scene.gltf(filepath=str(path))
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
    obj.name = "Gohan"
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def bounds(obj: bpy.types.Object) -> dict[str, float]:
    xs = [vertex.co.x for vertex in obj.data.vertices]
    ys = [vertex.co.y for vertex in obj.data.vertices]
    zs = [vertex.co.z for vertex in obj.data.vertices]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
    }


def choose_view(
    normal: Vector,
    center: Vector,
    head_center: float,
    middle_depth: float,
) -> str:
    # glTF +Z becomes Blender -Y. Front-facing polygons therefore have -Y normals.
    # Give the front reference a wider cone around the face so eye lines remain intact.
    in_head_band = center.z > head_center - 0.32
    if in_head_band:
        if normal.y < -0.04:
            return "front"
        if normal.y > 0.04:
            return "back"
    if abs(normal.z) > max(abs(normal.x), abs(normal.y)):
        return "front" if center.y <= middle_depth else "back"
    if abs(normal.y) >= abs(normal.x):
        return "front" if normal.y < 0 else "back"
    return "left" if normal.x > 0 else "right"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def projected_coordinate(view: str, coordinate: Vector, limits: dict[str, float]) -> tuple[float, float]:
    x_span = limits["x_max"] - limits["x_min"]
    y_span = limits["y_max"] - limits["y_min"]
    z_span = limits["z_max"] - limits["z_min"]
    vertical = clamp((coordinate.z - limits["z_min"]) / z_span)

    if view == "front":
        horizontal = clamp((coordinate.x - limits["x_min"]) / x_span)
    elif view == "back":
        horizontal = 1.0 - clamp((coordinate.x - limits["x_min"]) / x_span)
    elif view == "left":
        horizontal = clamp((coordinate.y - limits["y_min"]) / y_span)
    else:
        horizontal = 1.0 - clamp((coordinate.y - limits["y_min"]) / y_span)
    return horizontal, vertical


def assign_projected_uvs(
    obj: bpy.types.Object,
    metadata: dict[str, object],
    head_center: float,
) -> None:
    mesh = obj.data
    uv_layer = mesh.uv_layers.get("ProjectedUV") or mesh.uv_layers.new(name="ProjectedUV")
    uv_layer.active_render = True
    limits = bounds(obj)
    atlas_size = float(metadata["atlas_size"])
    tile_size = float(metadata["tile_size"])
    middle_depth = (limits["y_min"] + limits["y_max"]) * 0.5

    for polygon in mesh.polygons:
        view = choose_view(polygon.normal, polygon.center, head_center, middle_depth)
        view_meta = metadata["views"][view]
        tile_x, tile_y = view_meta["tile"]
        x_min, y_min, x_max, y_max = view_meta["bbox"]

        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            horizontal, vertical = projected_coordinate(view, vertex.co, limits)
            source_x = x_min + horizontal * (x_max - x_min)
            source_y = y_max - vertical * (y_max - y_min)
            atlas_x = tile_x * tile_size + source_x
            atlas_y = tile_y * tile_size + source_y
            uv_layer.data[loop_index].uv = (
                atlas_x / atlas_size,
                1.0 - atlas_y / atlas_size,
            )

    mesh.uv_layers.active = uv_layer
    mesh.update()


def create_material(obj: bpy.types.Object, atlas_path: Path) -> None:
    material = bpy.data.materials.new("GohanProjected4K")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "ProjectedUV"
    texture.image = bpy.data.images.load(str(atlas_path), check_existing=False)
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    principled.inputs["Roughness"].default_value = 0.82
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.22
    principled.inputs["Emission Strength"].default_value = 0.12

    links.new(uv_map.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(texture.outputs["Color"], principled.inputs["Emission Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)


def assign_hair_material(
    obj: bpy.types.Object,
    metadata: dict[str, object],
    head_center: float,
    hair_top_offset: float,
) -> None:
    hair_mask_value = metadata.get("hair_mask")
    if not isinstance(hair_mask_value, str):
        return
    hair_mask_path = Path(hair_mask_value)
    if not hair_mask_path.is_file():
        raise FileNotFoundError(f"Hair mask not found: {hair_mask_path}")

    hair_image = bpy.data.images.load(str(hair_mask_path), check_existing=False)
    width, height = (int(value) for value in hair_image.size)
    pixels = np.asarray(hair_image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    hair_mask = np.flipud(pixels[..., 0])
    limits = bounds(obj)
    middle_depth = (limits["y_min"] + limits["y_max"]) * 0.5
    tile_size = float(metadata["tile_size"])

    material = bpy.data.materials.new("GohanHair")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.004, 0.005, 0.008, 1.0)
    principled.inputs["Roughness"].default_value = 0.72
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.24
    obj.data.materials.append(material)
    material_index = len(obj.data.materials) - 1

    assigned = 0
    for polygon in obj.data.polygons:
        if polygon.center.z <= head_center - 0.25:
            continue
        if polygon.center.z > head_center + hair_top_offset:
            polygon.material_index = material_index
            assigned += 1
            continue
        view = choose_view(polygon.normal, polygon.center, head_center, middle_depth)
        horizontal, vertical = projected_coordinate(view, polygon.center, limits)
        view_meta = metadata["views"][view]
        tile_x, tile_y = view_meta["tile"]
        x_min, y_min, x_max, y_max = view_meta["bbox"]
        source_x = x_min + horizontal * (x_max - x_min)
        source_y = y_max - vertical * (y_max - y_min)
        atlas_x = int(round(tile_x * tile_size + source_x))
        atlas_y = int(round(tile_y * tile_size + source_y))
        atlas_x = max(0, min(width - 1, atlas_x))
        atlas_y = max(0, min(height - 1, atlas_y))
        if hair_mask[atlas_y, atlas_x] > 0.5:
            polygon.material_index = material_index
            assigned += 1
    print(f"Assigned solid hair material to {assigned} polygons")


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def render_head(
    obj: bpy.types.Object,
    output_path: Path,
    size: int,
    head_center: float,
    head_span: float,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.world.color = (0.008, 0.012, 0.025)

    camera_data = bpy.data.cameras.new("BenchmarkCamera")
    camera = bpy.data.objects.new("BenchmarkCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = head_span
    camera.location = (0.0, -3.0, head_center)
    point_at(camera, Vector((0.0, 0.0, head_center)))
    scene.camera = camera

    key_data = bpy.data.lights.new("Key", type="AREA")
    key_data.energy = 750.0
    key_data.shape = "DISK"
    key_data.size = 4.0
    key = bpy.data.objects.new("Key", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-1.2, -2.2, 2.4)
    point_at(key, Vector((0.0, 0.0, head_center)))

    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 400.0
    fill_data.size = 3.0
    fill = bpy.data.objects.new("Fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (1.4, -1.4, 1.2)
    point_at(fill, Vector((0.0, 0.0, head_center)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def export_glb(obj: bpy.types.Object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
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
    obj = import_mesh(args.mesh.resolve())
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    assign_projected_uvs(obj, metadata, args.head_center)
    create_material(obj, args.atlas.resolve())
    assign_hair_material(obj, metadata, args.head_center, args.hair_top_offset)
    export_glb(obj, args.output.resolve())
    if args.render:
        render_head(
            obj,
            args.render.resolve(),
            args.render_size,
            args.head_center,
            args.head_span,
        )
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
