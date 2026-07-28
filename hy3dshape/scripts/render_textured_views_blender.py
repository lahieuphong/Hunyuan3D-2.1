"""Render consistent close-up views of a textured GLB with Blender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


VIEWS = {
    "front": (0.0, -3.0),
    "front_right": (-2.12, -2.12),
    "right": (-3.0, 0.0),
    "back": (0.0, 3.0),
    "left": (3.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--head-center", type=float, default=0.72)
    parser.add_argument("--head-span", type=float, default=0.72)
    return parser.parse_args(argv)


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.mesh.resolve()))
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            for polygon in obj.data.polygons:
                polygon.use_smooth = True

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True

    target = Vector((0.0, 0.0, args.head_center))
    camera_data = bpy.data.cameras.new("FaceCamera")
    camera = bpy.data.objects.new("FaceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = args.head_span
    scene.camera = camera

    key_data = bpy.data.lights.new("Key", type="AREA")
    key_data.energy = 750.0
    key_data.shape = "DISK"
    key_data.size = 4.0
    key = bpy.data.objects.new("Key", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-1.2, -2.2, 2.4)
    point_at(key, target)

    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 350.0
    fill_data.size = 3.0
    fill = bpy.data.objects.new("Fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (1.4, -1.4, 1.2)
    point_at(fill, target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in VIEWS.items():
        camera.location = (x, y, args.head_center)
        point_at(camera, target)
        scene.render.filepath = str((args.output_dir / f"{name}.png").resolve())
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {name}: {scene.render.filepath}")


if __name__ == "__main__":
    main()
