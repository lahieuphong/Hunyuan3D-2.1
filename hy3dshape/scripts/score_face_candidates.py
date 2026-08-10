"""Render and rank a small Hunyuan3D face-candidate seed search.

The host process uses OpenCV/trimesh to score the candidates.  It launches
Blender once in background mode to make deterministic orthographic clay and
normal renders of the head.  The same file is also the Blender worker so the
benchmark remains easy to reproduce.

Example:
  .venv-win/Scripts/python.exe hy3dshape/scripts/score_face_candidates.py \
    --input-json hy3dshape/output_folder/webui/quality_tests/face_seed_search.json \
    --reference-dir hy3dshape/output_folder/webui/quality_tests/gohan_rgba_clean \
    --output-dir hy3dshape/output_folder/webui/quality_tests/face_seed_scores
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


HEAD_CENTER = 0.72
HEAD_SPAN = 0.72
RENDER_SIZE = 768


def _args_after_separator() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--webui-root",
        type=Path,
        default=Path("hy3dshape/output_folder/webui/generations"),
    )
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"),
    )
    parser.add_argument("--render-size", type=int, default=RENDER_SIZE)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--blender-worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(_args_after_separator())


def _mesh_path_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "url"):
            if key in value:
                result = _mesh_path_value(value[key])
                if result:
                    return result
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _mesh_path_value(item)
            if result:
                return result
    return None


def load_candidates(input_json: Path, webui_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    records = payload["candidates"] if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"No candidates found in {input_json}")

    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Candidate {index} is not a JSON object")
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        generation = (
            stats.get("generation") if isinstance(stats.get("generation"), dict) else {}
        )
        uid = record.get("generation_uid") or generation.get("uid")
        preferred = webui_root / str(uid) / "white_mesh.glb" if uid else None
        fallback_text = _mesh_path_value(record.get("mesh"))
        fallback = Path(fallback_text) if fallback_text else None
        mesh = preferred if preferred and preferred.is_file() else fallback
        if mesh is None or not mesh.is_file():
            raise FileNotFoundError(
                f"Candidate {index} mesh missing (uid={uid!r}, mesh={fallback_text!r})"
            )
        item = dict(record)
        item["generation_uid"] = uid
        item["_index"] = index
        item["_mesh"] = str(mesh.resolve())
        item["_key"] = f"{index:02d}_seed_{int(record.get('seed', index))}"
        result.append(item)
    return result


def write_worker_manifest(
    candidates: list[dict[str, Any]], output_dir: Path, render_size: int
) -> Path:
    manifest = {
        "output_dir": str(output_dir.resolve()),
        "render_size": render_size,
        "head_center": HEAD_CENTER,
        "head_span": HEAD_SPAN,
        "candidates": [
            {"key": item["_key"], "mesh": item["_mesh"]} for item in candidates
        ],
    }
    path = output_dir / "blender_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def launch_blender(blender: Path, script: Path, manifest: Path) -> None:
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(script.resolve()),
        "--",
        "--blender-worker",
        str(manifest.resolve()),
    ]
    subprocess.run(command, check=True)


def clear_blender_scene() -> None:
    import bpy

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


def import_and_normalize_mesh(path: Path):
    import bpy
    from mathutils import Vector

    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {path}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    coordinates = [vertex.co.copy() for vertex in obj.data.vertices]
    minimum = Vector(
        (
            min(value.x for value in coordinates),
            min(value.y for value in coordinates),
            min(value.z for value in coordinates),
        )
    )
    maximum = Vector(
        (
            max(value.x for value in coordinates),
            max(value.y for value in coordinates),
            max(value.z for value in coordinates),
        )
    )
    center = (minimum + maximum) * 0.5
    scale = 2.0 / max(maximum.z - minimum.z, 1e-8)
    for vertex in obj.data.vertices:
        vertex.co = (vertex.co - center) * scale
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.update()
    return obj


def point_at(obj, target) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def create_camera(name: str, location, target, ortho_scale: float):
    import bpy
    from mathutils import Vector

    data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(camera)
    data.type = "ORTHO"
    data.ortho_scale = ortho_scale
    data.lens = 55
    camera.location = Vector(location)
    point_at(camera, Vector(target))
    return camera


def create_clay_material():
    import bpy

    material = bpy.data.materials.new("BenchmarkClay")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = (0.58, 0.62, 0.68, 1.0)
    principled.inputs["Roughness"].default_value = 0.92
    principled.inputs["Metallic"].default_value = 0.0
    return material


def create_normal_material():
    import bpy

    material = bpy.data.materials.new("BenchmarkNormal")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "NORMAL"
    transform.convert_from = "WORLD"
    transform.convert_to = "CAMERA"
    multiply = nodes.new("ShaderNodeVectorMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(geometry.outputs["Normal"], transform.inputs["Vector"])
    links.new(transform.outputs["Vector"], multiply.inputs[0])
    links.new(multiply.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def setup_blender_scene(size: int) -> None:
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (0.035, 0.035, 0.035)

    for x in (-1.3, 1.3):
        data = bpy.data.lights.new(f"Front_{x}", type="AREA")
        data.energy = 95.0
        data.shape = "DISK"
        data.size = 3.2
        light = bpy.data.objects.new(f"Front_{x}", data)
        bpy.context.collection.objects.link(light)
        light.location = (x, -2.2, 1.7)
        point_at(light, Vector((0.0, 0.0, HEAD_CENTER)))

    top_data = bpy.data.lights.new("Top", type="AREA")
    top_data.energy = 65.0
    top_data.size = 3.0
    top = bpy.data.objects.new("Top", top_data)
    bpy.context.collection.objects.link(top)
    top.location = (0.0, 0.3, 2.8)
    point_at(top, Vector((0.0, 0.0, HEAD_CENTER)))


def render_worker_candidate(candidate: dict[str, Any], output_dir: Path, size: int) -> None:
    import bpy

    clear_blender_scene()
    obj = import_and_normalize_mesh(Path(candidate["mesh"]))
    setup_blender_scene(size)
    clay = create_clay_material()
    normal = create_normal_material()
    obj.data.materials.clear()
    obj.data.materials.append(clay)

    cameras = {
        "front": create_camera(
            "FrontCamera",
            (0.0, -3.0, HEAD_CENTER),
            (0.0, 0.0, HEAD_CENTER),
            HEAD_SPAN,
        ),
        "side": create_camera(
            "SideCamera",
            (3.0, 0.0, HEAD_CENTER),
            (0.0, 0.0, HEAD_CENTER),
            HEAD_SPAN,
        ),
    }
    candidate_dir = output_dir / candidate["key"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    for view, camera in cameras.items():
        scene.camera = camera
        obj.data.materials[0] = clay
        scene.render.filepath = str(candidate_dir / f"{view}_clay.png")
        bpy.ops.render.render(write_still=True)
        obj.data.materials[0] = normal
        scene.render.filepath = str(candidate_dir / f"{view}_normal.png")
        bpy.ops.render.render(write_still=True)


def blender_worker(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(manifest["output_dir"])
    for index, candidate in enumerate(manifest["candidates"], start=1):
        print(
            f"[face-score] rendering {index}/{len(manifest['candidates'])}: "
            f"{candidate['key']}",
            flush=True,
        )
        render_worker_candidate(candidate, output_dir, int(manifest["render_size"]))


def foreground_head(path: Path, head_fraction: float = 0.34):
    import cv2
    import numpy as np

    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(path)
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGRA)
    elif rgba.shape[2] == 3:
        alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)
        rgba = np.concatenate((rgba, alpha), axis=2)
    mask = rgba[..., 3] > 16
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"No foreground in {path}")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    head_bottom = min(y1, int(round(y0 + (y1 - y0) * head_fraction)))
    horizontal_padding = int(round((x1 - x0) * 0.03))
    x0 = max(0, x0 - horizontal_padding)
    x1 = min(rgba.shape[1], x1 + horizontal_padding)
    return rgba[y0:head_bottom, x0:x1]


def normalize_foreground(image, size: int = 512, margin: float = 0.04):
    import cv2
    import numpy as np

    alpha = image[..., 3] > 16
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return np.zeros((size, size, 4), dtype=np.uint8)
    crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    available = int(round(size * (1.0 - 2.0 * margin)))
    scale = min(available / crop.shape[1], available / crop.shape[0])
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x = (size - width) // 2
    y = (size - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return canvas


def silhouette_score(candidate, reference) -> float:
    import cv2
    import numpy as np

    candidate_mask = np.uint8(candidate[..., 3] > 16) * 255
    reference_mask = np.uint8(reference[..., 3] > 16) * 255
    candidate_edge = cv2.Canny(candidate_mask, 50, 150) > 0
    reference_edge = cv2.Canny(reference_mask, 50, 150) > 0
    if not candidate_edge.any() or not reference_edge.any():
        return 0.0
    candidate_distance = cv2.distanceTransform(
        np.uint8(~candidate_edge) * 255, cv2.DIST_L2, 3
    )
    reference_distance = cv2.distanceTransform(
        np.uint8(~reference_edge) * 255, cv2.DIST_L2, 3
    )
    distance = 0.5 * (
        float(reference_distance[candidate_edge].mean())
        + float(candidate_distance[reference_edge].mean())
    )
    return float(math.exp(-distance / (candidate.shape[0] * 0.025)))


def eye_artifact_metrics(normal_rgba, clay_rgba) -> dict[str, float]:
    import cv2
    import numpy as np

    height, width = normal_rgba.shape[:2]
    x0, x1 = int(width * 0.24), int(width * 0.76)
    y0, y1 = int(height * 0.40), int(height * 0.68)
    normal = normal_rgba[y0:y1, x0:x1, :3].astype(np.float32) / 255.0
    clay = clay_rgba[y0:y1, x0:x1, :3]
    alpha = normal_rgba[y0:y1, x0:x1, 3] > 16
    gray = cv2.cvtColor(clay, cv2.COLOR_BGRA2GRAY) if clay.shape[2] == 4 else cv2.cvtColor(clay, cv2.COLOR_BGR2GRAY)

    gradient_channels = []
    for channel in range(3):
        gx = cv2.Sobel(normal[..., channel], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normal[..., channel], cv2.CV_32F, 0, 1, ksize=3)
        gradient_channels.append(gx * gx + gy * gy)
    gradient = np.sqrt(np.sum(gradient_channels, axis=0))
    values = gradient[alpha]
    gradient_tail = float(np.mean(np.sort(values)[-max(1, len(values) // 5) :])) if len(values) else 1.0

    edges = cv2.Canny(gray, 45, 120)
    edges[~alpha] = 0
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(max(1, alpha.sum()))
    closed_area = 0.0
    closed_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if roi_area * 0.004 <= area <= roi_area * 0.28 and perimeter > 0:
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.04:
                closed_area += area
                closed_count += 1
    edge_density = float(np.count_nonzero(edges) / roi_area)
    closed_ratio = float(closed_area / roi_area)
    artifact = gradient_tail + 1.8 * edge_density + 1.2 * closed_ratio
    return {
        "eye_normal_gradient": gradient_tail,
        "eye_edge_density": edge_density,
        "eye_closed_contour_ratio": closed_ratio,
        "eye_closed_contour_count": float(closed_count),
        "eye_artifact_raw": artifact,
    }


def symmetry_score(rgba) -> float:
    import numpy as np

    height, width = rgba.shape[:2]
    x0, x1 = int(width * 0.20), int(width * 0.80)
    y0, y1 = int(height * 0.28), int(height * 0.88)
    mask = rgba[y0:y1, x0:x1, 3] > 16
    if mask.shape[1] % 2:
        mask = mask[:, :-1]
    flipped = np.fliplr(mask)
    union = np.logical_or(mask, flipped).sum()
    return float(np.logical_and(mask, flipped).sum() / max(1, union))


def component_metrics(mesh_path: Path) -> dict[str, float]:
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    parts = mesh.split(only_watertight=False)
    areas = sorted((float(part.area) for part in parts), reverse=True)
    total_area = sum(areas)
    largest_ratio = areas[0] / total_area if areas and total_area else 0.0
    significant = sum(area >= total_area * 0.001 for area in areas) if total_area else 0
    cleanliness = largest_ratio * math.exp(-0.20 * max(0, significant - 1))
    return {
        "vertices": float(len(mesh.vertices)),
        "faces": float(len(mesh.faces)),
        "components": float(len(parts)),
        "significant_components": float(significant),
        "largest_component_area_ratio": float(largest_ratio),
        "component_cleanliness": float(cleanliness),
    }


def read_rgba(path: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[2] == 3:
        image = np.dstack((image, np.full(image.shape[:2], 255, dtype=np.uint8)))
    return image


def minmax_quality(values: list[float], higher_is_better: bool = True) -> list[float]:
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum, rel_tol=1e-7, abs_tol=1e-9):
        return [1.0] * len(values)
    result = [(value - minimum) / (maximum - minimum) for value in values]
    return result if higher_is_better else [1.0 - value for value in result]


def score_candidates(
    candidates: list[dict[str, Any]], reference_dir: Path, output_dir: Path
) -> list[dict[str, Any]]:
    reference_front = normalize_foreground(
        foreground_head(reference_dir / "Gohan_Front_transparent.png")
    )
    reference_side = normalize_foreground(
        foreground_head(reference_dir / "Gohan_Left_transparent.png")
    )
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        directory = output_dir / candidate["_key"]
        front_clay = read_rgba(directory / "front_clay.png")
        front_normal = read_rgba(directory / "front_normal.png")
        side_clay = read_rgba(directory / "side_clay.png")
        front_normalized = normalize_foreground(front_clay)
        side_normalized = normalize_foreground(side_clay)
        metrics: dict[str, float] = {}
        metrics.update(eye_artifact_metrics(front_normal, front_clay))
        metrics.update(component_metrics(Path(candidate["_mesh"])))
        metrics["front_silhouette"] = silhouette_score(front_normalized, reference_front)
        metrics["side_silhouette"] = silhouette_score(side_normalized, reference_side)
        metrics["face_symmetry"] = symmetry_score(front_clay)
        scored.append(
            {
                "seed": candidate.get("seed"),
                "generation_uid": candidate.get("generation_uid"),
                "mesh": candidate["_mesh"],
                "key": candidate["_key"],
                "metrics": metrics,
            }
        )

    # Use absolute calibrated qualities here.  Min/max ranking would exaggerate
    # negligible differences (for example symmetry 0.977 versus 0.987) when
    # the seed-search batch is small.
    qualities = {
        "eye": [
            math.exp(-item["metrics"]["eye_artifact_raw"] / 2.0) for item in scored
        ],
        "front": [item["metrics"]["front_silhouette"] for item in scored],
        "side": [item["metrics"]["side_silhouette"] for item in scored],
        "components": [item["metrics"]["component_cleanliness"] for item in scored],
        "symmetry": [item["metrics"]["face_symmetry"] for item in scored],
    }
    weights = {
        "eye": 0.40,
        "front": 0.20,
        "side": 0.16,
        "components": 0.12,
        "symmetry": 0.12,
    }
    for index, item in enumerate(scored):
        item["quality"] = {
            name: round(values[index], 6) for name, values in qualities.items()
        }
        item["score"] = round(
            100.0
            * sum(weights[name] * qualities[name][index] for name in weights),
            3,
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank
    return scored


def contact_sheet(
    scored: list[dict[str, Any]],
    reference_dir: Path,
    output_dir: Path,
    destination: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tile = 300
    header = 82
    gap = 12
    columns = len(scored) + 1
    canvas = Image.new(
        "RGB",
        (gap + columns * (tile + gap), header + 3 * (tile + gap)),
        (13, 18, 31),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
        small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = font

    def place(path: Path, column: int, row: int, crop: bool = False) -> None:
        image = Image.open(path).convert("RGBA")
        if crop:
            # Reference files are full-body; show the head/shoulder third.
            alpha = image.getchannel("A")
            box = alpha.getbbox()
            if box:
                x0, y0, x1, y1 = box
                y1 = y0 + int((y1 - y0) * 0.34)
                image = image.crop((x0, y0, x1, y1))
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        panel = Image.new("RGBA", (tile, tile), (22, 29, 47, 255))
        panel.alpha_composite(image, ((tile - image.width) // 2, (tile - image.height) // 2))
        x = gap + column * (tile + gap)
        y = header + row * (tile + gap)
        canvas.paste(panel.convert("RGB"), (x, y))

    draw.text((gap, 14), "REFERENCE", fill=(245, 247, 255), font=font)
    draw.text((gap, 42), "Front / left / source crop", fill=(155, 166, 193), font=small)
    place(reference_dir / "Gohan_Front_transparent.png", 0, 0, crop=True)
    place(reference_dir / "Gohan_Left_transparent.png", 0, 1, crop=True)
    place(reference_dir / "Gohan_Front_transparent.png", 0, 2, crop=True)

    for column, item in enumerate(scored, start=1):
        x = gap + column * (tile + gap)
        draw.text(
            (x, 12),
            f"#{item['rank']}  seed {item['seed']}",
            fill=(116, 245, 165) if item["rank"] == 1 else (245, 247, 255),
            font=font,
        )
        draw.text(
            (x, 42),
            f"score {item['score']:.1f}  eye {item['quality']['eye']:.2f}",
            fill=(155, 166, 193),
            font=small,
        )
        directory = output_dir / item["key"]
        place(directory / "front_clay.png", column, 0)
        place(directory / "side_clay.png", column, 1)
        place(directory / "front_normal.png", column, 2)

    draw.text((gap, header - 20), "FRONT CLAY", fill=(112, 126, 158), font=small)
    draw.text((gap, header + tile + gap - 20), "SIDE CLAY", fill=(112, 126, 158), font=small)
    draw.text((gap, header + 2 * (tile + gap) - 20), "FRONT NORMAL", fill=(112, 126, 158), font=small)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def host_main(args: argparse.Namespace) -> None:
    if not args.input_json or not args.reference_dir or not args.output_dir:
        raise SystemExit("--input-json, --reference-dir and --output-dir are required")
    input_json = args.input_json.resolve()
    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(input_json, args.webui_root.resolve())
    manifest = write_worker_manifest(candidates, output_dir, args.render_size)
    if not args.skip_render:
        launch_blender(args.blender.resolve(), Path(__file__), manifest)
    scored = score_candidates(candidates, reference_dir, output_dir)
    report = {
        "source": str(input_json),
        "selection_rule": (
            "40% low eye ring/groove artifact, 20% front silhouette, "
            "16% side profile, 12% component cleanliness, 12% symmetry"
        ),
        "best": scored[0],
        "candidates": scored,
    }
    report_path = output_dir / "face_candidate_scores.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sheet_path = output_dir / "face_candidate_contact_sheet.png"
    contact_sheet(scored, reference_dir, output_dir, sheet_path)
    print(f"Best candidate: seed={scored[0]['seed']} score={scored[0]['score']}")
    print(f"Report: {report_path}")
    print(f"Contact sheet: {sheet_path}")


def main() -> None:
    args = parse_args()
    if args.blender_worker:
        blender_worker(args.blender_worker.resolve())
    else:
        host_main(args)


if __name__ == "__main__":
    main()
