# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# Apply torchvision compatibility fix before other imports

import sys
import importlib.util
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')


try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uuid
import numpy as np

from hy3dshape.utils import logger
MAX_SEED = 10_000_000
HAS_REMBG = importlib.util.find_spec('rembg') is not None
HAS_PYMESHLAB = importlib.util.find_spec('pymeshlab') is not None
ENV = "Local" # "Huggingface"

# These two profiles were exercised end-to-end with Hunyuan3D-2mv on the
# workspace RTX 3090 (24 GiB). Keep the conservative profile as the initial UI
# value, and let users opt into the denser mesh extraction explicitly.
RTX3090_PRESETS = {
    'safe': {
        'steps': 30,
        'guidance_scale': 5.0,
        'octree_resolution': 256,
        'num_chunks': 8000,
    },
    'quality': {
        'steps': 30,
        'guidance_scale': 5.0,
        'octree_resolution': 384,
        'num_chunks': 8000,
    },
}
spaces_api: Any

if ENV == 'Huggingface':
    """
    Setup environment for running on Huggingface platform.

    This block performs the following:
    - Changes directory to the differentiable renderer folder and runs a shell 
        script to compile the mesh painter.
    - Installs a custom rasterizer wheel package via pip.

    Note:
        This setup assumes the script is running in the Huggingface environment 
        with the specified directory structure.
    """
    import shlex
    # Hugging Face injects this optional module in Spaces deployments.
    import spaces as hf_spaces  # pyright: ignore[reportMissingImports]
    spaces_api = hf_spaces
    print("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    os.system("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    print('install custom')
    subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"),
                   check=True)
else:
    """
    Define a dummy Spaces API with a GPU decorator for the local environment.

    The GPU decorator is a no-op that simply returns the decorated function unchanged.
    This allows code that uses the `spaces_api.GPU` decorator to run without modification locally.
    """
    class _LocalSpaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func

    spaces_api = _LocalSpaces()


_RMBG_WORKER = None
_POSTPROCESSORS = None


def get_rtx3090_preset(profile):
    """Return Gradio values and an explanatory status card for a GPU preset."""
    if profile not in RTX3090_PRESETS:
        raise ValueError(f"Unknown RTX 3090 preset: {profile}")

    preset = RTX3090_PRESETS[profile]
    is_quality = profile == 'quality'
    profile_name = 'Chất lượng cao' if is_quality else 'Mặc định an toàn'
    profile_class = 'quality' if is_quality else 'safe'
    status_html = f"""
    <div class="rtx-preset-status {profile_class}" data-profile="{profile_class}">
        <div class="rtx-preset-status-heading">
            <div class="rtx-preset-status-title">
                <span class="rtx-preset-status-check ui-icon-slot" data-ui-icon="check" aria-hidden="true"></span>
                <span>RTX 3090 · 1 ảnh &amp; 4 ảnh · {profile_name}</span>
            </div>
            <span class="rtx-preset-current">Đang dùng</span>
        </div>
        <div class="rtx-preset-values">
            <span><b>{preset['steps']}</b><small>Steps</small></span>
            <span><b>{preset['guidance_scale']}</b><small>Guidance</small></span>
            <span><b>{preset['octree_resolution']}</b><small>Octree</small></span>
            <span><b>{preset['num_chunks']}</b><small>Chunks</small></span>
        </div>
    </div>
    """
    return (
        preset['steps'],
        preset['guidance_scale'],
        preset['octree_resolution'],
        preset['num_chunks'],
        status_html,
    )


def get_background_remover():
    """Load rembg only when the user explicitly requests background removal."""
    global _RMBG_WORKER
    if _RMBG_WORKER is None:
        try:
            from hy3dshape.rembg import BackgroundRemover
            _RMBG_WORKER = BackgroundRemover()
        except ModuleNotFoundError as error:
            raise gr.Error(
                "Background removal is not installed. Install the Windows Web UI "
                "dependencies or upload transparent PNG files and leave Remove Background off."
            ) from error
    return _RMBG_WORKER


def get_postprocessors():
    """Load PyMeshLab postprocessors only for optional transform operations."""
    global _POSTPROCESSORS
    if _POSTPROCESSORS is None:
        try:
            from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover
            _POSTPROCESSORS = (
                FloaterRemover(),
                DegenerateFaceRemover(),
                FaceReducer(),
            )
        except ModuleNotFoundError as error:
            raise gr.Error(
                "Mesh transform tools are not installed. The generated GLB can still be "
                "downloaded directly above."
            ) from error
    return _POSTPROCESSORS

def get_example_img_list():
    """
    Load and return a sorted list of example image file paths.

    Searches recursively for PNG images under the './assets/example_images/' directory.

    Returns:
        list[str]: Sorted list of file paths to example PNG images.
    """
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    """
    Load and return a list of example text prompts.

    Reads lines from the './assets/example_prompts.txt' file, stripping whitespace.

    Returns:
        list[str]: List of example text prompts.
    """
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


class GenerationUidConflictError(Exception):
    """Raised when a request tries to reuse an existing generation folder."""


def normalize_generation_uid(generation_uid=None):
    if generation_uid is None or not str(generation_uid).strip():
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str(generation_uid).strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise gr.Error("Generation UID không hợp lệ. Hãy tải lại trang và thử lại.") from exc


def generation_uid_query_from_request(request=None):
    if request is not None:
        referer = request.headers.get('referer', '')
        generation_values = parse_qs(urlparse(referer).query).get('generation', [])
        if generation_values:
            return normalize_generation_uid(generation_values[0])
    return None


def generation_uid_from_request(request=None):
    generation_uid = generation_uid_query_from_request(request)
    if generation_uid:
        return generation_uid
    return normalize_generation_uid()


def generation_storage_path(save_folder):
    try:
        storage_path = os.path.relpath(save_folder, start=os.getcwd())
    except ValueError:
        storage_path = os.path.abspath(save_folder)
    return storage_path.replace(os.sep, '/')


def write_generation_manifest(save_folder, **updates):
    manifest_path = os.path.join(save_folder, 'generation.json')
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read generation manifest: %s", manifest_path)

    manifest.update(updates)
    temp_path = f"{manifest_path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    for attempt in range(10):
        try:
            os.replace(temp_path, manifest_path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01 * (attempt + 1))
    return manifest_path


GENERATION_STAGE_PROGRESS = {
    'request_received': 2,
    'validating_input': 4,
    'input_validated': 7,
    'input_saved': 10,
    'preprocessing_input': 13,
    'input_ready': 16,
    'shape_generation': 18,
    'prepare_conditioning': 20,
    'encode_conditioning': 23,
    'conditioning_ready': 26,
    'prepare_timestep_schedule': 28,
    'latents_initialized': 30,
    'diffusion_started': 31,
    'diffusion_completed': 74,
    'vae_decoding': 77,
    'volume_decoding': 79,
    'volume_decoding_progress': 80,
    'volume_decoding_completed': 89,
    'surface_extraction': 90,
    'surface_extraction_completed': 91,
    'trimesh_conversion': 92,
    'extracting_mesh': 92,
    'mesh_ready': 93,
    'exporting_glb': 95,
    'building_preview': 98,
    'completed': 100,
    'failed': 100,
}


GENERATION_STAGE_MESSAGES = {
    'request_received': 'Generation request accepted',
    'validating_input': 'Validating input payload',
    'input_validated': 'Input validation completed',
    'input_saved': 'Input snapshots saved to source storage',
    'preprocessing_input': 'Preprocessing input views',
    'input_ready': 'Input tensor is ready for inference',
    'shape_generation': 'Starting Hunyuan3D inference pipeline',
    'prepare_conditioning': 'Preparing image conditioning tensors',
    'encode_conditioning': 'Encoding vision conditioning features',
    'conditioning_ready': 'Vision conditioning is ready',
    'prepare_timestep_schedule': 'Building diffusion timestep schedule',
    'latents_initialized': 'Latent noise tensor initialized',
    'diffusion_started': 'Diffusion sampling started',
    'diffusion_completed': 'Diffusion sampling completed',
    'vae_decoding': 'Decoding latent representation with ShapeVAE',
    'volume_decoding': 'Starting dense volume decoding',
    'volume_decoding_progress': 'Decoding dense volume chunks',
    'volume_decoding_completed': 'Dense volume decoding completed',
    'surface_extraction': 'Running marching-cubes surface extraction',
    'surface_extraction_completed': 'Surface extraction completed',
    'trimesh_conversion': 'Converting generated surface to Trimesh',
    'extracting_mesh': 'Converting generated surface to Trimesh',
    'mesh_ready': 'Mesh geometry is ready',
    'exporting_glb': 'Exporting binary GLB',
    'building_preview': 'Building interactive 3D preview',
    'completed': 'Generation completed successfully',
    'failed': 'Generation failed',
}


def update_generation_stage(
    save_folder,
    stage,
    message=None,
    progress=None,
    event_details=None,
    **updates,
):
    manifest_path = os.path.join(save_folder, 'generation.json')
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read generation stage history: %s", manifest_path)

    updated_at = utc_now_iso()
    events = list(manifest.get('events', []))
    stage_progress = (
        GENERATION_STAGE_PROGRESS.get(stage, 0)
        if progress is None
        else max(0, min(100, float(progress)))
    )
    event = {
        'stage': stage,
        'message': message or GENERATION_STAGE_MESSAGES.get(stage, stage),
        'at': updated_at,
        'progress': stage_progress,
    }
    if event_details:
        event.update(event_details)
    events.append(event)
    updates.update({
        'stage': stage,
        'progress': stage_progress,
        'updated_at': updated_at,
        'events': events,
    })
    return write_generation_manifest(save_folder, **updates)


def save_generation_inputs(save_folder, image):
    input_files = {}
    images = image if isinstance(image, dict) else {'image': image}
    for view_name, view_image in images.items():
        if view_image is None or not hasattr(view_image, 'save'):
            continue
        safe_view_name = ''.join(
            character for character in str(view_name).lower()
            if character.isalnum() or character in ('-', '_')
        ) or 'image'
        filename = f"input_{safe_view_name}.png"
        try:
            view_image.save(os.path.join(save_folder, filename), format='PNG')
            input_files[str(view_name)] = filename
        except Exception as exc:
            logger.warning("Could not save generation input %s: %s", view_name, exc)
    return input_files


def mark_generation_failed(generation_uid, error):
    if generation_uid is None:
        return
    save_folder = os.path.join(SAVE_DIR, str(generation_uid))
    if not os.path.isdir(save_folder):
        return
    try:
        update_generation_stage(
            save_folder,
            'failed',
            status='failed',
            failed_at=utc_now_iso(),
            error=str(error),
        )
    except Exception as manifest_error:
        logger.warning(
            "Could not mark generation %s as failed: %s",
            generation_uid,
            manifest_error,
        )


def gen_save_folder(max_size=200, generation_uid=None):
    """
    Generate a new save folder inside SAVE_DIR, maintaining a maximum number of folders.

    If the number of existing folders in SAVE_DIR exceeds `max_size`, the oldest folder is removed.

    Args:
        max_size (int, optional): Maximum number of folders to keep in SAVE_DIR. Defaults to 200.

        generation_uid (str, optional): UUID to use as the folder name. A UUID is
            generated automatically when omitted.

    Returns:
        str: Path to the newly created save folder.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    generation_uid = normalize_generation_uid(generation_uid)
    new_folder = os.path.join(SAVE_DIR, generation_uid)
    if os.path.exists(new_folder):
        raise GenerationUidConflictError(
            f"Generation UID đã tồn tại: {generation_uid}"
        )

    dirs = []
    for candidate in Path(SAVE_DIR).iterdir():
        if not candidate.is_dir():
            continue
        try:
            uuid.UUID(candidate.name)
        except ValueError:
            continue
        dirs.append(candidate)
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")
    try:
        os.makedirs(new_folder, exist_ok=False)
    except FileExistsError as exc:
        raise GenerationUidConflictError(
            f"Generation UID đã tồn tại: {generation_uid}"
        ) from exc
    print(f"Created new folder: {new_folder}")
    return new_folder


# Removed complex PBR conversion functions - using simple trimesh-based conversion
def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    Export a mesh to a file in the specified folder, optionally including textures.

    Args:
        mesh (trimesh.Trimesh): The mesh object to export.
        save_folder (str): Directory path where the mesh file will be saved.
        textured (bool, optional): Whether to include textures/normals in the export. Defaults to False.
        type (str, optional): File format to export ('glb' or 'obj' supported). Defaults to 'glb'.

    Returns:
        str: The full path to the exported mesh file.
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path




def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> None:
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    # 执行转换
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)
            


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def render_model_viewer_document(mesh_src, height, width, textured=False):
    template_name = (
        './assets/modelviewer-textured-template.html'
        if textured
        else './assets/modelviewer-template.html'
    )
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()
    return (
        template_html
        .replace('var(--viewer-height, 650px)', f'{height}px')
        .replace('#width#', str(width))
        .replace('#src#', mesh_src)
    )


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    if textured:
        related_path = "./textured_mesh.glb"
        output_html_path = os.path.join(save_folder, 'textured_mesh.html')
    else:
        related_path = "./white_mesh.glb"
        output_html_path = os.path.join(save_folder, 'white_mesh.html')
    offset = 50 if textured else 10
    template_html = render_model_viewer_document(
        related_path,
        height - offset,
        width,
        textured=textured,
    )

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR).replace(os.sep, '/')
    iframe_tag = f'<iframe src="/static/{rel_path}" \
height="{height}" width="100%" frameborder="0" title="Generated 3D mesh preview" allow="fullscreen" allowfullscreen></iframe>'
    print(f'Find html file {output_html_path}, \
{os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}px; width: 100%; overflow: hidden;'>
        {iframe_tag}
        </div>
    """


def stored_generation_file(save_folder, filename):
    """Resolve a manifest filename without allowing it outside its generation folder."""
    if not filename or os.path.basename(str(filename)) != str(filename):
        return None
    save_folder = os.path.abspath(save_folder)
    candidate = os.path.abspath(os.path.join(save_folder, str(filename)))
    try:
        if os.path.commonpath([save_folder, candidate]) != save_folder:
            return None
    except ValueError:
        return None
    return candidate if os.path.isfile(candidate) else None


def build_stored_model_viewer_html(save_folder, mesh_filename, height=660):
    """Embed a saved mesh through the current viewer UI without changing the GLB."""
    mesh_path = stored_generation_file(save_folder, mesh_filename)
    if not mesh_path:
        return HTML_OUTPUT_PLACEHOLDER

    generation_uid = os.path.basename(os.path.abspath(save_folder))
    cache_key = os.stat(mesh_path).st_mtime_ns
    iframe_tag = (
        f'<iframe src="/generation-viewer/{generation_uid}?v={cache_key}" '
        f'height="{height}" width="100%" frameborder="0" title="Generated 3D mesh preview" '
        f'allow="fullscreen" allowfullscreen></iframe>'
    )
    return f"""
        <div style='height: {height}px; width: 100%; overflow: hidden;'>
        {iframe_tag}
        </div>
    """


def restore_generation_from_request(request: gr.Request | None = None):
    """Restore saved inputs, mesh preview and settings from a generation URL."""
    unchanged = tuple(gr.update() for _ in range(17))
    generation_uid = generation_uid_query_from_request(request)
    if not generation_uid:
        return unchanged

    save_folder = os.path.join(SAVE_DIR, generation_uid)
    manifest_path = os.path.join(save_folder, 'generation.json')
    if not os.path.isfile(manifest_path):
        logger.warning("Saved generation was not found: %s", generation_uid)
        return unchanged

    try:
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not restore generation manifest: %s", manifest_path)
        return unchanged

    params = manifest.get('params') or {}
    inputs = manifest.get('inputs') or {}
    outputs = manifest.get('outputs') or {}
    input_mode = manifest.get('input_mode') or params.get('input_mode') or 'single'
    input_mode = 'four' if input_mode in {'four', '4-view', 'multi-view'} else 'single'

    def input_path(view_name, fallback_filename):
        return stored_generation_file(
            save_folder,
            inputs.get(view_name) or fallback_filename,
        )

    front_image = input_path('front', 'input_front.png')
    back_image = input_path('back', 'input_back.png')
    left_image = input_path('left', 'input_left.png')
    right_image = input_path('right', 'input_right.png')

    mesh_path = stored_generation_file(
        save_folder,
        outputs.get('mesh') or 'white_mesh.glb',
    )
    viewer_html = (
        build_stored_model_viewer_html(
            save_folder,
            outputs.get('mesh') or 'white_mesh.glb',
            HTML_HEIGHT,
        )
        if mesh_path
        else HTML_OUTPUT_PLACEHOLDER
    )

    return (
        input_mode,
        front_image if input_mode == 'single' else None,
        front_image if input_mode == 'four' else None,
        back_image if input_mode == 'four' else None,
        left_image if input_mode == 'four' else None,
        right_image if input_mode == 'four' else None,
        mesh_path,
        viewer_html,
        manifest.get('stats') or {},
        params.get('seed', 1234),
        params.get('steps', 30),
        params.get('guidance_scale', 5.0),
        params.get('octree_resolution', 256),
        params.get('check_box_rembg', not MV_MODE),
        params.get('num_chunks', 8000),
        False,
        gr.update(value=(
            'Generate 3D · 4 Images' if input_mode == 'four'
            else 'Generate 3D · 1 Image'
        )),
    )

@spaces_api.GPU(duration=60)
def _gen_shape(
    caption=None,
    input_mode='single',
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    generation_uid=None,
):
    tracking_enabled = generation_uid is not None and bool(str(generation_uid).strip())
    save_folder = None
    generation_created_at = None
    if tracking_enabled:
        generation_uid = normalize_generation_uid(generation_uid)
        save_folder = gen_save_folder(generation_uid=generation_uid)
        generation_created_at = utc_now_iso()
        write_generation_manifest(
            save_folder,
            schema_version=1,
            generation_uid=generation_uid,
            status='processing',
            stage='request_received',
            progress=GENERATION_STAGE_PROGRESS['request_received'],
            created_at=generation_created_at,
            updated_at=generation_created_at,
            storage_folder=generation_storage_path(save_folder),
            events=[{
                'stage': 'request_received',
                'message': GENERATION_STAGE_MESSAGES['request_received'],
                'at': generation_created_at,
                'progress': GENERATION_STAGE_PROGRESS['request_received'],
            }],
        )
        update_generation_stage(save_folder, 'validating_input')

    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Please provide either a caption or an image.")
    if MV_MODE:
        if input_mode == 'single':
            if image is None:
                raise gr.Error("Tab 1 ẢNH cần một ảnh chính diện của vật thể.")
            image = {'front': image}
        elif input_mode == 'four':
            multi_view_images = {
                'front': mv_image_front,
                'left': mv_image_left,
                'back': mv_image_back,
                'right': mv_image_right,
            }
            missing_views = [
                name.title() for name, view in multi_view_images.items() if view is None
            ]
            if missing_views:
                raise gr.Error(
                    "Tab 4 ẢNH cần đủ Front, Back, Left và Right. Còn thiếu: "
                    + ", ".join(missing_views)
                )
            image = multi_view_images
        else:
            raise gr.Error("Chế độ ảnh không hợp lệ. Hãy tải lại trang Web UI.")
    else:
        input_mode = 'single'

    if tracking_enabled:
        update_generation_stage(
            save_folder,
            'input_validated',
            input_mode=input_mode,
        )

    seed = int(randomize_seed_fn(seed, randomize_seed))

    octree_resolution = int(octree_resolution)
    if caption:
        print('prompt is', caption)
    if save_folder is None:
        save_folder = gen_save_folder()
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'input_mode': input_mode,
            'views_used': list(image) if isinstance(image, dict) else ['image'],
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    if tracking_enabled:
        stats['generation'] = {
            'uid': generation_uid,
            'status': 'processing',
            'created_at': generation_created_at,
            'storage_folder': generation_storage_path(save_folder),
        }
    time_meta = {}

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception:
            raise gr.Error("Text to 3D is disable. \
            Please enable it by `python gradio_app.py --enable_t23d`.")
        time_meta['text2image'] = time.time() - start_time

    if tracking_enabled:
        input_files = save_generation_inputs(save_folder, image)
        stats['generation']['inputs'] = input_files
        update_generation_stage(
            save_folder,
            'input_saved',
            model=stats['model'],
            params=stats['params'],
            inputs=input_files,
        )

    if tracking_enabled:
        update_generation_stage(save_folder, 'preprocessing_input')

    if MV_MODE:
        if not isinstance(image, dict):
            raise gr.Error("Multi-view input must contain named images.")
        start_time = time.time()
        for k, v in image.items():
            if v is None:
                raise gr.Error(f"Missing image for the {k} view.")
            if check_box_rembg or v.mode == "RGB":
                img = get_background_remover()(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
        if image is None or isinstance(image, dict):
            raise gr.Error("Please provide a valid input image.")
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            image = get_background_remover()(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time

    if tracking_enabled:
        update_generation_stage(save_folder, 'input_ready')

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    diffusion_clock: dict[str, float | None] = {
        'started_at': None,
        'last_step_at': None,
    }
    total_diffusion_steps = max(1, int(steps))

    def report_stage_safely(stage, message=None, progress=None, details=None, **updates):
        if not tracking_enabled:
            return
        try:
            update_generation_stage(
                save_folder,
                stage,
                message=message,
                progress=progress,
                event_details=details,
                **updates,
            )
        except Exception as stage_error:
            logger.warning(
                "Could not report generation stage %s for %s: %s",
                stage,
                generation_uid,
                stage_error,
            )

    def on_pipeline_stage(stage, details):
        details = dict(details or {})
        now = time.perf_counter()
        message = GENERATION_STAGE_MESSAGES.get(stage, stage)
        stage_progress = None

        if stage == 'prepare_conditioning':
            message = (
                f"Preparing {details.get('view_count', 1)} input view(s) "
                "for model conditioning"
            )
        elif stage == 'encode_conditioning':
            message = (
                "Encoding vision features from tensor "
                f"{details.get('image_shape', [])} ({details.get('dtype', '-')})"
            )
        elif stage == 'conditioning_ready':
            message = (
                "Vision conditioning ready for batch_size="
                f"{details.get('batch_size', 1)}"
            )
        elif stage == 'prepare_timestep_schedule':
            message = (
                f"Building {details.get('scheduler', 'diffusion')} schedule "
                f"with {details.get('requested_steps', total_diffusion_steps)} steps"
            )
        elif stage == 'latents_initialized':
            message = (
                f"Initialized latent noise {details.get('latent_shape', [])} "
                f"on {details.get('device', 'cuda')} as {details.get('dtype', '-')}"
            )
        elif stage == 'diffusion_started':
            diffusion_clock['started_at'] = now
            diffusion_clock['last_step_at'] = now
            message = f"Starting {total_diffusion_steps} real diffusion steps on CUDA"
        elif stage == 'diffusion_completed':
            started_at = diffusion_clock['started_at']
            sampling_seconds = (
                now - started_at
                if started_at is not None
                else 0.0
            )
            details['sampling_seconds'] = round(sampling_seconds, 3)
            message = f"Diffusion sampling completed in {sampling_seconds:.2f}s"
        elif stage == 'vae_decoding':
            message = (
                f"ShapeVAE decoding latent {details.get('latent_shape', [])} "
                f"({details.get('dtype', '-')})"
            )
        elif stage == 'volume_decoding':
            message = (
                f"Starting {details.get('decoder', 'volume decoder')} · "
                f"octree={details.get('octree_resolution', octree_resolution)} "
                f"chunks={details.get('num_chunks', num_chunks)} "
            )
        elif stage == 'volume_decoding_progress':
            volume_percent = float(details.get('volume_percent', 0.0))
            stage_progress = 79.0 + (volume_percent / 100.0 * 10.0)
            message = (
                f"Volume chunk {details.get('chunk', '-')}/{details.get('total_chunks', '-')} · "
                f"points {details.get('processed_points', '-')}/{details.get('total_points', '-')} · "
                f"{volume_percent:.1f}% · ETA {float(details.get('eta_seconds', 0.0)):.1f}s"
            )
        elif stage == 'volume_decoding_completed':
            message = (
                f"Dense volume ready from {details.get('decoder', 'decoder')} · "
                f"grid={details.get('grid_shape', [])}"
            )
        elif stage == 'surface_extraction':
            message = (
                "Running marching cubes with "
                f"{details.get('extractor', 'surface extractor')}"
            )
        elif stage == 'surface_extraction_completed':
            message = (
                "Surface extraction completed · mesh_count="
                f"{details.get('mesh_count', '-') }"
            )

        logger.info("[generation %s] %s", generation_uid, message)
        report_stage_safely(
            stage,
            message=message,
            progress=stage_progress,
            details=details,
            pipeline_stage=details,
        )

    def on_diffusion_step(step_idx, timestep, _outputs):
        now = time.perf_counter()
        started_at = diffusion_clock['started_at']
        if started_at is None:
            started_at = now
            diffusion_clock['started_at'] = started_at
        last_step_at = diffusion_clock['last_step_at']
        if last_step_at is None:
            last_step_at = started_at
            diffusion_clock['last_step_at'] = last_step_at

        step_number = min(total_diffusion_steps, int(step_idx) + 1)
        step_seconds = now - last_step_at
        elapsed_seconds = now - started_at
        diffusion_clock['last_step_at'] = now
        eta_seconds = (
            elapsed_seconds / max(1, step_number)
            * max(0, total_diffusion_steps - step_number)
        )
        diffusion_percent = step_number / total_diffusion_steps * 100.0
        overall_progress = 31.0 + (diffusion_percent / 100.0 * 43.0)

        try:
            timestep_value = (
                float(timestep.detach().float().item())
                if torch.is_tensor(timestep)
                else float(timestep)
            )
        except (TypeError, ValueError, RuntimeError):
            timestep_value = str(timestep)
        timestep_display = (
            f"{timestep_value:.4f}"
            if isinstance(timestep_value, (int, float))
            else timestep_value
        )

        allocated_gb = 0.0
        reserved_gb = 0.0
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

        details = {
            'step': step_number,
            'total_steps': total_diffusion_steps,
            'timestep': timestep_value,
            'diffusion_percent': round(diffusion_percent, 2),
            'step_seconds': round(step_seconds, 3),
            'elapsed_seconds': round(elapsed_seconds, 3),
            'eta_seconds': round(eta_seconds, 3),
            'vram_allocated_gb': round(allocated_gb, 3),
            'vram_reserved_gb': round(reserved_gb, 3),
        }
        message = (
            f"Step {step_number:02d}/{total_diffusion_steps:02d} · "
            f"t={timestep_display} · {step_seconds:.2f}s · "
            f"ETA {eta_seconds:.1f}s · VRAM {allocated_gb:.2f}/{reserved_gb:.2f} GB"
        )
        logger.info("[generation %s] %s", generation_uid, message)
        report_stage_safely(
            'diffusion_step',
            message=message,
            progress=overall_progress,
            details=details,
            diffusion=details,
        )

    if tracking_enabled:
        update_generation_stage(
            save_folder,
            'shape_generation',
            message=(
                f"Launching Hunyuan3D inference · steps={total_diffusion_steps} "
                f"guidance={guidance_scale} octree={octree_resolution} chunks={num_chunks}"
            ),
        )
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh',
        callback=on_diffusion_step if tracking_enabled else None,
        callback_steps=1,
        stage_callback=on_pipeline_stage if tracking_enabled else None,
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

    tmp_start = time.time()
    if tracking_enabled:
        update_generation_stage(save_folder, 'extracting_mesh')
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    if tracking_enabled:
        update_generation_stage(
            save_folder,
            'mesh_ready',
            number_of_faces=stats['number_of_faces'],
            number_of_vertices=stats['number_of_vertices'],
        )

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image.get('front', next(iter(image.values())))
    return mesh, main_image, save_folder, stats, seed

@spaces_api.GPU(duration=60)
def generation_all(
    caption=None,
    input_mode='single',
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption=caption,
        input_mode=input_mode,
        image=image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
    )
    path = export_mesh(mesh, save_folder, textured=False)
    

    print(path)
    print('='*40)

    # tmp_time = time.time()
    # mesh = floater_remove_worker(mesh)
    # mesh = degenerate_face_remove_worker(mesh)
    # logger.info("---Postprocessing takes %s seconds ---" % (time.time() - tmp_time))
    # stats['time']['postprocessing'] = time.time() - tmp_time

    tmp_time = time.time()
    _, _, face_reduce_worker = get_postprocessors()
    mesh = face_reduce_worker(mesh)

    # path = export_mesh(mesh, save_folder, textured=False, type='glb')
    path = export_mesh(mesh, save_folder, textured=False, type='obj') # 这样操作也会 core dump

    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time

    tmp_time = time.time()

    text_path = os.path.join(save_folder, 'textured_mesh.obj')
    path_textured = tex_pipeline(mesh_path=path, image_path=image, output_mesh_path=text_path, save_glb=False)
        
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time

    tmp_time = time.time()
    # Convert textured OBJ to GLB using obj2gltf with PBR support
    glb_path_textured = os.path.join(save_folder, 'textured_mesh.glb')
    quick_convert_with_obj2gltf(path_textured, glb_path_textured)

    logger.info("---Convert textured OBJ to GLB takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['convert textured OBJ to GLB'] = time.time() - tmp_time
    stats['time']['total'] = time.time() - start_time_0
    model_viewer_html_textured = build_model_viewer_html(save_folder, 
                                                         height=HTML_HEIGHT, 
                                                         width=HTML_WIDTH, textured=True)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        gr.update(value=glb_path_textured),
        model_viewer_html_textured,
        stats,
        seed,
    )

@spaces_api.GPU(duration=60)
def shape_generation(
    caption=None,
    input_mode='single',
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    request: gr.Request | None = None,
):
    start_time_0 = time.time()
    generation_uid = generation_uid_from_request(request)
    try:
        result = _gen_shape(
            caption=caption,
            input_mode=input_mode,
            image=image,
            mv_image_front=mv_image_front,
            mv_image_back=mv_image_back,
            mv_image_left=mv_image_left,
            mv_image_right=mv_image_right,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            octree_resolution=octree_resolution,
            check_box_rembg=check_box_rembg,
            num_chunks=num_chunks,
            randomize_seed=randomize_seed,
            generation_uid=generation_uid,
        )
    except GenerationUidConflictError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        mark_generation_failed(generation_uid, exc)
        raise

    mesh, image, save_folder, stats, seed = result
    stats['time']['total'] = time.time() - start_time_0
    completed_at = utc_now_iso()
    outputs = {
        'mesh': 'white_mesh.glb',
        'viewer': 'white_mesh.html',
    }
    stats['generation'].update({
        'status': 'completed',
        'completed_at': completed_at,
        'outputs': outputs,
    })
    mesh.metadata['extras'] = stats

    try:
        update_generation_stage(save_folder, 'exporting_glb')
        path = export_mesh(mesh, save_folder, textured=False)
        update_generation_stage(save_folder, 'building_preview')
        model_viewer_html = build_model_viewer_html(
            save_folder,
            height=HTML_HEIGHT,
            width=HTML_WIDTH,
        )
        update_generation_stage(
            save_folder,
            'completed',
            status='completed',
            completed_at=completed_at,
            outputs=outputs,
            stats=stats,
        )
    except Exception as exc:
        mark_generation_failed(generation_uid, exc)
        raise

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
    )


def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1–4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'

    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    title_parts = title.split(':', 1)
    brand_name = title_parts[0]
    workspace_title = title_parts[1].strip() if len(title_parts) == 2 else title
    runtime_device = {
        'cuda': 'CUDA',
        'cpu': 'CPU',
        'mps': 'MPS',
    }.get(str(args.device).lower(), 'LOCAL')
    runtime_dtype = {
        'float16': 'FP16',
        'bfloat16': 'BF16',
        'float32': 'FP32',
    }.get(str(getattr(args, 'dtype', 'float16')).lower(), 'FP16')
    runtime_label = f'{runtime_device} {runtime_dtype}'
    rtx_profile_action = """
            <button id="app-rtx-profile" class="app-topbar-button" type="button">
                <span class="ui-icon-slot" data-ui-icon="memory" aria-hidden="true"></span>
                <span>RTX 3090</span>
            </button>
    """ if MV_MODE and args.device == 'cuda' else ''

    title_html = f"""
    <header id="app-topbar" class="app-topbar">
        <div class="app-brand" aria-label="{brand_name}">
            <span class="app-brand-mark" aria-hidden="true">
                <img class="app-standard-logo" src="/favicon.ico" alt="" draggable="false">
            </span>
            <strong>{brand_name}</strong>
            <span class='app-version-badge'>v1.0</span>
        </div>
        <div class="app-title-block">
            <span class="app-title-mark" aria-hidden="true">
                <img class="app-standard-logo" src="/favicon.ico" alt="" draggable="false">
            </span>
            <div>
                <h1>{workspace_title}</h1>
                <p>Transform images into high-quality 3D assets with AI</p>
            </div>
        </div>
        <nav class="app-topbar-actions" aria-label="Application actions">
            <button id="app-api-docs" class="app-topbar-button" type="button">
                <span class="ui-icon-slot" data-ui-icon="code" aria-hidden="true"></span>
                <span>API Docs</span>
            </button>
            <button id="app-theme-settings" class="app-topbar-button app-topbar-icon-button" type="button" aria-label="Settings">
                <span class="ui-icon-slot" data-ui-icon="settings" aria-hidden="true"></span>
            </button>
            {rtx_profile_action}
        </nav>
    </header>
    """
    custom_css = """
    /* Shared appearance only; each component keeps its existing scroll behavior. */
    :root {
        --ui-scrollbar-size: 9px;
        --ui-scrollbar-track: #0b1020;
        --ui-scrollbar-thumb: #5f67ed;
        --ui-scrollbar-thumb-hover: #7379ff;
        --ui-scrollbar-thumb-active: #858aff;
        --ui-scrollbar-radius: 999px;
    }

    * {
        scrollbar-color: var(--ui-scrollbar-thumb) var(--ui-scrollbar-track);
        scrollbar-width: thin;
    }

    *::-webkit-scrollbar {
        width: var(--ui-scrollbar-size);
        height: var(--ui-scrollbar-size);
    }

    *::-webkit-scrollbar-track,
    *::-webkit-scrollbar-corner {
        background: var(--ui-scrollbar-track);
        border-radius: var(--ui-scrollbar-radius);
        margin: 14px 0;
    }

    *::-webkit-scrollbar-thumb {
        background: var(--ui-scrollbar-thumb);
        background-clip: padding-box;
        border: 2px solid var(--ui-scrollbar-track);
        border-radius: var(--ui-scrollbar-radius);
        min-height: 48px;
    }

    *::-webkit-scrollbar-thumb:hover {
        background: var(--ui-scrollbar-thumb-hover);
        background-clip: padding-box;
    }

    *::-webkit-scrollbar-thumb:active {
        background: var(--ui-scrollbar-thumb-active);
        background-clip: padding-box;
    }

    .gradio-container {
        --ui-font: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        --ui-mono: "Cascadia Code", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        --body-background-fill: #080b16;
        --block-background-fill: #101522;
        --background-fill-secondary: #151b2b;
        --block-border-color: #252d40;
        --border-color-primary: #313b53;
        --body-text-color: #f4f6fb;
        --body-text-color-subdued: #9099ad;
        --input-background-fill: #0d1220;
        --primary-500: #6670ff;
        --primary-600: #5862f2;
        --ui-bg: #080b16;
        --ui-surface: #101522;
        --ui-surface-muted: #151b2b;
        --ui-border: #252d40;
        --ui-border-strong: #313b53;
        --ui-text: #f4f6fb;
        --ui-muted: #9099ad;
        --ui-primary: #6670ff;
        --ui-primary-hover: #5862f2;
        --ui-success: #51cf66;
        --ui-warning: #f59f00;
        --ui-danger: #ff6b6b;
        --ui-space-1: 4px;
        --ui-space-2: 8px;
        --ui-space-3: 12px;
        --ui-space-4: 16px;
        --ui-space-5: 20px;
        --ui-space-6: 24px;
        --ui-radius-sm: 6px;
        --ui-radius-md: 8px;
        --ui-radius-lg: 12px;
        --ui-radius-xl: 16px;
        --ui-radius-pill: 999px;
        --ui-control-height: 40px;
        --ui-panel-header-height: 44px;
        --ui-stage-height: 820px;
        --ui-workspace-height: 900px;
        font-family: var(--ui-font);
        max-width: none !important;
        padding: 20px 24px 30px !important;
    }

    #generation-console-panel {
        background: transparent !important;
        border: 0 !important;
        min-width: 320px;
        padding: 0 !important;
    }

    #generation-console-panel .html-container {
        height: 100%;
        overflow: visible;
        padding: 0;
    }

    .generation-console {
        --console-accent: #5c7cfa;
        --console-border: #30343d;
        --console-green: #51cf66;
        --console-muted: #8b93a7;
        background: #0b0d11;
        border: 1px solid var(--console-border);
        border-radius: 10px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
        color: #d8dee9;
        display: flex;
        flex-direction: column;
        font-family: "Cascadia Code", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        height: 690px;
        min-height: 520px;
        overflow: hidden;
    }

    .generation-console-windowbar {
        align-items: center;
        background: #151820;
        border-bottom: 1px solid var(--console-border);
        display: flex;
        gap: 11px;
        min-height: 54px;
        padding: 0 14px;
    }

    .generation-console-dots {
        display: flex;
        gap: 6px;
    }

    .generation-console-dots i {
        border-radius: 999px;
        display: block;
        height: 8px;
        width: 8px;
    }

    .generation-console-dots i:nth-child(1) { background: #ff6b6b; }
    .generation-console-dots i:nth-child(2) { background: #ffd43b; }
    .generation-console-dots i:nth-child(3) { background: #51cf66; }

    .generation-console-title {
        flex: 1;
        min-width: 0;
    }

    .generation-console-title strong,
    .generation-console-title span {
        display: block;
    }

    .generation-console-title strong {
        color: #f1f3f5;
        font-family: var(--font);
        font-size: 13px;
        line-height: 1.25;
    }

    .generation-console-title span {
        color: var(--console-muted);
        font-size: 9px;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }

    .generation-console-status {
        align-items: center;
        background: rgba(139, 147, 167, 0.12);
        border: 1px solid rgba(139, 147, 167, 0.28);
        border-radius: 999px;
        color: #adb5bd;
        display: inline-flex;
        font-size: 9px;
        font-weight: 800;
        gap: 6px;
        letter-spacing: 0.06em;
        padding: 5px 8px;
    }

    .generation-console-status::before {
        background: currentColor;
        border-radius: 999px;
        content: "";
        height: 6px;
        width: 6px;
    }

    .generation-console[data-state="running"] .generation-console-status {
        background: rgba(81, 207, 102, 0.1);
        border-color: rgba(81, 207, 102, 0.32);
        color: var(--console-green);
    }

    .generation-console[data-state="running"] .generation-console-status::before {
        animation: generation-console-pulse 1.2s ease-in-out infinite;
        box-shadow: 0 0 0 4px rgba(81, 207, 102, 0.1);
    }

    .generation-console[data-state="completed"] .generation-console-status {
        background: rgba(81, 207, 102, 0.12);
        border-color: rgba(81, 207, 102, 0.35);
        color: var(--console-green);
    }

    .generation-console[data-state="failed"] .generation-console-status {
        background: rgba(255, 107, 107, 0.12);
        border-color: rgba(255, 107, 107, 0.35);
        color: #ff8787;
    }

    .generation-console-jobbar {
        align-items: center;
        background: #10131a;
        border-bottom: 1px solid #242832;
        display: flex;
        gap: 8px;
        min-height: 44px;
        padding: 0 14px;
    }

    .generation-console-jobbar > span:first-child {
        color: #74c0fc;
        font-size: 12px;
    }

    .generation-console-job {
        color: #aab2c3;
        flex: 1;
        font-size: 10px;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .generation-console-mode {
        background: #1a1e27;
        border: 1px solid #303642;
        border-radius: 4px;
        color: #bac8ff;
        font-size: 9px;
        padding: 4px 6px;
        white-space: nowrap;
    }

    .generation-console-log {
        flex: 1;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
        padding: 14px 12px 18px;
    }

    .generation-console-line {
        align-items: start;
        display: grid;
        font-size: 10px;
        gap: 7px;
        grid-template-columns: 54px 44px minmax(0, 1fr);
        line-height: 1.55;
        margin-bottom: 7px;
    }

    .generation-console-time {
        color: #596273;
        user-select: none;
    }

    .generation-console-level {
        color: #74c0fc;
        font-weight: 800;
        user-select: none;
    }

    .generation-console-message {
        color: #c7cedb;
        overflow-wrap: anywhere;
    }

    .generation-console-line[data-kind="command"] .generation-console-level,
    .generation-console-line[data-kind="command"] .generation-console-message {
        color: #b197fc;
    }

    .generation-console-line[data-kind="success"] .generation-console-level,
    .generation-console-line[data-kind="success"] .generation-console-message {
        color: #69db7c;
    }

    .generation-console-line[data-kind="error"] .generation-console-level,
    .generation-console-line[data-kind="error"] .generation-console-message {
        color: #ff8787;
    }

    .generation-console-line[data-kind="muted"] .generation-console-level,
    .generation-console-line[data-kind="muted"] .generation-console-message {
        color: #70798c;
    }

    .generation-console-progress-wrap {
        background: #10131a;
        border-top: 1px solid #242832;
        padding: 11px 14px 10px;
    }

    .generation-console-progress-meta {
        align-items: center;
        color: #828b9e;
        display: flex;
        font-size: 9px;
        justify-content: space-between;
        margin-bottom: 7px;
    }

    .generation-console-progress-track {
        background: #242832;
        border-radius: 999px;
        height: 4px;
        overflow: hidden;
    }

    .generation-console-progress-bar {
        background: linear-gradient(90deg, #4263eb, #748ffc, #4dabf7);
        border-radius: inherit;
        height: 100%;
        position: relative;
        transition: width 420ms ease;
        width: 0%;
    }

    .generation-console[data-state="running"] .generation-console-progress-bar::after {
        animation: generation-console-scan 1.4s linear infinite;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        content: "";
        inset: 0;
        position: absolute;
        transform: translateX(-100%);
    }

    .generation-console[data-state="failed"] .generation-console-progress-bar {
        background: #fa5252;
    }

    .generation-console-footer {
        align-items: center;
        background: #151820;
        border-top: 1px solid var(--console-border);
        color: #6f788b;
        display: flex;
        font-size: 9px;
        justify-content: space-between;
        min-height: 32px;
        padding: 0 14px;
    }

    .generation-console-footer strong {
        color: #748ffc;
        font-weight: 700;
    }

    @keyframes generation-console-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    @keyframes generation-console-scan {
        to { transform: translateX(100%); }
    }

    @media (max-width: 1500px) {
        .gradio-container {
            max-width: 100% !important;
        }

        .generation-console {
            height: 620px;
        }
    }

    @media (max-width: 1080px) {
        #generation-console-panel {
            min-width: min(100%, 420px);
        }

        .generation-console {
            height: 430px;
            min-height: 430px;
        }
    }

    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }

    #prompt-mode-tabs button[role="tab"] {
        border: 1px solid var(--block-border-color);
        border-radius: 10px;
        flex: 1;
        font-size: 14px;
        font-weight: 700;
        min-height: 44px;
        padding: 10px 14px;
    }

    #prompt-mode-tabs button[role="tab"][aria-selected="true"] {
        background: var(--primary-500);
        border-color: var(--primary-500);
        color: white;
    }

    .input-mode-guide {
        align-items: center;
        background: linear-gradient(135deg, rgba(63, 81, 181, 0.14), rgba(33, 150, 243, 0.06));
        border: 1px solid rgba(99, 130, 255, 0.35);
        border-radius: 12px;
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
        padding: 12px 14px;
    }

    .input-mode-number {
        align-items: center;
        background: #4263eb;
        border-radius: 10px;
        color: white;
        display: flex;
        flex: 0 0 38px;
        font-size: 18px;
        font-weight: 800;
        height: 38px;
        justify-content: center;
    }

    .input-mode-copy strong,
    .input-mode-copy span {
        display: block;
    }

    .input-mode-copy strong {
        font-size: 14px;
        margin-bottom: 3px;
    }

    .input-mode-copy span {
        color: var(--body-text-color-subdued);
        font-size: 12px;
        line-height: 1.4;
    }

    footer .rtx3090-footer-trigger {
        align-items: center;
        background: transparent;
        border: 0;
        color: var(--body-text-color-subdued);
        cursor: pointer;
        display: inline-flex;
        font: inherit;
        gap: 5px;
        padding: 0 2px;
        white-space: nowrap;
    }

    footer .rtx3090-footer-trigger:hover {
        color: var(--body-text-color);
        text-decoration: underline;
    }

    footer .rtx3090-footer-icon {
        color: #748ffc;
        font-size: 12px;
    }

    footer .rtx3090-footer-divider {
        align-items: center;
        color: var(--body-text-color-subdued);
        display: inline-flex;
        margin: 0 9px;
    }

    /* Match Gradio's Settings and API Docs inset frame to the RTX modal. */
    .api-docs {
        box-sizing: border-box !important;
        padding: 16px 18px !important;
    }

    .api-docs > .api-docs-wrap {
        border-radius: 18px !important;
        height: 100%;
        max-height: 100%;
    }

    body.rtx3090-modal-open {
        overflow: hidden;
    }

    #rtx3090-modal {
        --rtx-accent: #655dff;
        --rtx-accent-strong: #5267f7;
        --rtx-border: #2c354a;
        --rtx-card: rgba(20, 27, 44, 0.92);
        --rtx-card-muted: rgba(25, 33, 53, 0.94);
        --rtx-muted: #a8b0c3;
        --rtx-panel: #0d1321;
        align-items: stretch;
        backdrop-filter: blur(8px);
        background: rgba(2, 5, 12, 0.7);
        box-sizing: border-box;
        display: none !important;
        flex-direction: row !important;
        height: 100dvh !important;
        inset: 0;
        justify-content: flex-end;
        margin: 0 !important;
        max-width: none !important;
        padding: 16px 18px !important;
        position: fixed !important;
        width: 100vw !important;
        z-index: var(--layer-top, 2147483647);
    }

    #rtx3090-modal.rtx-open {
        display: flex !important;
    }

    #rtx3090-modal > .rtx3090-modal-panel {
        background:
            radial-gradient(circle at 46% -8%, rgba(83, 91, 220, 0.11), transparent 32%),
            linear-gradient(180deg, #101625 0%, var(--rtx-panel) 100%);
        border: 1px solid var(--rtx-border) !important;
        border-radius: 18px;
        box-shadow: 0 28px 86px rgba(0, 0, 0, 0.48);
        box-sizing: border-box;
        flex: 0 0 min(940px, calc(100vw - 36px)) !important;
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        gap: 0 !important;
        height: calc(100dvh - 32px);
        max-height: calc(100dvh - 32px);
        max-width: min(940px, calc(100vw - 36px)) !important;
        overflow-x: hidden;
        overflow-y: auto;
        overscroll-behavior-y: contain;
        padding: 0 28px 34px;
        scrollbar-gutter: stable;
        width: min(940px, calc(100vw - 36px)) !important;
    }

    #rtx3090-modal > .rtx3090-modal-panel > * {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        box-sizing: border-box;
        flex-shrink: 0 !important;
        max-width: 100%;
        min-width: 0;
        padding: 0 !important;
        width: 100%;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .block > .html-container,
    #rtx3090-modal > .rtx3090-modal-panel > .block > .html-container.padding {
        padding: 0 !important;
    }

    #rtx3090-modal .html-container {
        overflow: visible !important;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-modal-header-block {
        background: #101625 !important;
        margin: 0 -28px !important;
        max-width: none !important;
        overflow: visible !important;
        position: sticky;
        top: 0;
        width: calc(100% + 56px) !important;
        z-index: 5;
    }

    #rtx3090-modal > .rtx3090-modal-panel
    > .rtx3090-modal-header-block > .html-container.padding {
        overflow: visible !important;
        padding: 0 !important;
    }

    .rtx3090-modal-header {
        align-items: center;
        background: rgba(16, 22, 37, 0.98);
        border-bottom: 1px solid var(--rtx-border);
        box-sizing: border-box;
        display: flex;
        gap: 16px;
        justify-content: space-between;
        margin: 0;
        min-height: 72px;
        padding: 18px 20px 18px 28px;
        position: relative;
    }

    .rtx3090-header-main,
    .rtx3090-header-actions {
        align-items: center;
        display: flex;
    }

    .rtx3090-header-main {
        gap: 9px;
        min-width: 0;
    }

    .rtx3090-header-icon {
        align-items: center;
        color: #f7f8ff;
        display: inline-flex;
        flex: 0 0 auto;
        justify-content: center;
    }

    .rtx3090-header-icon .ui-icon {
        height: 20px;
        width: 20px;
    }

    .rtx3090-modal-header h2 {
        color: #f7f8ff;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.35;
        margin: 0;
        white-space: nowrap;
    }

    .rtx3090-header-scope {
        color: #837dff;
        font-size: 16px;
        font-weight: 520;
        white-space: nowrap;
    }

    .rtx3090-header-actions {
        flex: 0 0 auto;
        gap: 10px;
    }

    .rtx3090-verified {
        align-items: center;
        background: rgba(35, 187, 109, 0.12);
        border: 1px solid rgba(42, 211, 124, 0.24);
        border-radius: 6px;
        color: #45d58e !important;
        display: inline-flex;
        font-size: 13px;
        font-weight: 650;
        gap: 5px;
        margin-right: 30px;
        padding: 5px 8px;
        white-space: nowrap;
    }

    .rtx3090-verified-dot {
        background: #35d07f;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(53, 208, 127, 0.12);
        height: 7px;
        width: 7px;
    }

    .rtx3090-preset-count {
        font-size: 16px;
        white-space: nowrap;
    }

    .rtx3090-preset-count b {
        color: #827cff;
        font-weight: 700;
    }

    #rtx3090-modal-close {
        align-items: center;
        background: transparent;
        border: 0;
        border-left: 1px solid var(--rtx-border);
        border-radius: 0;
        box-sizing: border-box;
        color: #eef0fa;
        cursor: pointer;
        display: flex;
        flex: 0 0 52px;
        height: 30px;
        justify-content: center;
        line-height: 1;
        margin: 0 0 0 4px !important;
        min-width: 52px;
        padding: 0 0 0 18px;
        width: 52px;
    }

    #rtx3090-modal-close .rtx3090-close-icon {
        color: inherit;
        display: block;
        fill: currentColor;
        flex: 0 0 16px;
        height: 16px;
        width: 16px;
    }

    #rtx3090-modal-close:hover {
        background: transparent;
        color: #8d88ff;
    }

    #rtx3090-modal-close:focus-visible {
        outline: 2px solid var(--color-accent, var(--primary-500));
        outline-offset: -2px;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-intro-block {
        margin-top: 11px !important;
    }

    .rtx3090-api-intro {
        padding: 0;
    }

    .rtx3090-api-intro p {
        color: #d5d9e6;
        font-size: 15px;
        line-height: 1.5;
        margin: 17px 0 0;
        max-width: 820px;
    }

    .rtx3090-api-intro p strong {
        color: #817aff;
        font-weight: 700;
    }

    .rtx3090-context-tabs {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 23px;
    }

    .rtx3090-context-tabs span {
        align-items: center;
        background: rgba(17, 23, 38, 0.75);
        border: 1px solid #313a50;
        border-radius: 7px;
        color: #c2c8d8;
        display: inline-flex;
        font-size: 13px;
        gap: 7px;
        min-height: 34px;
        padding: 6px 12px;
    }

    .rtx3090-context-tabs span.active {
        border-color: #736cff;
        box-shadow: inset 0 0 0 1px rgba(115, 108, 255, 0.3);
        color: #f5f6ff;
    }

    .rtx3090-section-heading {
        color: #f4f6fb;
        display: block;
        margin: 0;
    }

    .rtx3090-section-heading b {
        display: block;
        font-size: 16px;
        font-weight: 720;
        line-height: 1.5;
    }

    .rtx3090-section-heading span {
        color: var(--rtx-muted);
        display: block;
        font-size: 14px;
        line-height: 1.5;
        margin-top: 3px;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-section-one {
        margin-top: 21px !important;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-machine-block {
        margin-top: 20px !important;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-section-two {
        margin-top: 22px !important;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-profiles-block {
        margin-top: 14px !important;
    }

    .rtx3090-machine-strip {
        align-items: center;
        background: linear-gradient(135deg, rgba(21, 29, 48, 0.98), rgba(17, 24, 40, 0.96));
        border: 1px solid var(--rtx-border);
        border-radius: 13px;
        box-sizing: border-box;
        display: flex;
        gap: 14px;
        min-height: 96px;
        padding: 16px 20px;
    }

    .rtx3090-machine-badge {
        background: linear-gradient(145deg, #6258f4, #745eff);
        border: 1px solid rgba(167, 160, 255, 0.38);
        border-radius: 10px;
        box-shadow: 0 8px 22px rgba(78, 67, 222, 0.24);
        color: white;
        box-sizing: border-box;
        flex: 0 0 52px;
        font-size: 13px;
        font-weight: 800;
        padding: 10px 0;
        text-align: center;
        width: 52px;
    }

    .rtx3090-machine-copy {
        flex: 1 1 auto;
        min-width: 0;
    }

    .rtx3090-machine-copy strong,
    .rtx3090-machine-copy span {
        display: block;
    }

    .rtx3090-machine-copy strong {
        color: #f5f6fb;
        font-size: 17px;
        line-height: 1.4;
        margin-bottom: 4px;
    }

    .rtx3090-machine-copy span {
        color: var(--rtx-muted);
        font-size: 14px;
        line-height: 1.5;
    }

    .rtx3090-machine-check {
        align-items: center;
        border: 2px solid #32cf80;
        border-radius: 50%;
        color: #32cf80;
        display: inline-flex !important;
        flex: 0 0 24px;
        height: 24px;
        justify-content: center;
        width: 24px;
    }

    .rtx3090-machine-check .ui-icon {
        height: 14px;
        stroke-width: 3;
        width: 14px;
    }

    .rtx3090-profile-grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .rtx3090-profile-card {
        background: linear-gradient(135deg, rgba(24, 28, 38, 0.98), rgba(19, 23, 32, 0.98));
        border: 1px solid #3a3f4d;
        border-radius: 13px;
        box-sizing: border-box;
        cursor: pointer;
        min-height: 152px;
        outline: none;
        padding: 15px 15px 13px;
        transition:
            border-color 160ms ease,
            box-shadow 160ms ease,
            background 160ms ease,
            opacity 160ms ease;
        user-select: none;
    }

    .rtx3090-profile-card.is-selected {
        background: linear-gradient(135deg, rgba(22, 42, 36, 0.98), rgba(17, 30, 29, 0.98));
        border-color: #45d58e;
        box-shadow:
            inset 0 0 0 1px rgba(69, 213, 142, 0.24),
            0 10px 24px rgba(24, 150, 91, 0.14);
    }

    .rtx3090-profile-card:not(.is-selected) {
        opacity: 0.78;
    }

    .rtx3090-profile-card:not(.is-selected):hover {
        border-color: #555b6b;
        opacity: 0.92;
    }

    .rtx3090-profile-card:focus-visible {
        border-color: #8b84ff;
        box-shadow: 0 0 0 3px rgba(113, 104, 255, 0.24);
    }

    .rtx3090-profile-card.is-selected:focus-visible {
        border-color: #58dda0;
        box-shadow:
            inset 0 0 0 1px rgba(69, 213, 142, 0.24),
            0 0 0 3px rgba(53, 208, 127, 0.2);
    }

    .rtx3090-profile-heading {
        align-items: center;
        display: flex;
        gap: 12px;
        justify-content: space-between;
    }

    .rtx3090-profile-card h3 {
        color: #f4f6fb;
        font-size: 16px;
        font-weight: 680;
        line-height: 1.4;
        margin: 0;
    }

    .rtx3090-profile-card.is-selected h3 {
        color: #ffffff;
        font-weight: 780;
    }

    .rtx3090-profile-card:not(.is-selected) h3 {
        color: #c6cad3;
    }

    .rtx3090-profile-selector {
        align-items: center;
        border: 2px solid #687086;
        border-radius: 50%;
        color: #fff;
        display: inline-flex;
        flex: 0 0 22px;
        height: 22px;
        justify-content: center;
        width: 22px;
    }

    .rtx3090-profile-card.is-selected .rtx3090-profile-selector {
        background: linear-gradient(145deg, #22b96e, #45d58e);
        border-color: #45d58e;
        box-shadow: 0 0 0 3px rgba(53, 208, 127, 0.12);
    }

    .rtx3090-profile-selector::after {
        content: "";
    }

    .rtx3090-profile-card.is-selected .rtx3090-profile-selector::after {
        border-color: #fff;
        border-style: solid;
        border-width: 0 2px 2px 0;
        content: "";
        height: 8px;
        transform: rotate(45deg) translate(-1px, -1px);
        width: 4px;
    }

    .rtx3090-profile-card p {
        color: var(--rtx-muted);
        font-size: 13px;
        line-height: 1.5;
        margin: 5px 0 13px;
    }

    .rtx3090-profile-card:not(.is-selected) p {
        color: #8e95a5;
    }

    .rtx3090-profile-values {
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }

    .rtx3090-profile-values span {
        align-items: center;
        background: var(--rtx-card-muted);
        border-radius: 8px;
        color: var(--rtx-muted);
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 56px;
        padding: 7px 5px;
        text-align: center;
    }

    .rtx3090-profile-values b {
        color: #f4f6fb;
        display: block;
        font-size: 14px;
        line-height: 1.25;
        margin-bottom: 3px;
    }

    .rtx3090-profile-values small {
        color: var(--rtx-muted);
        font-size: 12px;
        line-height: 1.2;
    }

    .rtx3090-profile-card:not(.is-selected) .rtx3090-profile-values span {
        background: #1b1f2a;
    }

    .rtx3090-profile-card:not(.is-selected) .rtx3090-profile-values b {
        color: #c8ccd6;
    }

    #rtx3090-modal .rtx-preset-actions {
        gap: 16px;
        margin-top: 19px !important;
    }

    #rtx3090-modal .rtx-preset-actions button {
        background: linear-gradient(135deg, #555761, #474952) !important;
        border: 1px solid #5d606b !important;
        border-radius: 8px;
        box-shadow: none !important;
        color: #f3f4f8 !important;
        font-size: 16px;
        font-weight: 700;
        min-height: 45px;
        opacity: 0.78;
        transition:
            background 160ms ease,
            border-color 160ms ease,
            box-shadow 160ms ease,
            opacity 160ms ease;
    }

    #rtx3090-modal .rtx-preset-actions button:hover {
        border-color: #747784 !important;
        opacity: 0.94;
    }

    #rtx3090-modal .rtx-preset-actions button.rtx-preset-action-active {
        background: linear-gradient(135deg, #12834d 0%, #148850 100%) !important;
        border-color: #45d58e !important;
        box-shadow: 0 9px 20px rgba(25, 164, 98, 0.22) !important;
        opacity: 1;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-status-block {
        margin-top: 29px !important;
    }

    .rtx-preset-status {
        background: linear-gradient(135deg, rgba(21, 29, 48, 0.98), rgba(17, 24, 40, 0.96));
        border: 1px solid var(--rtx-border);
        border-radius: 13px;
        box-sizing: border-box;
        min-height: 122px;
        padding: 15px;
    }

    .rtx-preset-status-heading {
        align-items: center;
        display: flex;
        gap: 14px;
        justify-content: space-between;
    }

    .rtx-preset-status-title {
        align-items: center;
        display: flex;
        font-size: 14px;
        font-weight: 720;
        gap: 9px;
        line-height: 1.4;
        min-width: 0;
    }

    .rtx-preset-status-check {
        align-items: center;
        background: #2abf75;
        border-radius: 50%;
        color: white;
        display: inline-flex;
        flex: 0 0 18px;
        height: 18px;
        justify-content: center;
        width: 18px;
    }

    .rtx-preset-status-check .ui-icon {
        height: 11px;
        stroke-width: 3;
        width: 11px;
    }

    .rtx-preset-current {
        background: rgba(117, 126, 151, 0.18);
        border: 1px solid rgba(144, 153, 178, 0.14);
        border-radius: 999px;
        color: #c6ccda !important;
        flex: 0 0 auto;
        font-size: 12px;
        font-weight: 650;
        line-height: 1;
        padding: 5px 9px;
    }

    .rtx-preset-values {
        display: grid;
        gap: 8px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin: 12px 0 0;
    }

    .rtx-preset-values span {
        align-items: center;
        background: var(--rtx-card-muted);
        border-radius: 8px;
        color: var(--rtx-muted);
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 56px;
        padding: 7px;
        text-align: center;
    }

    .rtx-preset-values b {
        color: #f4f6fb;
        display: block;
        font-size: 14px;
        line-height: 1.25;
        margin-bottom: 3px;
    }

    .rtx-preset-values small {
        color: var(--rtx-muted);
        font-size: 12px;
        line-height: 1.2;
    }

    #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-note-block {
        margin-top: 20px !important;
    }

    .rtx3090-modal-note {
        align-items: flex-start;
        background: linear-gradient(135deg, rgba(19, 27, 45, 0.94), rgba(16, 23, 39, 0.92));
        border: 1px solid var(--rtx-border);
        border-radius: 13px;
        box-sizing: border-box;
        color: var(--rtx-muted);
        display: flex;
        font-size: 14px;
        gap: 15px;
        line-height: 1.5;
        min-height: 100px;
        padding: 15px 18px;
    }

    .rtx3090-modal-note p {
        margin: 0;
    }

    .rtx3090-modal-note strong {
        color: #f2f4fa;
    }

    .rtx3090-note-icon {
        align-items: center;
        border: 2px solid #766eff;
        border-radius: 50%;
        color: #8079ff;
        display: inline-flex;
        flex: 0 0 22px;
        height: 22px;
        justify-content: center;
        margin-top: 1px;
        width: 22px;
    }

    .rtx3090-note-icon .ui-icon {
        height: 13px;
        stroke-width: 2.5;
        width: 13px;
    }

    @media (max-width: 900px) {
        .rtx3090-header-scope,
        .rtx3090-verified {
            display: none;
        }

        .rtx3090-modal-header h2 {
            font-size: 19px;
        }
    }

    @media (max-width: 767px) {
        #rtx3090-modal {
            padding: 0 !important;
        }

        #rtx3090-modal > .rtx3090-modal-panel {
            border-radius: 0 !important;
            flex-basis: 100vw !important;
            height: 100dvh;
            max-height: 100dvh;
            max-width: 100vw !important;
            padding: 0 18px 28px;
            width: 100vw !important;
        }

        #rtx3090-modal > .rtx3090-modal-panel > .rtx3090-modal-header-block {
            margin-left: -18px !important;
            margin-right: -18px !important;
            width: calc(100% + 36px) !important;
        }

        .rtx3090-modal-header {
            min-height: 64px;
            padding: 15px 8px 15px 18px;
        }

        .rtx3090-header-scope,
        .rtx3090-verified,
        .rtx3090-preset-count {
            display: none;
        }

        .rtx3090-modal-header h2 {
            font-size: 17px;
            white-space: normal;
        }

        .rtx3090-api-intro p {
            font-size: 14px;
        }

        .rtx3090-context-tabs {
            gap: 7px;
        }

        .rtx3090-context-tabs span {
            font-size: 12px;
            min-height: 32px;
            padding: 5px 9px;
        }

        .rtx3090-section-heading b {
            font-size: 15px;
        }

        .rtx3090-section-heading span {
            font-size: 13px;
        }

        .rtx3090-machine-strip {
            align-items: flex-start;
            padding: 15px;
        }

        .rtx3090-machine-strip strong {
            font-size: 14px;
        }

        .rtx3090-machine-check {
            display: none !important;
        }

        .rtx3090-profile-grid {
            grid-template-columns: 1fr;
        }

        .rtx3090-profile-values,
        .rtx-preset-values {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        #rtx3090-modal .rtx-preset-actions {
            flex-direction: column;
        }

        .rtx-preset-status-heading {
            align-items: flex-start;
        }

        .rtx3090-modal-note {
            font-size: 13px;
        }
    }

    /* Unified application shell */
    .app-hero {
        align-items: center;
        display: flex;
        gap: var(--ui-space-3);
        justify-content: center;
        margin: 2px auto var(--ui-space-5);
        min-height: 54px;
        text-align: left;
    }

    .app-hero-mark {
        align-items: center;
        background: linear-gradient(145deg, #4263eb, #6d8cff);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: var(--ui-radius-lg);
        box-shadow: 0 10px 28px rgba(66, 99, 235, 0.24);
        color: white;
        display: inline-flex;
        flex: 0 0 42px;
        height: 42px;
        justify-content: center;
        width: 42px;
    }

    .app-hero-copy h1 {
        color: var(--ui-text);
        font-size: clamp(22px, 1.55vw, 28px);
        font-weight: 750;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0;
    }

    .app-hero-copy p {
        color: var(--ui-muted);
        font-size: 12px;
        line-height: 1.4;
        margin: 4px 0 0;
    }

    #workspace-grid {
        align-items: start;
        display: grid !important;
        gap: var(--ui-space-5) !important;
        grid-template-columns: minmax(340px, 400px) minmax(540px, 1fr) minmax(330px, 400px);
        width: 100%;
    }

    #workspace-grid > .column {
        min-width: 0 !important;
        width: 100%;
    }

    #input-panel {
        background: color-mix(in srgb, var(--ui-surface) 94%, transparent);
        border: 1px solid var(--ui-border);
        border-radius: var(--ui-radius-xl);
        gap: var(--ui-space-3) !important;
        padding: var(--ui-space-3) !important;
    }

    #viewport-panel {
        gap: 0 !important;
    }

    /* One icon geometry across custom UI and Gradio-native controls. */
    .ui-icon {
        display: block;
        fill: none;
        flex: 0 0 auto;
        height: 16px;
        stroke: currentColor;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 1.5;
        width: 16px;
    }

    .app-hero-mark .ui-icon {
        height: 22px;
        width: 22px;
    }

    .ui-action-icon {
        height: 17px;
        margin-right: 2px;
        width: 17px;
    }

    #input-panel button svg,
    #mesh-download-card button svg,
    footer button svg,
    footer a svg {
        height: 16px !important;
        width: 16px !important;
    }

    /* Segmented input mode; shared headers for output and settings. */
    #prompt-mode-tabs .tab-nav {
        background: var(--ui-surface-muted);
        border: 1px solid var(--ui-border);
        border-radius: var(--ui-radius-lg);
        gap: var(--ui-space-1);
        padding: var(--ui-space-1);
    }

    #prompt-mode-tabs button[role="tab"] {
        background: transparent;
        border: 1px solid transparent !important;
        border-radius: var(--ui-radius-md);
        color: var(--ui-muted);
        font-size: 13px;
        font-weight: 650;
        min-height: var(--ui-control-height);
        padding: 8px 12px;
    }

    #prompt-mode-tabs button[role="tab"][aria-selected="true"] {
        background: var(--ui-primary);
        border-color: var(--ui-primary) !important;
        box-shadow: 0 5px 14px rgba(66, 99, 235, 0.22);
        color: white;
    }

    #output-tabs,
    #settings-tabs {
        min-width: 0;
    }

    #output-tabs {
        background: color-mix(in srgb, var(--ui-surface) 94%, transparent);
        border: 1px solid var(--ui-border);
        border-radius: var(--ui-radius-xl);
        overflow: hidden;
        padding: 0 var(--ui-space-3) var(--ui-space-3);
    }

    #output-tabs .tab-nav,
    #settings-tabs .tab-nav {
        align-items: stretch;
        background: transparent;
        border-bottom: 1px solid var(--ui-border);
        gap: var(--ui-space-1);
        min-height: var(--ui-panel-header-height);
        padding: 0;
    }

    #output-tabs .tab-nav button[role="tab"],
    #settings-tabs .tab-nav button[role="tab"] {
        background: transparent;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0;
        color: var(--ui-muted);
        font-size: 12px;
        font-weight: 650;
        min-height: var(--ui-panel-header-height);
        padding: 0 12px;
    }

    #output-tabs .tab-nav button[role="tab"][aria-selected="true"],
    #settings-tabs .tab-nav button[role="tab"][aria-selected="true"] {
        border-bottom-color: var(--ui-primary) !important;
        color: var(--ui-primary);
    }

    /* Upload and download surfaces */
    .input-mode-guide {
        background: color-mix(in srgb, var(--ui-primary) 10%, var(--ui-surface));
        border-color: color-mix(in srgb, var(--ui-primary) 45%, var(--ui-border));
        border-radius: var(--ui-radius-lg);
        gap: var(--ui-space-3);
        margin: var(--ui-space-3) 0;
        min-height: 64px;
        padding: var(--ui-space-3);
    }

    .input-mode-number {
        background: var(--ui-primary);
        border-radius: var(--ui-radius-md);
        flex-basis: 36px;
        font-size: 16px;
        height: 36px;
    }

    .input-mode-copy strong {
        font-size: 13px;
        font-weight: 700;
    }

    .input-mode-copy span {
        font-size: 11px;
        line-height: 1.45;
    }

    .mv-upload-row {
        display: grid !important;
        gap: var(--ui-space-3) !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .mv-upload-row > * {
        min-width: 0 !important;
        width: 100%;
    }

    #input-panel .ui-upload {
        background: var(--ui-surface-muted) !important;
        border: 1px solid var(--ui-border) !important;
        border-radius: var(--ui-radius-lg) !important;
        overflow: hidden;
    }

    #input-panel .ui-upload:hover {
        border-color: color-mix(in srgb, var(--ui-primary) 55%, var(--ui-border)) !important;
    }

    .generate-actions {
        gap: var(--ui-space-2) !important;
        margin-top: var(--ui-space-1);
    }

    #generate-3d-button {
        align-items: center;
        border-radius: var(--ui-radius-md) !important;
        box-shadow: 0 8px 18px rgba(66, 99, 235, 0.2);
        display: inline-flex;
        font-size: 13px;
        font-weight: 700;
        gap: var(--ui-space-2);
        justify-content: center;
        min-height: 44px;
    }

    #mesh-download-card {
        background: var(--ui-surface-muted);
        border: 1px solid var(--ui-border);
        border-radius: var(--ui-radius-lg);
        overflow: hidden;
    }

    #mesh-download-card .block {
        border: 0 !important;
        border-radius: 0 !important;
    }

    /* Fixed, repeatable form grid */
    #settings-tabs {
        margin-top: var(--ui-space-1);
    }

    #advanced-settings-form {
        gap: var(--ui-space-3) !important;
        padding-top: var(--ui-space-3);
    }

    .ui-control-row {
        display: grid !important;
        gap: var(--ui-space-3) !important;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .ui-control-row-checks {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }

    #advanced-settings-form .ui-control {
        background: var(--ui-surface-muted) !important;
        border: 1px solid var(--ui-border) !important;
        border-radius: var(--ui-radius-lg) !important;
        box-sizing: border-box;
        margin: 0 !important;
        min-height: 82px;
        min-width: 0 !important;
        padding: 10px 12px !important;
        width: 100%;
    }

    #advanced-settings-form .ui-control-checkbox {
        align-items: center;
        display: flex;
        min-height: var(--ui-control-height);
        padding: 8px 10px !important;
    }

    #advanced-settings-form .ui-control label {
        font-size: 12px;
        font-weight: 600;
        line-height: 1.35;
    }

    #advanced-settings-form input[type="number"],
    #advanced-settings-form input[type="text"] {
        min-height: 34px;
    }

    /* Viewer, statistics and console now share one visual system. */
    #output-tabs > .tab-wrapper {
        align-items: stretch;
        flex: 0 0 44px !important;
        height: 44px !important;
        min-height: 44px !important;
        padding: 0 !important;
    }

    #output-tabs > .tab-wrapper > .tab-container[role='tablist'] {
        align-items: stretch;
        border-bottom: 1px solid var(--ui-border);
        display: flex !important;
        min-height: 44px !important;
        padding: 0 4px !important;
    }

    #output-tabs > .tab-wrapper > .tab-container[role='tablist'] > button[role='tab'] {
        font-size: 12px;
        min-height: 44px !important;
        padding: 0 14px !important;
    }

    #output-tabs > .tabitem {
        padding: 0 !important;
    }

    #mesh-viewer,
    #mesh-export-viewer,
    #mesh-stats {
        background: transparent !important;
        border: 0 !important;
        border-radius: var(--ui-radius-lg) !important;
        margin-top: var(--ui-space-3);
        overflow: hidden;
        padding: 0 !important;
    }

    #mesh-viewer .html-container,
    #mesh-export-viewer .html-container {
        padding: 0 !important;
    }

    #mesh-viewer iframe,
    #mesh-export-viewer iframe {
        border-radius: var(--ui-radius-lg);
        display: block;
        width: 100%;
    }

    #generation-console-panel {
        min-width: 0;
    }

    .generation-console {
        border-color: var(--ui-border);
        border-radius: var(--ui-radius-xl);
        box-shadow: none;
        height: var(--ui-workspace-height);
        min-height: var(--ui-workspace-height);
    }

    .generation-console-windowbar {
        gap: var(--ui-space-2);
        min-height: var(--ui-panel-header-height);
        padding: 0 var(--ui-space-3);
    }

    .generation-console-title strong {
        font-size: 12px;
    }

    .generation-console-title span,
    .generation-console-status,
    .generation-console-mode,
    .generation-console-progress-meta,
    .generation-console-footer {
        font-size: 10px;
    }

    .generation-console-line {
        font-size: 10.5px;
        grid-template-columns: 56px 48px minmax(0, 1fr);
        line-height: 1.55;
    }

    /* Modal and footer consume the same radii, spacing and icon scale. */
    #rtx3090-modal > .rtx3090-modal-panel {
        gap: 0;
    }

    .rtx3090-profile-card,
    .rtx-preset-status,
    .rtx3090-modal-note,
    .rtx3090-machine-strip {
        border-radius: 13px;
    }

    .rtx3090-context-tabs .ui-icon {
        height: 13px;
        width: 13px;
    }

    footer .rtx3090-footer-trigger,
    footer button,
    footer a {
        align-items: center;
        gap: 5px;
    }

    footer .ui-icon {
        color: currentColor;
        height: 14px;
        width: 14px;
    }

    @media (max-width: 1650px) {
        .gradio-container {
            padding-left: 24px !important;
            padding-right: 24px !important;
        }

        #workspace-grid {
            gap: var(--ui-space-4) !important;
            grid-template-columns: minmax(320px, 360px) minmax(500px, 1fr) minmax(310px, 350px);
        }

        .generation-console {
            height: var(--ui-workspace-height);
            min-height: var(--ui-workspace-height);
        }
    }

    @media (max-width: 1360px) {
        #workspace-grid {
            grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
        }

        #generation-console-panel {
            grid-column: 1 / -1;
        }

        .generation-console {
            height: 520px;
            min-height: 520px;
        }
    }

    @media (max-width: 900px) {
        .gradio-container {
            padding: 14px 14px 76px !important;
        }

        .app-hero {
            justify-content: flex-start;
        }

        #workspace-grid {
            grid-template-columns: minmax(0, 1fr);
        }

        #generation-console-panel {
            grid-column: auto;
        }
    }

    @media (max-width: 560px) {
        .app-hero-mark {
            display: none;
        }

        .app-hero-copy h1 {
            font-size: 20px;
        }

        .mv-upload-row,
        .ui-control-row {
            grid-template-columns: minmax(0, 1fr);
        }

        #input-panel {
            border-radius: var(--ui-radius-lg);
            padding: 10px !important;
        }

        #output-tabs .tab-nav button[role="tab"] {
            font-size: 11px;
            padding-left: 7px;
            padding-right: 7px;
        }
    }

    /* Target dashboard composition: one navy system across every native block. */
    html,
    body {
        background: #080b16 !important;
        color-scheme: dark;
    }

    body::before {
        background:
            radial-gradient(circle at 54% -20%, rgba(87, 92, 255, 0.12), transparent 34%),
            linear-gradient(180deg, #090d19 0%, #070a13 100%);
        content: "";
        inset: 0;
        pointer-events: none;
        position: fixed;
        z-index: -1;
    }

    .gradio-container {
        background: transparent !important;
        color: var(--ui-text);
        min-height: 100vh;
    }

    .gradio-container main.app {
        max-width: none !important;
        padding: 0 !important;
        width: 100% !important;
    }

    .gradio-container .prose,
    .gradio-container label,
    .gradio-container span,
    .gradio-container p {
        color: inherit;
    }

    .app-topbar {
        align-items: center;
        display: grid;
        gap: 18px;
        grid-template-columns:
            clamp(320px, 19vw, 360px)
            minmax(600px, 1fr)
            clamp(384px, 22vw, 420px);
        margin: 0;
        min-height: 42px;
        width: 100%;
    }

    .gradio-container .block:has(#app-topbar),
    .gradio-container .html-container:has(#app-topbar) {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .app-brand,
    .app-title-block,
    .app-topbar-actions {
        align-items: center;
        display: flex;
        min-width: 0;
    }

    .app-brand {
        gap: 11px;
    }

    .app-brand strong {
        color: var(--ui-text);
        font-size: 20px;
        font-weight: 760;
        letter-spacing: -0.02em;
        white-space: nowrap;
    }

    .app-version-badge {
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(129, 140, 248, 0.42);
        border-radius: var(--ui-radius-pill);
        color: #c7cbff !important;
        font-size: 10px;
        font-weight: 700;
        line-height: 1;
        padding: 6px 8px;
        white-space: nowrap;
    }

    .app-brand-mark,
    .app-title-mark {
        align-items: center;
        background: transparent;
        border: 0;
        box-shadow: none;
        display: inline-flex;
        flex: 0 0 38px;
        height: 38px;
        justify-content: center;
        width: 38px;
    }

    .app-brand-mark .app-standard-logo,
    .app-title-mark .app-standard-logo {
        display: block;
        height: 30px;
        object-fit: contain;
        pointer-events: none;
        user-select: none;
        width: 30px;
    }

    .app-title-block {
        gap: 12px;
    }

    .app-title-block h1 {
        color: var(--ui-text);
        font-size: clamp(19px, 1.35vw, 24px);
        font-weight: 740;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .app-title-block p {
        color: var(--ui-muted) !important;
        font-size: 11px;
        line-height: 1.4;
        margin: 3px 0 0;
    }

    .app-topbar-actions {
        gap: 10px;
        justify-content: flex-end;
    }

    .app-topbar-button {
        align-items: center;
        background: #111625;
        border: 1px solid var(--ui-border);
        border-radius: 9px;
        box-sizing: border-box;
        color: #d8dcef;
        cursor: pointer;
        display: inline-flex;
        font-family: var(--ui-font);
        font-size: 12px;
        font-weight: 650;
        gap: 8px;
        height: 38px;
        justify-content: center;
        margin: 0 !important;
        padding: 0 13px;
        transition: border-color 160ms ease, background 160ms ease, color 160ms ease;
        white-space: nowrap;
    }

    .app-topbar-button:hover {
        background: #171d2d;
        border-color: #414b69;
        color: #fff;
    }

    .app-topbar-icon-button {
        padding: 0;
        width: 38px;
    }

    footer {
        display: none !important;
    }

    #workspace-grid {
        align-items: stretch;
        gap: 18px !important;
        grid-template-columns:
            clamp(320px, 19vw, 360px)
            minmax(600px, 1fr)
            clamp(384px, 22vw, 420px);
    }

    #input-panel,
    #output-tabs,
    #generation-console-panel {
        background: rgba(16, 21, 34, 0.96) !important;
        border: 1px solid var(--ui-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 18px 52px rgba(0, 0, 0, 0.12);
        box-sizing: border-box;
    }

    #input-panel {
        align-self: start;
        flex-flow: column nowrap !important;
        gap: 12px !important;
        height: var(--ui-workspace-height);
        max-height: var(--ui-workspace-height);
        min-height: var(--ui-workspace-height);
        overflow-x: hidden;
        overflow-y: auto;
        padding: 16px !important;
        overscroll-behavior: contain;
        scrollbar-gutter: stable;
    }

    #prompt-mode-tabs,
    #input-panel > .generate-actions,
    #settings-tabs {
        flex-shrink: 0 !important;
    }

    .panel-heading {
        align-items: center;
        color: var(--ui-text);
        display: flex;
        font-size: 14px;
        font-weight: 720;
        justify-content: space-between;
        min-height: 26px;
        padding: 0 4px;
    }

    .panel-heading span {
        color: var(--ui-muted) !important;
        font-size: 10px;
        font-weight: 550;
    }

    #prompt-mode-tabs,
    #prompt-mode-tabs > div,
    #prompt-mode-tabs .tabitem {
        min-width: 0 !important;
        width: 100% !important;
    }

    #prompt-mode-tabs {
        gap: 0 !important;
    }

    #prompt-mode-tabs .tab-nav {
        background: #171c2b;
        border-color: #222a3d;
        border-radius: 8px;
        min-height: 36px;
        padding: 3px;
    }

    #prompt-mode-tabs button[role="tab"] {
        border-radius: 6px;
        font-size: 11px;
        min-height: 32px;
        padding: 5px 8px;
    }

    #prompt-mode-tabs button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4f58de, #6b70f7);
        box-shadow: 0 5px 16px rgba(61, 67, 210, 0.2);
    }

    .input-mode-guide {
        background: linear-gradient(135deg, rgba(82, 88, 225, 0.1), rgba(17, 22, 37, 0.82));
        border-color: #30395a;
        border-radius: 9px;
        margin: 10px 0;
        min-height: 48px;
        padding: 9px 10px;
    }

    .input-mode-number {
        background: rgba(101, 110, 255, 0.16);
        border: 1px solid rgba(111, 120, 255, 0.42);
        border-radius: 999px;
        color: #9198ff;
        flex-basis: 30px;
        height: 30px;
        width: 30px;
    }

    .input-mode-number .ui-icon {
        height: 15px;
        width: 15px;
    }

    .input-mode-copy strong {
        font-size: 11px;
    }

    .input-mode-copy span {
        color: var(--ui-muted) !important;
        font-size: 9.5px;
        line-height: 1.4;
    }

    .mv-upload-row {
        gap: 10px !important;
        width: 100% !important;
    }

    #input-panel .ui-upload {
        background: #0d1220 !important;
        border-color: #283047 !important;
        border-radius: 9px !important;
        min-width: 0 !important;
    }

    #input-panel .ui-upload label {
        font-size: 11.5px !important;
        font-weight: 620 !important;
    }

    .input-upload-meta {
        align-items: center;
        border: 1px dashed #2c3449;
        border-radius: 8px;
        color: var(--ui-muted);
        display: flex;
        font-size: 10px;
        justify-content: center;
        margin-top: 9px;
        min-height: 38px;
        padding: 7px 10px;
        text-align: center;
    }

    #prompt-mode-tabs .tabitem > .column {
        gap: 8px !important;
    }

    #prompt-mode-tabs .block:has(.input-mode-guide),
    #prompt-mode-tabs .block:has(.input-upload-meta) {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    #prompt-mode-tabs .block:has(.input-mode-guide) .html-container,
    #prompt-mode-tabs .block:has(.input-upload-meta) .html-container {
        padding: 0 !important;
    }

    #prompt-mode-tabs .input-mode-guide {
        margin: 4px 0;
        min-height: 44px;
    }

    #prompt-mode-tabs .input-upload-meta {
        margin-top: 0;
        min-height: 32px;
    }

    #generate-3d-button {
        background: linear-gradient(135deg, #4d4ff0 0%, #6670ff 58%, #7a70ff 100%) !important;
        border: 1px solid rgba(143, 150, 255, 0.58) !important;
        border-radius: 9px !important;
        box-shadow: 0 10px 24px rgba(70, 73, 222, 0.22);
        min-height: 50px;
    }

    #settings-tabs,
    #settings-tabs > div,
    #settings-tabs .tabitem,
    #advanced-settings-form,
    #advanced-settings-form > div {
        align-self: stretch !important;
        box-sizing: border-box !important;
        flex: 1 1 auto !important;
        max-width: none !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    #settings-tabs {
        overflow: visible !important;
    }

    #settings-tabs .tab-nav {
        min-height: 40px;
    }

    #settings-tabs .tab-nav button[role="tab"] {
        font-size: 11px;
        min-height: 40px;
        padding: 0 10px;
    }

    #advanced-settings-form {
        gap: 10px !important;
        overflow: visible !important;
        padding-top: 10px;
    }

    .ui-control-row {
        align-self: stretch !important;
        display: grid !important;
        flex: 1 1 auto !important;
        gap: 8px !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        max-width: none !important;
        min-width: 0 !important;
        overflow: visible !important;
        width: 100% !important;
    }

    .ui-control-row > *,
    #advanced-settings-form .ui-control,
    #advanced-settings-form .ui-control > div,
    #advanced-settings-form .ui-control .wrap,
    #advanced-settings-form .ui-control .slider-container {
        box-sizing: border-box !important;
        flex: none !important;
        max-width: 100% !important;
        min-width: 0 !important;
        overflow: visible !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control {
        background: #131827 !important;
        border-color: #262e43 !important;
        border-radius: 9px !important;
        min-height: 74px;
        padding: 9px 10px !important;
    }

    #advanced-settings-form .ui-control-checkbox {
        min-height: 38px;
        padding: 7px 9px !important;
    }

    #advanced-settings-form .ui-control label,
    #advanced-settings-form .ui-control .label-wrap {
        font-size: 11px !important;
        min-width: 0 !important;
    }

    #advanced-settings-form input[type="range"] {
        max-width: 100% !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    #output-tabs {
        align-self: start;
        box-sizing: border-box;
        gap: 0 !important;
        min-height: var(--ui-workspace-height);
        padding: 8px 12px 12px !important;
    }

    #output-tabs .tab-nav {
        min-height: 44px;
        padding: 0 4px;
    }

    #output-tabs .tab-nav button[role="tab"] {
        font-size: 11px;
        min-height: 44px;
        padding: 0 14px;
    }

    #mesh-viewer,
    #mesh-export-viewer,
    #mesh-stats {
        margin-top: 12px;
    }

    #mesh-stats {
        background: #090e19 !important;
        border: 1px solid #252d40 !important;
        border-radius: 11px !important;
        box-sizing: border-box;
        height: var(--ui-stage-height) !important;
        min-height: var(--ui-stage-height) !important;
        max-height: var(--ui-stage-height) !important;
        overflow: hidden !important;
        padding: 0 !important;
    }

    #mesh-stats .json-holder {
        box-sizing: border-box;
        height: calc(100% - 26px) !important;
        max-height: calc(100% - 26px) !important;
        min-height: 0;
        overflow: auto !important;
        overscroll-behavior: contain;
        padding: 14px 16px 16px;
        scrollbar-gutter: stable;
    }

    #mesh-stats .json-node .line {
        align-items: center;
        min-height: 22px;
    }

    #mesh-stats .json-node button.toggle {
        align-items: center;
        background: transparent !important;
        border: 0 !important;
        border-radius: 5px !important;
        color: #7f89a5 !important;
        cursor: pointer;
        display: inline-flex;
        flex: 0 0 20px;
        height: 20px;
        justify-content: center;
        line-height: 0;
        margin: 0 5px 0 -2px !important;
        min-width: 20px !important;
        padding: 0 !important;
        transition: background-color 140ms ease, color 140ms ease;
        width: 20px;
    }

    /* CSS fallback until the shared SVG icon installer wires dynamic JSON nodes. */
    #mesh-stats .json-node button.toggle::before {
        border-bottom: 1.5px solid currentColor;
        border-right: 1.5px solid currentColor;
        box-sizing: border-box;
        content: "" !important;
        height: 6px;
        pointer-events: none;
        transform: rotate(45deg) !important;
        transition: transform 160ms ease;
        width: 6px;
    }

    #mesh-stats .json-node .line.collapsed button.toggle::before {
        transform: rotate(-45deg);
    }

    #mesh-stats .json-node button.toggle[data-ui-disclosure-icon-wired="true"]::before {
        display: none;
    }

    #mesh-stats .json-node button.toggle .ui-disclosure-icon {
        height: 14px;
        pointer-events: none;
        transform: rotate(0deg);
        transition: transform 160ms ease;
        width: 14px;
    }

    #mesh-stats .json-node .line.collapsed button.toggle .ui-disclosure-icon {
        transform: rotate(-90deg);
    }

    #mesh-stats .json-node .line:not(.collapsed) button.toggle {
        background: rgba(102, 112, 255, 0.08) !important;
        color: #969cff !important;
    }

    #mesh-stats .json-node button.toggle:hover {
        background: rgba(102, 112, 255, 0.14) !important;
        color: #bcc0ff !important;
    }

    #mesh-stats .json-node button.toggle:active {
        background: rgba(102, 112, 255, 0.2) !important;
    }

    #mesh-stats .json-node button.toggle:focus-visible {
        background: rgba(102, 112, 255, 0.14) !important;
        outline: 2px solid #858aff;
        outline-offset: 1px;
    }

    #mesh-stats .json-node button.toggle:disabled {
        cursor: default;
        opacity: 0.38;
    }

    #mesh-stats .empty-wrapper {
        height: 100%;
        min-height: 0 !important;
    }

    #mesh-viewer iframe,
    #mesh-export-viewer iframe {
        border: 0;
        border-radius: 11px;
        height: var(--ui-stage-height);
    }

    .viewer-empty-state {
        align-items: center;
        background:
            radial-gradient(circle at 50% 44%, rgba(80, 89, 220, 0.1), transparent 34%),
            #090e19;
        border: 1px solid #252d40;
        border-radius: 11px;
        box-sizing: border-box;
        display: flex;
        justify-content: center;
        overflow: hidden;
        position: relative;
        width: 100%;
    }

    .viewer-empty-state::after {
        background-image:
            linear-gradient(rgba(90, 103, 148, 0.14) 1px, transparent 1px),
            linear-gradient(90deg, rgba(90, 103, 148, 0.14) 1px, transparent 1px);
        background-size: 28px 28px;
        bottom: -38%;
        content: "";
        height: 58%;
        left: 8%;
        mask-image: linear-gradient(to bottom, transparent, #000 45%, transparent 100%);
        opacity: 0.55;
        perspective: 520px;
        position: absolute;
        transform: perspective(420px) rotateX(64deg);
        width: 84%;
    }

    .viewer-empty-copy {
        align-items: center;
        color: var(--ui-muted);
        display: flex;
        flex-direction: column;
        font-size: 12.5px;
        gap: 8px;
        position: relative;
        text-align: center;
        z-index: 1;
    }

    .viewer-empty-mark {
        align-items: center;
        background: rgba(102, 112, 255, 0.12);
        border: 1px solid rgba(102, 112, 255, 0.34);
        border-radius: 12px;
        color: #8d94ff;
        display: inline-flex;
        height: 42px;
        justify-content: center;
        width: 42px;
    }

    .viewer-empty-copy strong {
        color: #dfe3f1;
        font-size: 14px;
    }

    .viewer-empty-copy p {
        color: var(--ui-muted) !important;
        margin: 0;
    }

    #generation-console-panel {
        align-self: start;
        display: flex !important;
        flex-direction: column;
        gap: 12px !important;
        min-height: calc(var(--ui-stage-height) + 80px);
        padding: 16px !important;
    }

    #generation-console-panel > .html-container,
    #generation-console-panel > .form,
    #generation-console-panel > .block {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        height: auto;
        margin: 0 !important;
        min-width: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    .generation-console {
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        height: 420px;
        min-height: 420px;
    }

    .generation-console-windowbar {
        background: transparent;
        border-bottom: 0;
        min-height: 36px;
        padding: 0 2px 8px;
    }

    .generation-console-dots {
        display: none;
    }

    .generation-console-title strong {
        font-family: var(--ui-font);
        font-size: 14.5px;
    }

    .generation-console-title span {
        font-size: 9px;
        letter-spacing: 0.07em;
    }

    .generation-console-progress-wrap {
        background: transparent;
        border: 0;
        padding: 6px 2px 12px;
    }

    .generation-console-progress-track {
        background: #232a3e;
        height: 5px;
    }

    .generation-console-progress-meta,
    .generation-console-status,
    .generation-console-mode,
    .generation-console-footer {
        font-size: 10px;
    }

    .generation-console-time {
        color: #6f7a91;
    }

    .generation-console-message {
        color: #d0d5e2;
    }

    .generation-console-jobbar {
        background: #0d1220;
        border: 1px solid #252d40;
        border-radius: 8px 8px 0 0;
        min-height: 36px;
        padding: 0 10px;
    }

    .generation-console-log {
        background: #090e19;
        border: 1px solid #252d40;
        border-bottom: 0;
        border-top: 0;
        padding: 12px 10px;
    }

    .generation-console-line {
        font-size: 10.5px;
        gap: 6px;
        grid-template-columns: 52px 46px minmax(0, 1fr);
        line-height: 1.58;
        margin-bottom: 7px;
    }

    .generation-console-footer {
        background: #0d1220;
        border: 1px solid #252d40;
        border-radius: 0 0 8px 8px;
        min-height: 28px;
        padding: 0 10px;
    }

    .generation-details-card,
    #generation-output-card {
        background: #131827 !important;
        border: 1px solid #262e43 !important;
        border-radius: 10px !important;
        box-sizing: border-box;
        overflow: hidden;
        width: 100%;
    }

    .generation-details-card {
        background: linear-gradient(145deg, #151b2c 0%, #111725 100%) !important;
        border-color: #2d3750 !important;
        border-radius: 12px !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.025),
            0 8px 20px rgba(0, 0, 0, 0.1);
        margin-top: 12px;
        min-height: 148px;
        padding: 12px;
    }

    .generation-details-card .dashboard-card-title {
        gap: 7px;
        justify-content: flex-start;
        margin: 0 0 8px;
        min-height: 20px;
        padding: 0 2px;
    }

    .generation-details-card .dashboard-card-title::before {
        background: #747cff;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(116, 124, 255, 0.12);
        content: "";
        flex: 0 0 6px;
        height: 6px;
        width: 6px;
    }

    .dashboard-card-title {
        align-items: center;
        color: var(--ui-text);
        display: flex;
        font-family: var(--ui-font);
        font-size: 12px;
        font-weight: 700;
        justify-content: space-between;
        margin: 0 0 11px;
    }

    .generation-details-grid {
        display: grid;
        gap: 6px 8px;
        grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
    }

    .generation-detail {
        align-items: center;
        background: linear-gradient(180deg, rgba(13, 19, 34, 0.88), rgba(10, 16, 29, 0.78));
        border: 1px solid rgba(48, 58, 85, 0.74);
        border-radius: 7px;
        box-sizing: border-box;
        display: flex;
        font-size: 9.5px;
        gap: 6px;
        justify-content: space-between;
        min-height: 28px;
        min-width: 0;
        padding: 5px 7px;
    }

    .generation-detail span:first-child {
        color: var(--ui-muted) !important;
        flex: 0 0 auto;
    }

    .generation-detail strong {
        color: #eef1fb;
        font-family: var(--ui-mono);
        font-size: 10.5px;
        font-variant-numeric: tabular-nums;
        font-weight: 680;
        max-width: 65%;
        min-width: 0;
        overflow: hidden;
        text-align: right;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    #generation-info-model {
        max-width: 72%;
    }

    #generation-output-card {
        background: linear-gradient(145deg, #151b2c 0%, #111725 100%) !important;
        border-color: #2d3750 !important;
        border-radius: 12px !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.025),
            0 8px 20px rgba(0, 0, 0, 0.1);
        flex: 0 0 auto !important;
        max-height: none;
        min-height: 110px !important;
        padding: 12px !important;
    }

    #generation-output-card > .styler {
        background: transparent !important;
        gap: 8px !important;
        overflow: visible !important;
    }

    #generation-output-card .dashboard-card-title {
        gap: 8px;
        min-height: 20px;
        margin: 0 !important;
        padding: 0 2px;
    }

    #generation-output-card .dashboard-card-title > span:first-child {
        align-items: center;
        display: inline-flex;
        gap: 7px;
        letter-spacing: 0.01em;
    }

    #generation-output-card .dashboard-card-title > span:first-child::before {
        background: #747cff;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(116, 124, 255, 0.12);
        content: "";
        flex: 0 0 6px;
        height: 6px;
        width: 6px;
    }

    #generation-output-card .generation-output-heading {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        min-height: 20px !important;
        max-height: none !important;
        overflow: visible !important;
        padding: 0 !important;
    }

    #generation-output-card .generation-output-heading > .wrap {
        display: none !important;
    }

    #generation-output-card .generation-output-heading .html-container {
        background: transparent !important;
        padding: 0 !important;
    }

    #generation-output-card .generation-output-heading .prose {
        background: transparent !important;
    }

    #generation-output-card .generation-output-file {
        background: linear-gradient(180deg, #0d1322 0%, #0a101d 100%) !important;
        border: 1px solid #303a55 !important;
        border-radius: 10px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.018);
        max-height: none !important;
        min-height: 54px !important;
        overflow: hidden !important;
        padding: 0 !important;
    }

    #generation-output-card .generation-output-file > label[data-testid='block-label'] {
        display: none !important;
    }

    #generation-output-card .generation-output-file .file-preview-holder {
        height: 52px;
        max-height: none;
        overflow: hidden;
        width: 100%;
    }

    #generation-output-card .generation-output-file .file-preview {
        border: 0;
        font-size: 11px !important;
        margin: 0 !important;
        max-height: none;
        width: 100%;
    }

    #generation-output-card .generation-output-file .file-preview tbody,
    #generation-output-card .generation-output-file .file-preview tr {
        width: 100%;
    }

    #generation-output-card .generation-output-file .file-preview tr {
        align-items: center;
        background: transparent;
        display: flex;
        min-height: 52px;
    }

    #generation-output-card .generation-output-file td.filename {
        align-items: center;
        color: #eef1fb;
        display: flex;
        flex: 1 1 auto;
        font-family: var(--ui-mono);
        font-size: 11px;
        font-weight: 650;
        gap: 0;
        height: 52px;
        min-width: 0;
        overflow: hidden;
        padding: 0 10px !important;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    #generation-output-card .generation-output-file td.filename::before {
        align-items: center;
        background: rgba(116, 124, 255, 0.11);
        border: 1px solid rgba(116, 124, 255, 0.28);
        border-radius: 6px;
        color: #9ca3ff;
        content: "3D";
        display: inline-flex;
        flex: 0 0 auto;
        font-family: var(--ui-font);
        font-size: 8px;
        font-weight: 800;
        height: 22px;
        justify-content: center;
        letter-spacing: 0.05em;
        margin-right: 9px;
        padding: 0 6px;
    }

    #generation-output-card .generation-output-file td.filename .stem {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    #generation-output-card .generation-output-file td.filename .ext {
        flex: 0 0 auto;
    }

    #generation-output-card .generation-output-file td.download {
        align-items: center;
        display: flex !important;
        flex: 0 0 auto;
        height: 52px;
        justify-content: flex-end;
        padding: 6px 8px 6px 4px !important;
        width: auto !important;
    }

    #generation-output-card .generation-output-file td.download a {
        align-items: center;
        background: linear-gradient(135deg, rgba(83, 92, 218, 0.2), rgba(62, 71, 177, 0.13));
        border: 1px solid #5661ca;
        border-radius: 8px;
        box-shadow: 0 5px 13px rgba(36, 43, 128, 0.15);
        color: #b5baff;
        display: inline-flex !important;
        font-family: var(--ui-font);
        font-size: 10.5px;
        font-weight: 700;
        gap: 6px;
        height: 32px;
        justify-content: center;
        min-width: 90px;
        padding: 0 9px;
        text-decoration: none;
        transition:
            background 150ms ease,
            border-color 150ms ease,
            box-shadow 150ms ease,
            color 150ms ease,
            transform 150ms ease;
    }

    #generation-output-card .generation-output-file td.download a:hover {
        background: linear-gradient(135deg, rgba(100, 110, 244, 0.3), rgba(74, 84, 206, 0.2));
        border-color: #7780f5;
        box-shadow: 0 7px 16px rgba(47, 56, 164, 0.22);
        color: #d4d7ff;
        transform: translateY(-1px);
    }

    #generation-output-card .generation-output-file td.download a:focus-visible {
        outline: 2px solid #858aff;
        outline-offset: 2px;
    }

    #generation-output-card .generation-output-file td.download .ui-icon {
        height: 14px;
        width: 14px;
    }

    #generation-output-card .generation-output-file .empty.large {
        align-items: center;
        color: #7f899f;
        display: flex;
        font-family: var(--ui-font);
        font-size: 10.5px;
        justify-content: center;
        min-height: 56px !important;
        padding: 10px 12px !important;
        text-align: center;
    }

    #generation-output-card .generation-output-file .empty.large::before {
        content: 'Generated mesh will appear here';
    }

    #generation-output-card .generation-output-file table {
        border-spacing: 0 !important;
        font-variant-numeric: tabular-nums;
    }

    .generation-output-meta {
        background: rgba(128, 138, 170, 0.08);
        border: 1px solid rgba(128, 138, 170, 0.15);
        border-radius: 999px;
        color: #9ca5ba !important;
        font-size: 9px;
        font-weight: 600;
        line-height: 1.2;
        max-width: 70%;
        min-width: 0;
        overflow: hidden;
        padding: 3px 7px;
        text-align: right;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Left rail v2: compact, target-aligned controls without changing Gradio inputs. */
    #input-panel > .block:has(.panel-heading) {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        flex-shrink: 0 !important;
        min-height: 24px !important;
        overflow: visible !important;
        padding: 0 !important;
    }

    #input-panel > .block:has(.panel-heading) .html-container {
        padding: 0 !important;
    }

    #input-panel .panel-heading {
        font-size: 14px;
        min-height: 24px;
        padding: 0 2px;
    }

    #prompt-mode-tabs > .tab-wrapper > .tab-container[role="tablist"] {
        background: #171c2b;
        border: 1px solid #222a3d;
        border-radius: 8px;
        display: grid !important;
        gap: 3px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        min-height: 36px;
        padding: 3px;
        width: 100%;
    }

    #prompt-mode-tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"] {
        align-items: center;
        border: 0 !important;
        border-radius: 6px;
        display: flex;
        font-size: 11.5px;
        justify-content: center;
        min-height: 30px;
        padding: 4px 7px;
        text-align: center;
        width: 100%;
    }

    #prompt-mode-tabs .tabitem {
        padding-top: 10px !important;
    }

    #prompt-mode-tabs .input-mode-guide {
        box-sizing: border-box;
        gap: 10px;
        margin: 0;
        min-height: 50px;
        padding: 8px 10px;
    }

    #prompt-mode-tabs .input-mode-number {
        flex-basis: 30px;
        height: 30px;
        width: 30px;
    }

    #prompt-mode-tabs .input-mode-number .ui-icon {
        height: 14px;
        width: 14px;
    }

    #prompt-mode-tabs .input-mode-copy {
        min-width: 0;
    }

    #prompt-mode-tabs .input-mode-copy strong {
        font-size: 11.5px;
        line-height: 1.25;
        margin-bottom: 2px;
    }

    #prompt-mode-tabs .input-mode-copy span {
        font-size: 10px;
        line-height: 1.35;
    }

    #prompt-mode-tabs .tabitem > .column {
        gap: 8px !important;
    }

    #prompt-mode-tabs .mv-upload-row {
        gap: 10px !important;
    }

    #prompt-mode-tabs .mv-image button .wrap {
        font-size: 11px;
        line-height: 1.35;
    }

    #prompt-mode-tabs .mv-image {
        min-height: 180px !important;
    }

    #prompt-mode-tabs .mv-image .icon-wrap {
        width: 24px;
    }

    #prompt-mode-tabs .input-upload-meta {
        margin-top: 0;
        min-height: 34px;
    }

    #prompt-mode-tabs .input-upload-meta--stacked {
        flex-direction: column;
        gap: 4px;
        min-height: 58px;
        padding: 8px 10px;
    }

    .input-upload-meta-title {
        align-items: center;
        color: #c5cbdb;
        display: flex;
        font-size: 10.5px;
        font-weight: 650;
        gap: 5px;
        justify-content: center;
        line-height: 1.2;
    }

    .input-upload-meta-title .ui-icon {
        height: 12px;
        width: 12px;
    }

    .input-upload-meta-subtitle {
        color: var(--ui-muted);
        font-size: 9.5px;
        line-height: 1.3;
    }

    .generate-actions {
        gap: 0 !important;
        margin-top: 2px;
    }

    #generate-3d-button {
        border-radius: 8px !important;
        font-size: 12.5px;
        gap: 7px;
        min-height: 50px;
    }

    #generate-3d-button .ui-action-icon {
        height: 14px;
        width: 14px;
    }

    .generate-button-copy {
        align-items: center;
        display: flex;
        flex-direction: column;
        gap: 1px;
        justify-content: center;
        line-height: 1.1;
    }

    .generate-button-copy strong {
        color: #fff;
        font-size: 12.5px;
        font-weight: 700;
    }

    .generate-button-copy small {
        color: rgba(255, 255, 255, 0.72);
        font-size: 10px;
        font-weight: 520;
    }

    #settings-tabs,
    #settings-tabs > div,
    #settings-tabs .tabitem,
    #advanced-settings-form,
    #advanced-settings-form > div {
        flex: none !important;
        height: auto !important;
    }

    #settings-tabs {
        margin-top: 4px;
    }

    #settings-tabs > .tab-wrapper > .tab-container[role="tablist"]:has(> #advanced-settings-form-button) {
        background: #151a28;
        border: 1px solid #252d40;
        border-radius: 8px;
        display: flex;
        min-height: 38px;
        padding: 3px;
        width: 100%;
    }

    #settings-tabs > .tab-wrapper > .tab-container[role="tablist"]:has(> #advanced-settings-form-button)
    > #advanced-settings-form-button {
        align-items: center;
        border: 0 !important;
        color: #aeb5c8;
        display: flex;
        font-size: 11.5px;
        justify-content: flex-start;
        min-height: 30px;
        padding: 0 8px;
        width: 100%;
    }

    #settings-tabs > .tab-wrapper > .tab-container[role="tablist"]:has(> #advanced-settings-form-button)
    > #advanced-settings-form-button::after {
        background: transparent !important;
        border: 0 !important;
        color: #7e879e;
        content: "⌄";
        display: block !important;
        font-size: 13px;
        height: auto !important;
        margin-left: auto;
        position: static !important;
        width: auto !important;
        border-color: #8f99af !important;
        border-style: solid !important;
        border-width: 0 1.5px 1.5px 0 !important;
        content: "" !important;
        flex: 0 0 6px;
        height: 6px !important;
        margin: -3px 3px 0 auto !important;
        transform: rotate(45deg) !important;
        width: 6px !important;
    }

    #advanced-settings-form {
        padding-top: 8px;
    }

    #advanced-settings-form > .column {
        gap: 12px !important;
    }

    #advanced-settings-form .ui-control-wide {
        height: auto !important;
        min-height: 116px !important;
        overflow: visible !important;
        padding: 12px !important;
    }

    #advanced-settings-form .ui-control-wide .head {
        align-items: stretch !important;
        display: grid !important;
        gap: 8px !important;
        grid-template-columns: minmax(0, 1fr) !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-wide .head label,
    #advanced-settings-form .ui-control-wide .label-wrap {
        font-size: 11px !important;
        font-weight: 650 !important;
    }

    #advanced-settings-form .ui-control-wide input {
        font-size: 12px !important;
        font-variant-numeric: tabular-nums;
    }

    #advanced-settings-form .ui-control-wide .tab-like-container {
        display: flex !important;
        height: 36px !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-wide .tab-like-container input {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }

    #advanced-settings-form .ui-control-wide .tab-like-container button {
        flex: 0 0 36px !important;
        min-width: 36px !important;
        width: 36px !important;
    }

    #advanced-settings-form .ui-control-wide .slider-container {
        gap: 10px !important;
        margin-top: 8px !important;
    }

    #advanced-settings-form .ui-control-row {
        display: block !important;
        flex: none !important;
    }

    #advanced-settings-form .ui-control-row > .form {
        display: grid !important;
        gap: 10px !important;
        grid-template-columns: minmax(0, 1fr) !important;
        max-width: none !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-row-compact .ui-control {
        height: auto !important;
        min-height: 88px !important;
        overflow: visible !important;
        padding: 11px 12px !important;
    }

    #advanced-settings-form .ui-control-row-compact .head {
        align-items: stretch !important;
        display: grid !important;
        gap: 8px !important;
        grid-template-columns: minmax(0, 1fr) !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-row-compact .head label,
    #advanced-settings-form .ui-control-row-compact [data-testid="block-info"],
    #advanced-settings-form .ui-control-row-compact .ui-control > label.block.container {
        font-size: 11px !important;
        font-weight: 650 !important;
        line-height: 1.35 !important;
        min-width: 0 !important;
        white-space: normal !important;
    }

    #advanced-settings-form .ui-control-row-compact .ui-control > label.block.container {
        background: transparent !important;
        border: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        height: auto !important;
        padding: 0 !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-row-compact .tab-like-container {
        display: flex !important;
        flex: 0 0 36px !important;
        height: 36px !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-row-compact .tab-like-container input {
        flex: 1 1 auto !important;
        font-size: 12px !important;
        font-variant-numeric: tabular-nums;
        min-width: 0 !important;
        padding: 6px 10px !important;
        text-align: left !important;
        width: auto !important;
    }

    #advanced-settings-form .ui-control-row-compact .tab-like-container button {
        flex: 0 0 36px !important;
        min-width: 36px !important;
        width: 36px !important;
    }

    #advanced-settings-form .ui-control-row-compact input[type="number"] {
        -moz-appearance: textfield;
        appearance: textfield;
        background: #0d1220 !important;
        border: 1px solid #30384e !important;
        border-radius: 5px !important;
        font-size: 12px !important;
        font-variant-numeric: tabular-nums;
        height: 36px !important;
        min-height: 36px !important;
        min-width: 0 !important;
        padding: 6px 10px !important;
        text-align: left !important;
        width: 100% !important;
    }

    #advanced-settings-form .ui-control-row-compact .slider_input_container {
        display: none !important;
    }

    #advanced-settings-form input[type="number"]::-webkit-inner-spin-button,
    #advanced-settings-form input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }

    #advanced-settings-form .ui-control-row-checks > .form {
        grid-template-columns: minmax(0, 1fr) !important;
    }

    #advanced-settings-form .ui-control-row-checks .ui-control.hidden {
        display: none !important;
    }

    #advanced-settings-form .ui-control-row .ui-control-checkbox {
        height: auto !important;
        min-height: 48px !important;
        padding: 10px 12px !important;
    }

    #advanced-settings-form .ui-control-row .ui-control-checkbox label,
    #advanced-settings-form .ui-control-row .ui-control-checkbox label span {
        font-size: 11px !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        white-space: normal !important;
    }

    /* Advanced Options visual polish; native input/reset behavior stays intact. */
    #settings-tabs > .tab-wrapper > .tab-container[role="tablist"]:has(> #advanced-settings-form-button) {
        background: linear-gradient(135deg, #171e30 0%, #111827 100%);
        border: 1px solid #303b59;
        border-radius: 10px;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.035),
            0 4px 12px rgba(0, 0, 0, 0.08);
        min-height: 40px;
    }

    #advanced-settings-form-button {
        background: rgba(116, 124, 255, 0.07) !important;
        border-radius: 7px !important;
        color: #e6e9f4 !important;
        font-size: 11.5px !important;
        font-weight: 650 !important;
        gap: 9px;
        letter-spacing: 0.005em;
        min-height: 32px !important;
        padding: 0 10px !important;
        transition: background 150ms ease, color 150ms ease;
    }

    #advanced-settings-form-button::before {
        background: #7b83ff;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(123, 131, 255, 0.14);
        content: "";
        flex: 0 0 7px;
        height: 7px;
        width: 7px;
    }

    #advanced-settings-form-button:hover {
        background: rgba(116, 124, 255, 0.1) !important;
        color: #f3f5ff !important;
    }

    #advanced-settings-form-button:focus-visible {
        box-shadow: 0 0 0 2px rgba(116, 124, 255, 0.38);
        outline: 0;
    }

    #advanced-settings-form {
        --advanced-card-bg: linear-gradient(145deg, #151c2d 0%, #101624 100%);
        --advanced-card-border: #303c5a;
        --advanced-field-bg: linear-gradient(180deg, #0d1424 0%, #09101d 100%);
        padding: 10px 8px 0 !important;
    }

    #advanced-settings-form > .column {
        gap: 8px !important;
    }

    #advanced-settings-form > .column > .form,
    #advanced-settings-form .ui-control-row > .form {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        gap: 8px !important;
        padding: 0 !important;
    }

    #advanced-settings-form .ui-control {
        background: var(--advanced-card-bg) !important;
        border: 1px solid var(--advanced-card-border) !important;
        border-radius: 11px !important;
        box-sizing: border-box !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }

    #advanced-settings-form .ui-control:hover {
        border-color: #435174 !important;
    }

    #advanced-settings-form .ui-control:focus-within {
        border-color: #6873e3 !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
    }

    #advanced-settings-form .ui-control-wide {
        min-height: 106px !important;
        padding: 10px !important;
    }

    #advanced-settings-form .ui-control-wide [data-testid="block-info"] {
        font-size: 11px !important;
        line-height: 1.35 !important;
    }

    #advanced-settings-form .ui-control-row-compact .ui-control {
        min-height: 82px !important;
        padding: 9px 10px !important;
    }

    #advanced-settings-form .ui-control-row .ui-control-checkbox {
        min-height: 42px !important;
        padding: 8px 10px !important;
    }

    #advanced-settings-form .ui-control-wide .head,
    #advanced-settings-form .ui-control-row-compact .head,
    #advanced-settings-form .ui-control-row-compact .ui-control > label.block.container {
        gap: 7px !important;
        margin-bottom: 0 !important;
    }

    #advanced-settings-form .ui-control [data-testid="block-info"] {
        margin-bottom: 0 !important;
    }

    #advanced-settings-form .ui-control .head label,
    #advanced-settings-form .ui-control [data-testid="block-info"],
    #advanced-settings-form .ui-control > label.block.container,
    #advanced-settings-form .ui-control-checkbox label span {
        color: #e3e7f2 !important;
        font-size: 11.5px !important;
        font-weight: 650 !important;
        letter-spacing: 0.005em;
        line-height: 1.35 !important;
    }

    #advanced-settings-form .ui-control input[type="number"] {
        background: var(--advanced-field-bg) !important;
        border: 1px solid #364361 !important;
        border-radius: 7px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025) !important;
        box-sizing: border-box;
        color: #f2f4fb !important;
        font-size: 12.5px !important;
        font-variant-numeric: tabular-nums;
        font-weight: 600 !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 6px 10px !important;
        text-align: left !important;
    }

    #advanced-settings-form .ui-control-wide .tab-like-container,
    #advanced-settings-form .ui-control-row-compact .tab-like-container {
        border: 0 !important;
        border-radius: 7px !important;
        height: 34px !important;
        overflow: visible !important;
    }

    #advanced-settings-form .ui-control-row-compact .tab-like-container {
        flex: 0 0 34px !important;
    }

    #advanced-settings-form .ui-control .tab-like-container input[type="number"] {
        border-radius: 7px 0 0 7px !important;
    }

    #advanced-settings-form button[data-testid="reset-button"] {
        background: linear-gradient(180deg, #151d2e 0%, #101726 100%) !important;
        border: 1px solid #364361 !important;
        border-left: 0 !important;
        border-radius: 0 7px 7px 0 !important;
        box-sizing: border-box;
        color: #aab3c8 !important;
        height: 34px !important;
        margin: 0 !important;
        min-height: 34px !important;
        transition: background 150ms ease, border-color 150ms ease, color 150ms ease;
    }

    #advanced-settings-form .ui-control-wide button[data-testid="reset-button"],
    #advanced-settings-form .ui-control-row-compact button[data-testid="reset-button"] {
        border: 1px solid #364361 !important;
        border-left: 0 !important;
        flex: 0 0 34px !important;
        max-width: 34px !important;
        min-width: 34px !important;
        width: 34px !important;
    }

    #advanced-settings-form button[data-testid="reset-button"]:hover {
        background: #1a2235 !important;
        border-color: #5661ca !important;
        color: #c7ccff !important;
    }

    #advanced-settings-form button[data-testid="reset-button"]:focus-visible {
        box-shadow: 0 0 0 2px rgba(116, 124, 255, 0.3);
        outline: 0;
        position: relative;
        z-index: 1;
    }

    #advanced-settings-form button[data-testid="reset-button"] .ui-icon {
        height: 14px !important;
        width: 14px !important;
    }

    #advanced-settings-form .ui-control input[type="number"]:focus {
        border-color: #6872e8 !important;
        box-shadow: 0 0 0 2px rgba(104, 114, 232, 0.14) !important;
        outline: 0;
    }

    #advanced-settings-form .slider_input_container {
        --neutral-200: #2c354d;
        --slider-color: #6872f1;
        color: #8f99b0;
        gap: 9px !important;
    }

    #advanced-settings-form .slider_input_container .min_value,
    #advanced-settings-form .slider_input_container .max_value {
        color: #939db4 !important;
        font-family: var(--ui-mono);
        font-size: 9.5px;
        font-variant-numeric: tabular-nums;
    }

    #advanced-settings-form .ui-control-wide .slider_input_container {
        transform: translateY(4px);
    }

    #advanced-settings-form .slider_input_container input[type="range"]::-webkit-slider-runnable-track {
        height: 6px;
    }

    #advanced-settings-form .slider_input_container input[type="range"]::-webkit-slider-thumb {
        background: #f4f5ff;
        border: 2px solid #dfe2ff;
        box-shadow: 0 0 0 3px rgba(104, 114, 241, 0.12), 0 2px 5px rgba(0, 0, 0, 0.24);
        height: 16px;
        margin-top: -5px;
        width: 16px;
    }

    #advanced-settings-form .slider_input_container input[type="range"]::-moz-range-track,
    #advanced-settings-form .slider_input_container input[type="range"]::-moz-range-progress {
        height: 6px;
    }

    #advanced-settings-form .slider_input_container input[type="range"]::-moz-range-thumb {
        background: #f4f5ff;
        border: 2px solid #dfe2ff;
        box-shadow: 0 0 0 3px rgba(104, 114, 241, 0.12), 0 2px 5px rgba(0, 0, 0, 0.24);
        height: 16px;
        width: 16px;
    }

    #advanced-settings-form .ui-control-checkbox label {
        align-items: center;
        cursor: pointer;
        gap: 9px !important;
    }

    #advanced-settings-form .ui-control-checkbox input[data-testid="checkbox"] {
        accent-color: #6872f1;
        flex: 0 0 16px;
        height: 16px;
        width: 16px;
    }

    @media (max-width: 1400px) {
        .app-topbar {
            grid-template-columns: 320px minmax(0, 1fr);
        }

        .app-topbar-actions {
            grid-column: 1 / -1;
            justify-content: flex-start;
        }

        #workspace-grid {
            grid-template-columns: 320px minmax(0, 1fr);
        }

        #generation-console-panel {
            grid-column: 1 / -1;
            min-height: auto;
        }

        .generation-console {
            height: 430px;
            min-height: 430px;
        }
    }

    @media (max-width: 900px) {
        .app-topbar,
        #workspace-grid {
            grid-template-columns: minmax(0, 1fr);
        }

        #input-panel {
            height: auto;
            max-height: none;
            min-height: 0;
            overflow: visible;
            scrollbar-gutter: auto;
        }

        .app-title-block {
            grid-row: 1;
        }

        .app-brand {
            display: none;
        }

        .app-topbar-actions {
            grid-column: auto;
        }

        #generation-console-panel {
            grid-column: auto;
        }
    }

    @media (max-width: 520px) {
        .gradio-container {
            padding: 12px !important;
        }

        .app-title-mark {
            display: none;
        }

        .app-title-block h1 {
            font-size: 18px;
            white-space: normal;
        }

        .app-topbar-actions {
            flex-wrap: wrap;
        }

        .ui-control-row,
        .generation-details-grid {
            grid-template-columns: minmax(0, 1fr) !important;
        }

        #advanced-settings-form .ui-control-row > .form {
            grid-template-columns: minmax(0, 1fr) !important;
        }
    }

    """

    custom_js = r"""
    () => {
        const modalId = "rtx3090-modal";
        const footerButtonId = "rtx3090-footer-trigger";
        const presetButtonIds = {
            safe: "rtx3090-safe-preset",
            quality: "rtx3090-quality-preset",
        };
        const modal = () => document.getElementById(modalId);
        const uiIconPaths = {
            box: '<path d="m3 8 9-5 9 5-9 5-9-5Z"></path><path d="m3 8v8l9 5 9-5V8"></path><path d="M12 13v8"></path>',
            terminal: '<path d="m5 7 5 5-5 5"></path><path d="M12 19h7"></path>',
            zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"></path>',
            x: '<path d="M18 6 6 18"></path><path d="m6 6 12 12"></path>',
            check: '<path d="m5 12 4 4L19 6"></path>',
            info: '<circle cx="12" cy="12" r="9"></circle><path d="M12 11v5"></path><path d="M12 8h.01"></path>',
            memory: '<rect x="5" y="6" width="14" height="12" rx="2"></rect><path d="M9 10h6v4H9z"></path><path d="M8 3v3M12 3v3M16 3v3M8 18v3M12 18v3M16 18v3"></path>',
            wand: '<path d="m15 4 5 5L7 22H2v-5L15 4Z"></path><path d="m14 5 5 5"></path><path d="M6 3v4M4 5h4M19 15v4M17 17h4"></path>',
            code: '<path d="m8 9-3 3 3 3"></path><path d="m16 9 3 3-3 3"></path><path d="m14 5-4 14"></path>',
            settings: '<path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"></path><circle cx="16" cy="6" r="2"></circle><circle cx="8" cy="12" r="2"></circle><circle cx="13" cy="18" r="2"></circle>',
            rotate: '<path d="M3 12a9 9 0 1 0 3-6.7"></path><path d="M3 4v6h6"></path>',
            download: '<path d="M12 3v12"></path><path d="m7 10 5 5 5-5"></path><path d="M5 21h14"></path>',
            chevronDown: '<path d="m6 9 6 6 6-6"></path>',
        };

        const uiIconMarkup = (name, extraClass = "") => {
            const paths = uiIconPaths[name];
            if (!paths) return "";
            return '<svg class="ui-icon ' + extraClass + '" viewBox="0 0 24 24" aria-hidden="true">'
                + paths + '</svg>';
        };

        const syncGenerateButtonCopy = (forcedMultiView = null) => {
            const generateButton = document.getElementById("generate-3d-button");
            if (!generateButton) return;

            const isMultiView = typeof forcedMultiView === "boolean"
                ? forcedMultiView
                : document.querySelector(
                    '#prompt-mode-tabs button[data-tab-id="tab_mv_prompt"]'
                )?.getAttribute("aria-selected") === "true";
            let copy = generateButton.querySelector(".generate-button-copy");
            if (!copy) {
                copy = document.createElement("span");
                copy.className = "generate-button-copy";
                copy.append(document.createElement("strong"), document.createElement("small"));
                generateButton.replaceChildren();
                generateButton.insertAdjacentHTML("afterbegin", uiIconMarkup("wand", "ui-action-icon"));
                generateButton.append(copy);
            }
            const title = copy.querySelector("strong");
            const subtitle = copy.querySelector("small");
            const nextSubtitle = isMultiView
                ? "4 synchronized views"
                : "1 front image";
            if (title.textContent !== "Generate 3D") {
                title.textContent = "Generate 3D";
            }
            if (subtitle.textContent !== nextSubtitle) {
                subtitle.textContent = nextSubtitle;
            }
            generateButton.dataset.uiActionIcon = "true";
        };

        const installUnifiedIcons = () => {
            document.querySelectorAll("[data-ui-icon]").forEach((element) => {
                if (element.dataset.uiIconWired === "true") return;
                const iconName = element.dataset.uiIcon;
                if (!uiIconPaths[iconName]) return;
                element.dataset.uiIconWired = "true";
                element.innerHTML = uiIconMarkup(iconName);
            });

            syncGenerateButtonCopy();

            document.querySelectorAll('button.reset-button[data-testid="reset-button"]').forEach((button) => {
                if (button.dataset.uiIconWired === "true") return;
                button.dataset.uiIconWired = "true";
                button.innerHTML = uiIconMarkup("rotate");
            });

            document.querySelectorAll("#mesh-stats button.toggle").forEach((button) => {
                if (button.dataset.uiDisclosureIconWired === "true") return;
                button.dataset.uiDisclosureIconWired = "true";
                button.innerHTML = uiIconMarkup("chevronDown", "ui-disclosure-icon");
            });

            document.querySelectorAll(".file-preview td.download a").forEach((link) => {
                if (link.dataset.uiIconWired === "true") return;
                const filenameCell = link.closest("tr.file")?.querySelector("td.filename");
                const filename = filenameCell?.getAttribute("aria-label") ?? link.getAttribute("download") ?? "generated mesh";
                link.dataset.uiIconWired = "true";
                link.innerHTML = "Download " + uiIconMarkup("download");
                link.setAttribute("aria-label", "Download " + filename);
                link.setAttribute("title", "Download " + filename);
                if (filenameCell) filenameCell.setAttribute("title", filename);
            });

            const footerIcons = [
                ["button.show-api", "code"],
                ["button.settings", "settings"],
            ];
            footerIcons.forEach(([selector, iconName]) => {
                document.querySelectorAll("footer " + selector).forEach((element) => {
                    if (element.dataset.uiIconWired === "true") return;
                    element.dataset.uiIconWired = "true";
                    element.querySelector("img")?.remove();
                    element.insertAdjacentHTML("afterbegin", uiIconMarkup(iconName));
                });
            });
        };

        const tabRoutes = [
            {slug: "single-view", index: 0},
            {slug: "multi-view", index: 1},
        ];
        let tabRouteInitialized = false;

        const currentAppUrl = () => {
            const url = new URL(window.location.href);
            url.pathname = url.pathname.replace(/\/{2,}/g, "/");
            return url;
        };
        let activeGenerationRouteUid = currentAppUrl().searchParams.get("generation");

        const createGenerationUid = () => {
            if (window.crypto && typeof window.crypto.randomUUID === "function") {
                return window.crypto.randomUUID();
            }
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
                /[xy]/g,
                (character) => {
                    const randomValue = Math.floor(Math.random() * 16);
                    const value = character === "x" ? randomValue : (randomValue & 0x3) | 0x8;
                    return value.toString(16);
                }
            );
        };

        let generationConsoleTimer = null;
        let generationConsoleUid = null;
        let generationConsoleStartedAt = null;
        let generationConsoleSeenEvents = new Set();
        let generationConsoleParamsRendered = false;
        let generationConsolePollMisses = 0;

        const generationConsoleStageLevels = {
            request_received: "QUEUE",
            validating_input: "CHECK",
            input_validated: "INPUT",
            input_saved: "STORE",
            preprocessing_input: "PREP",
            input_ready: "READY",
            shape_generation: "CUDA",
            prepare_conditioning: "IMAGE",
            encode_conditioning: "ENCODE",
            conditioning_ready: "COND",
            prepare_timestep_schedule: "SCHED",
            latents_initialized: "LATENT",
            diffusion_started: "CUDA",
            diffusion_step: "STEP",
            diffusion_completed: "CUDA",
            vae_decoding: "VAE",
            volume_decoding: "VOLUME",
            volume_decoding_progress: "VOLUME",
            volume_decoding_completed: "VOLUME",
            surface_extraction: "OCTREE",
            surface_extraction_completed: "MESH",
            trimesh_conversion: "MESH",
            extracting_mesh: "MESH",
            mesh_ready: "MESH",
            exporting_glb: "WRITE",
            building_preview: "VIEW",
            completed: "DONE",
            failed: "ERROR",
        };

        const generationConsoleElement = (id) => document.getElementById(id);

        const generationConsoleElapsed = (timestamp = null) => {
            if (!generationConsoleStartedAt) return "+00.0s";
            const target = timestamp ? new Date(timestamp).getTime() : Date.now();
            const elapsed = Math.max(0, (target - generationConsoleStartedAt) / 1000);
            return "+" + elapsed.toFixed(1).padStart(4, "0") + "s";
        };

        const appendGenerationConsoleLine = (level, message, kind = "info", timestamp = null) => {
            const log = generationConsoleElement("generation-console-log");
            if (!log) return;

            const line = document.createElement("div");
            line.className = "generation-console-line";
            line.dataset.kind = kind;

            const time = document.createElement("span");
            time.className = "generation-console-time";
            time.textContent = generationConsoleElapsed(timestamp);

            const levelElement = document.createElement("span");
            levelElement.className = "generation-console-level";
            levelElement.textContent = level;

            const messageElement = document.createElement("span");
            messageElement.className = "generation-console-message";
            messageElement.textContent = message;

            line.append(time, levelElement, messageElement);
            log.appendChild(line);
            while (log.children.length > 200) log.firstElementChild?.remove();
            log.scrollTop = log.scrollHeight;
        };

        const setGenerationConsoleProgress = (progress, stage) => {
            const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
            const bar = generationConsoleElement("generation-console-progress");
            const percent = generationConsoleElement("generation-console-percent");
            const stageElement = generationConsoleElement("generation-console-stage");
            if (bar) bar.style.width = safeProgress + "%";
            if (percent) percent.textContent = Math.round(safeProgress) + "%";
            if (stageElement && stage) stageElement.textContent = stage;
        };

        const setGenerationConsoleState = (state, label) => {
            const root = generationConsoleElement("generation-console");
            const status = generationConsoleElement("generation-console-status");
            if (root) root.dataset.state = state;
            if (status) status.textContent = label;
        };

        const setGenerationDetail = (id, value) => {
            const element = generationConsoleElement(id);
            if (element) element.textContent = value ?? "—";
        };

        const formatGenerationCount = (value) => {
            const number = Number(value);
            if (!Number.isFinite(number)) return "—";
            if (number >= 1000000) return (number / 1000000).toFixed(2).replace(/\.00$/, "") + "M";
            if (number >= 1000) return (number / 1000).toFixed(1).replace(/\.0$/, "") + "K";
            return Math.round(number).toLocaleString("en-US");
        };

        const formatGenerationBytes = (value) => {
            const bytes = Number(value);
            if (!Number.isFinite(bytes) || bytes <= 0) return null;
            if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
            if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
            return bytes + " B";
        };

        const resetGenerationDetails = (mode = "") => {
            setGenerationDetail("generation-info-model", "—");
            setGenerationDetail("generation-info-views", mode === "4-VIEW" ? "4" : mode === "1-VIEW" ? "1" : "—");
            setGenerationDetail("generation-info-time", "—");
            setGenerationDetail("generation-info-resolution", "—");
            setGenerationDetail("generation-info-polygons", "—");
            setGenerationDetail("generation-info-vertices", "—");
            setGenerationDetail("generation-output-meta", "Awaiting generated mesh");
        };

        const updateGenerationDetails = (manifest) => {
            const params = manifest.params || manifest.stats?.params || {};
            const stats = manifest.stats || {};
            const rawModel = manifest.model?.shapegen || stats.model?.shapegen || "";
            const modelName = String(rawModel).split("/").filter(Boolean).at(-1) || "—";
            const viewCount = Array.isArray(params.views_used)
                ? params.views_used.length
                : params.input_mode === "four" ? 4 : params.input_mode ? 1 : "—";
            const totalSeconds = Number(stats.time?.total);
            const faces = stats.number_of_faces ?? manifest.number_of_faces;
            const vertices = stats.number_of_vertices ?? manifest.number_of_vertices;

            setGenerationDetail("generation-info-model", modelName);
            setGenerationDetail("generation-info-views", String(viewCount));
            setGenerationDetail(
                "generation-info-time",
                Number.isFinite(totalSeconds) ? totalSeconds.toFixed(1) + " s" : "—"
            );
            setGenerationDetail("generation-info-resolution", params.octree_resolution ?? "—");
            setGenerationDetail("generation-info-polygons", formatGenerationCount(faces));
            setGenerationDetail("generation-info-vertices", formatGenerationCount(vertices));

            const outputMeta = generationConsoleElement("generation-output-meta");
            if (!outputMeta) return;
            if (manifest.status !== "completed") {
                outputMeta.textContent = "Generation in progress";
                return;
            }

            const uid = manifest.generation_uid;
            const meshFilename = String(manifest.outputs?.mesh || "white_mesh.glb");
            outputMeta.textContent = "GLB · saved to source";
            fetch(
                "/static/" + encodeURIComponent(uid) + "/" + encodeURIComponent(meshFilename),
                {method: "HEAD", cache: "no-store"}
            ).then((response) => {
                if (!response.ok || generationConsoleUid !== uid) return;
                const size = formatGenerationBytes(response.headers.get("content-length"));
                if (size) outputMeta.textContent = "GLB · " + size + " · saved";
            }).catch(() => {});
        };

        const stopGenerationConsolePolling = () => {
            if (generationConsoleTimer !== null) {
                window.clearInterval(generationConsoleTimer);
                generationConsoleTimer = null;
            }
        };

        const renderGenerationManifest = (manifest) => {
            if (!manifest || manifest.generation_uid !== generationConsoleUid) return;

            updateGenerationDetails(manifest);

            if (
                manifest.storage_folder
                && !generationConsoleSeenEvents.has("__storage__")
            ) {
                generationConsoleSeenEvents.add("__storage__");
                appendGenerationConsoleLine(
                    "STORE",
                    "Target: " + manifest.storage_folder,
                    "muted"
                );
            }

            if (manifest.params && !generationConsoleParamsRendered) {
                generationConsoleParamsRendered = true;
                const params = manifest.params;
                appendGenerationConsoleLine(
                    "CONFIG",
                    "steps=" + params.steps
                        + " guidance=" + params.guidance_scale
                        + " octree=" + params.octree_resolution
                        + " chunks=" + params.num_chunks
                        + " seed=" + params.seed,
                    "command"
                );
            }

            (manifest.events || []).forEach((event) => {
                const eventKey = event.stage + "|" + event.at;
                if (generationConsoleSeenEvents.has(eventKey)) return;
                generationConsoleSeenEvents.add(eventKey);
                const kind = event.stage === "completed"
                    ? "success"
                    : event.stage === "failed" ? "error" : "info";
                appendGenerationConsoleLine(
                    generationConsoleStageLevels[event.stage] || "INFO",
                    event.message || event.stage,
                    kind,
                    event.at
                );
            });

            setGenerationConsoleProgress(
                manifest.progress,
                (manifest.events || []).at(-1)?.message || manifest.stage
            );

            const clock = generationConsoleElement("generation-console-clock");
            if (clock) clock.textContent = "LIVE " + generationConsoleElapsed();

            if (manifest.status === "completed") {
                const stats = manifest.stats || {};
                if (!generationConsoleSeenEvents.has("__mesh_stats__")) {
                    generationConsoleSeenEvents.add("__mesh_stats__");
                    appendGenerationConsoleLine(
                        "STATS",
                        "vertices=" + (stats.number_of_vertices ?? "-")
                            + " faces=" + (stats.number_of_faces ?? "-")
                            + " total=" + Number(stats.time?.total || 0).toFixed(2) + "s",
                        "success"
                    );
                    const storageFolder = String(
                        manifest.storage_folder || ("/static/" + generationConsoleUid)
                    ).replace(/[\/]+$/, "");
                    const meshFilename = String(manifest.outputs?.mesh || "white_mesh.glb");
                    appendGenerationConsoleLine(
                        "OUTPUT",
                        storageFolder + "/" + meshFilename,
                        "success"
                    );
                }
                setGenerationConsoleState("completed", "COMPLETED");
                setGenerationConsoleProgress(100, "3D model is ready");
                if (clock) clock.textContent = "SAVED TO SOURCE";
                stopGenerationConsolePolling();
            } else if (manifest.status === "failed") {
                if (!generationConsoleSeenEvents.has("__error__")) {
                    generationConsoleSeenEvents.add("__error__");
                    appendGenerationConsoleLine(
                        "ERROR",
                        String(manifest.error || "Unknown generation error").replace(/^'|'$/g, ""),
                        "error"
                    );
                }
                setGenerationConsoleState("failed", "FAILED");
                setGenerationConsoleProgress(100, "Generation stopped with an error");
                if (clock) clock.textContent = "ERROR SAVED TO MANIFEST";
                stopGenerationConsolePolling();
            } else {
                setGenerationConsoleState("running", "RUNNING");
            }
        };

        const pollGenerationManifest = async () => {
            const uid = generationConsoleUid;
            if (!uid) return;
            try {
                const response = await fetch(
                    "/static/" + encodeURIComponent(uid) + "/generation.json?t=" + Date.now(),
                    {cache: "no-store"}
                );
                if (!response.ok) {
                    generationConsolePollMisses += 1;
                    if (generationConsolePollMisses === 4) {
                        appendGenerationConsoleLine("QUEUE", "Waiting for the backend worker...", "muted");
                        setGenerationConsoleProgress(2, "Waiting in the Gradio queue");
                    }
                    return;
                }
                generationConsolePollMisses = 0;
                renderGenerationManifest(await response.json());
            } catch (error) {
                generationConsolePollMisses += 1;
                if (generationConsolePollMisses === 8) {
                    appendGenerationConsoleLine("WARN", "Manifest polling will retry automatically", "muted");
                }
            }
        };

        const startGenerationConsole = (uid, resumed = false) => {
            const root = generationConsoleElement("generation-console");
            if (!root || !uid) return;

            stopGenerationConsolePolling();
            generationConsoleUid = uid;
            generationConsoleStartedAt = Date.now();
            generationConsoleSeenEvents = new Set();
            generationConsoleParamsRendered = false;
            generationConsolePollMisses = 0;

            const log = generationConsoleElement("generation-console-log");
            if (log) log.replaceChildren();
            const job = generationConsoleElement("generation-console-job");
            if (job) job.textContent = "generation/" + uid;
            const mode = currentAppUrl().searchParams.get("tab") === "multi-view" ? "4-VIEW" : "1-VIEW";
            const modeElement = generationConsoleElement("generation-console-mode");
            if (modeElement) modeElement.textContent = mode;
            const clock = generationConsoleElement("generation-console-clock");
            if (clock) clock.textContent = "CONNECTING TO MANIFEST";
            resetGenerationDetails(mode);

            setGenerationConsoleState("running", resumed ? "RESTORING" : "STARTING");
            setGenerationConsoleProgress(1, resumed ? "Restoring generation state" : "Dispatching request");
            appendGenerationConsoleLine(
                resumed ? "RESUME" : "$",
                (resumed ? "restore" : "hunyuan3d.generate")
                    + " --mode " + mode.toLowerCase()
                    + " --uid " + uid,
                "command"
            );
            appendGenerationConsoleLine(
                "STORE",
                "Target: waiting for generation manifest",
                "muted"
            );

            window.setTimeout(pollGenerationManifest, 120);
            generationConsoleTimer = window.setInterval(pollGenerationManifest, 700);
        };

        const syncGenerationConsoleFromUrl = () => {
            const uid = currentAppUrl().searchParams.get("generation");
            if (uid && uid !== generationConsoleUid) startGenerationConsole(uid, true);
        };

        const beginGeneration = () => {
            const url = currentAppUrl();
            const uid = createGenerationUid();
            url.searchParams.set("generation", uid);
            window.history.pushState({}, "", url);
            activeGenerationRouteUid = uid;
            startGenerationConsole(uid);
        };

        const installGenerationRouting = () => {
            const buttonRoot = document.getElementById("generate-3d-button");
            if (!buttonRoot || buttonRoot.dataset.generationRouteWired === "true") return;
            buttonRoot.dataset.generationRouteWired = "true";
            buttonRoot.addEventListener("click", beginGeneration, {capture: true});
        };

        const promptTabButtons = () => Array.from(
            document.querySelectorAll('#prompt-mode-tabs button[role="tab"]')
        ).slice(0, tabRoutes.length);

        const syncTabFromUrl = () => {
            const buttons = promptTabButtons();
            if (buttons.length !== tabRoutes.length) return false;

            const url = currentAppUrl();
            const requestedSlug = url.searchParams.get("tab");
            const route = tabRoutes.find((item) => item.slug === requestedSlug) || tabRoutes[0];

            if (requestedSlug !== route.slug || url.href !== window.location.href) {
                url.searchParams.set("tab", route.slug);
                window.history.replaceState({}, "", url);
            }

            const target = buttons[route.index];
            if (target.getAttribute("aria-selected") !== "true") {
                target.click();
            }
            window.setTimeout(() => syncGenerateButtonCopy(route.index === 1), 0);
            return true;
        };

        const installTabRouting = () => {
            const buttons = promptTabButtons();
            if (buttons.length !== tabRoutes.length) return;

            buttons.forEach((button, index) => {
                if (button.dataset.urlRouteWired === "true") return;
                button.dataset.urlRouteWired = "true";
                button.addEventListener("click", () => {
                    const slug = tabRoutes[index].slug;
                    const url = currentAppUrl();
                    window.setTimeout(() => syncGenerateButtonCopy(index === 1), 0);
                    if (url.searchParams.get("tab") === slug) {
                        if (url.href !== window.location.href) {
                            window.history.replaceState({}, "", url);
                        }
                        return;
                    }
                    url.searchParams.set("tab", slug);
                    window.history.pushState({}, "", url);
                });
            });

            if (!tabRouteInitialized) {
                tabRouteInitialized = true;
                [0, 100, 400].forEach((delay) => {
                    window.setTimeout(syncTabFromUrl, delay);
                });
            }
        };

        const setModalOpen = (isOpen, shouldFocusClose = false) => {
            const element = modal();
            if (!element) return;
            element.classList.toggle("rtx-open", isOpen);
            element.setAttribute("aria-hidden", String(!isOpen));
            document.body.classList.toggle("rtx3090-modal-open", isOpen);
            if (isOpen && shouldFocusClose) {
                document.getElementById("rtx3090-modal-close")?.focus();
            }
        };

        const renderPresetSelection = (activeProfile) => {
            const element = modal();
            if (!element || !presetButtonIds[activeProfile]) return;

            element.querySelectorAll(".rtx3090-profile-card[data-profile]").forEach((card) => {
                const isActive = card.dataset.profile === activeProfile;
                card.classList.toggle("is-selected", isActive);
                card.setAttribute("aria-pressed", String(isActive));
            });

            Object.entries(presetButtonIds).forEach(([profile, buttonId]) => {
                const button = document.getElementById(buttonId);
                if (!button) return;
                const isActive = profile === activeProfile;
                button.classList.toggle("rtx-preset-action-active", isActive);
                button.setAttribute("aria-pressed", String(isActive));
            });
        };

        const syncPresetSelection = () => {
            const activeProfile = modal()?.querySelector(
                ".rtx-preset-status[data-profile]"
            )?.dataset.profile;
            if (activeProfile) renderPresetSelection(activeProfile);
        };

        const wirePresetCards = () => {
            const element = modal();
            if (!element) return;

            element.querySelectorAll(".rtx3090-profile-card[data-profile]").forEach((card) => {
                if (card.dataset.rtxPresetWired === "true") return;
                card.dataset.rtxPresetWired = "true";

                const applyCardPreset = () => {
                    const profile = card.dataset.profile;
                    const button = document.getElementById(presetButtonIds[profile]);
                    if (!button) return;
                    renderPresetSelection(profile);
                    button.click();
                };

                card.addEventListener("click", applyCardPreset);
                card.addEventListener("keydown", (event) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    applyCardPreset();
                });
            });

            Object.entries(presetButtonIds).forEach(([profile, buttonId]) => {
                const button = document.getElementById(buttonId);
                if (!button || button.dataset.rtxPresetWired === "true") return;
                button.dataset.rtxPresetWired = "true";
                button.addEventListener("click", () => renderPresetSelection(profile));
            });
        };

        let modalOpenedFromApp = false;

        const syncFromUrl = () => {
            const url = currentAppUrl();
            setModalOpen(url.searchParams.get("view") === "rtx3090");
        };

        const openModal = (event) => {
            const url = currentAppUrl();
            const shouldFocusClose = Boolean(event?.currentTarget?.matches?.(":focus-visible"));
            url.searchParams.set("view", "rtx3090");
            window.history.pushState({}, "", url);
            modalOpenedFromApp = true;
            setModalOpen(true, shouldFocusClose);
        };

        const closeModal = () => {
            const url = currentAppUrl();
            setModalOpen(false);
            if (url.searchParams.get("view") !== "rtx3090") {
                modalOpenedFromApp = false;
                return;
            }
            if (modalOpenedFromApp) {
                modalOpenedFromApp = false;
                window.history.back();
                return;
            }
            url.searchParams.delete("view");
            window.history.replaceState({}, "", url);
        };

        const wireTopbar = () => {
            const apiButton = document.getElementById("app-api-docs");
            if (apiButton && apiButton.dataset.uiWired !== "true") {
                apiButton.dataset.uiWired = "true";
                apiButton.addEventListener("click", () => {
                    document.querySelector("footer button.show-api")?.click();
                });
            }

            const settingsButton = document.getElementById("app-theme-settings");
            if (settingsButton && settingsButton.dataset.uiWired !== "true") {
                settingsButton.dataset.uiWired = "true";
                settingsButton.addEventListener("click", () => {
                    document.querySelector("footer button.settings")?.click();
                });
            }

            const rtxButton = document.getElementById("app-rtx-profile");
            if (rtxButton && rtxButton.dataset.uiWired !== "true") {
                rtxButton.dataset.uiWired = "true";
                rtxButton.addEventListener("click", openModal);
            }
        };

        const installFooterItem = () => {
            if (!modal()) return;
            const footer = Array.from(document.querySelectorAll("gradio-app footer, footer")).find(
                (element) => element.querySelector("button.show-api, a.built-with, button.settings")
            );
            if (!footer || document.getElementById(footerButtonId)) return;

            const builtWith = footer.querySelector("a.built-with");
            const settings = footer.querySelector("button.settings");
            const anchor = builtWith || settings;
            if (!anchor) return;

            const trigger = document.createElement("button");
            trigger.id = footerButtonId;
            trigger.type = "button";
            trigger.className = "rtx3090-footer-trigger";
            trigger.setAttribute("aria-haspopup", "dialog");
            trigger.innerHTML = '<span class="rtx3090-footer-icon ui-icon-slot" data-ui-icon="zap"></span><span>RTX 3090 - Cấu hình đề xuất</span>';
            trigger.addEventListener("click", openModal);

            const divider = document.createElement("div");
            divider.className = "divider rtx3090-footer-divider";
            divider.textContent = "·";

            footer.insertBefore(trigger, anchor);
            footer.insertBefore(divider, anchor);
        };

        const wireModal = () => {
            const element = modal();
            if (!element || element.dataset.rtxWired === "true") return;
            element.dataset.rtxWired = "true";
            element.setAttribute("role", "dialog");
            element.setAttribute("aria-modal", "true");
            element.setAttribute("aria-labelledby", "rtx3090-modal-title");
            element.setAttribute("aria-hidden", "true");
            element.addEventListener("click", (event) => {
                if (event.target === element) closeModal();
            });
            document.getElementById("rtx3090-modal-close")?.addEventListener("click", closeModal);
        };

        const observer = new MutationObserver(() => {
            installFooterItem();
            installUnifiedIcons();
            wireTopbar();
            wireModal();
            wirePresetCards();
            syncPresetSelection();
            installTabRouting();
            installGenerationRouting();
            syncGenerationConsoleFromUrl();
        });
        observer.observe(document.body, {childList: true, subtree: true});

        installFooterItem();
        installUnifiedIcons();
        wireTopbar();
        wireModal();
        wirePresetCards();
        syncPresetSelection();
        installTabRouting();
        installGenerationRouting();
        syncFromUrl();
        syncGenerationConsoleFromUrl();

        window.addEventListener("popstate", () => {
            const nextGenerationUid = currentAppUrl().searchParams.get("generation");
            if (nextGenerationUid !== activeGenerationRouteUid) {
                window.location.reload();
                return;
            }
            syncFromUrl();
            syncTabFromUrl();
            syncGenerationConsoleFromUrl();
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && modal()?.classList.contains("rtx-open")) {
                closeModal();
            }
        });
    }
    """

    file_out = None
    file_out2 = None

    with gr.Blocks(
        theme=gr.Theme(),
        title=title,
        analytics_enabled=False,
        css=custom_css,
        js=custom_js,
    ) as demo:
        gr.HTML(title_html)

        with gr.Row(elem_id='workspace-grid'):
            with gr.Column(scale=3, elem_id='input-panel'):
                gr.HTML('<div class="panel-heading">Input Views</div>')
                input_mode = gr.Textbox(value='single', visible=False, label='Input mode')
                with gr.Tabs(selected='tab_single_prompt', elem_id='prompt-mode-tabs'):
                    with gr.Tab('Single View', id='tab_single_prompt') as tab_ip:
                        gr.HTML("""
                        <div class="input-mode-guide">
                            <div class="input-mode-number ui-icon-slot" data-ui-icon="box" aria-hidden="true"></div>
                            <div class="input-mode-copy">
                                <strong>Best results with one front view</strong>
                                <span>Use a sharp subject on a clean or transparent background.</span>
                            </div>
                        </div>
                        """)
                        image = gr.Image(
                            label='Front View',
                            type='pil',
                            image_mode='RGBA',
                            height=240,
                            elem_classes=['single-image', 'ui-upload'],
                        )
                        gr.HTML('<div class="input-upload-meta">PNG or JPG · one front-facing image</div>')
                        caption = gr.State(None)
#                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
#                        caption = gr.Textbox(label='Text Prompt',
#                                             placeholder='HunyuanDiT will be used to generate image.',
#                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('Multi View (1–4)', id='tab_mv_prompt', visible=MV_MODE) as tab_mv:
                        gr.HTML("""
                        <div class="input-mode-guide">
                            <div class="input-mode-number ui-icon-slot" data-ui-icon="box" aria-hidden="true"></div>
                            <div class="input-mode-copy">
                                <strong>Best results with four views</strong>
                                <span>Upload Front, Back, Left and Right views of the same object.</span>
                            </div>
                        </div>
                        """)
                        with gr.Row(elem_classes='mv-upload-row'):
                            mv_image_front = gr.Image(label='1 · Front', type='pil', image_mode='RGBA', height=180,
                                                      min_width=100, elem_classes=['mv-image', 'ui-upload'])
                            mv_image_back = gr.Image(label='2 · Back', type='pil', image_mode='RGBA', height=180,
                                                     min_width=100, elem_classes=['mv-image', 'ui-upload'])
                        with gr.Row(elem_classes='mv-upload-row'):
                            mv_image_left = gr.Image(label='3 · Left', type='pil', image_mode='RGBA', height=180,
                                                     min_width=100, elem_classes=['mv-image', 'ui-upload'])
                            mv_image_right = gr.Image(label='4 · Right', type='pil', image_mode='RGBA', height=180,
                                                      min_width=100, elem_classes=['mv-image', 'ui-upload'])
                        gr.HTML("""
                        <div class="input-upload-meta input-upload-meta--stacked">
                            <div class="input-upload-meta-title">
                                <span class="ui-icon-slot" data-ui-icon="box" aria-hidden="true"></span>
                                <strong>Four synchronized views</strong>
                            </div>
                            <span class="input-upload-meta-subtitle">PNG or JPG · Front, Back, Left and Right</span>
                        </div>
                        """)

                with gr.Row(elem_classes='generate-actions'):
                    btn = gr.Button(
                        value='Generate 3D · 1 Image',
                        variant='primary',
                        min_width=100,
                        elem_id='generate-3d-button',
                    )
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100)

                if not MV_MODE:
                    with gr.Group(elem_id='mesh-download-card'):
                        file_out = gr.File(label="Generated mesh (direct download)", visible=True,
                                           interactive=False)
                        file_out2 = gr.File(label="File", visible=False)

                selected_options_tab = (
                    'tab_options' if TURBO_MODE
                    else 'tab_export' if HAS_PYMESHLAB
                    else 'tab_advanced_options'
                )
                with gr.Tabs(selected=selected_options_tab, elem_id='settings-tabs'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(
                            label='Generation Mode',
                            info='Recommendation: Turbo for most cases, \
Fast for very complex cases, Standard seldom use.',
                            choices=['Turbo', 'Fast', 'Standard'], 
                            value='Turbo')
                        decode_mode = gr.Radio(
                            label='Decoding Mode',
                            info='The resolution for exporting mesh from generated vectset',
                            choices=['Low', 'Standard', 'High'],
                            value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options', elem_id='advanced-settings-form'):
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                            elem_classes=['ui-control', 'ui-control-wide'],
                        )
                        with gr.Row(elem_classes=['ui-control-row', 'ui-control-row-compact']):
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps',
                                                  elem_classes='ui-control')
                            octree_resolution = gr.Slider(maximum=512, 
                                                          minimum=16, 
                                                          value=256, 
                                                          label='Octree Resolution',
                                                          elem_classes='ui-control')
                        with gr.Row(elem_classes=['ui-control-row', 'ui-control-row-compact']):
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100,
                                                  elem_classes='ui-control')
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100,
                                                   elem_classes='ui-control')
                        with gr.Row(elem_classes=['ui-control-row', 'ui-control-row-checks']):
                            check_box_rembg = gr.Checkbox(
                                value=not MV_MODE,
                                label='Remove Background',
                                visible=HAS_REMBG,
                                min_width=100,
                                elem_classes=['ui-control', 'ui-control-checkbox'])
                            randomize_seed = gr.Checkbox(
                                label="Randomize seed",
                                value=True,
                                min_width=100,
                                elem_classes=['ui-control', 'ui-control-checkbox'])
                    with gr.Tab("Export", id='tab_export', visible=HAS_PYMESHLAB):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', 
                                                    choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', 
                                                      value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6, elem_id='viewport-panel'):
                with gr.Tabs(selected='gen_mesh_panel', elem_id='output-tabs') as tabs_output:
                    with gr.Tab('Generation', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output', elem_id='mesh-viewer')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output', elem_id='mesh-export-viewer')
                    with gr.Tab('Statistics', id='stats_panel'):
                        stats = gr.Json(
                            {},
                            label='Mesh Stats',
                            elem_id='mesh-stats',
                            height=HTML_HEIGHT,
                            min_height=HTML_HEIGHT,
                            max_height=HTML_HEIGHT,
                        )

            with gr.Column(scale=3, visible=MV_MODE, elem_id='generation-console-panel'):
                gr.HTML(f"""
                <section id="generation-console" class="generation-console" data-state="idle">
                    <header class="generation-console-windowbar">
                        <span class="generation-console-title">
                            <strong>Generation Console</strong>
                            <span>REAL-TIME INFERENCE PIPELINE</span>
                        </span>
                        <span id="generation-console-status" class="generation-console-status">IDLE</span>
                    </header>
                    <div class="generation-console-progress-wrap">
                        <div class="generation-console-progress-meta">
                            <span id="generation-console-stage">Waiting for a generation request</span>
                            <span id="generation-console-percent">0%</span>
                        </div>
                        <div class="generation-console-progress-track">
                            <div id="generation-console-progress" class="generation-console-progress-bar"></div>
                        </div>
                    </div>
                    <div class="generation-console-jobbar">
                        <span class="ui-icon-slot" data-ui-icon="terminal" aria-hidden="true"></span>
                        <span id="generation-console-job" class="generation-console-job">No active generation</span>
                        <span id="generation-console-mode" class="generation-console-mode">WAITING</span>
                    </div>
                    <div id="generation-console-log" class="generation-console-log" aria-live="polite">
                        <div class="generation-console-line" data-kind="muted">
                            <span class="generation-console-time">+00.0s</span>
                            <span class="generation-console-level">READY</span>
                            <span class="generation-console-message">Upload input images, then press Generate 3D.</span>
                        </div>
                        <div class="generation-console-line" data-kind="muted">
                            <span class="generation-console-time">+00.0s</span>
                            <span class="generation-console-level">INFO</span>
                            <span class="generation-console-message">Console shows generation/inference progress, not model training.</span>
                        </div>
                    </div>
                    <div class="generation-console-footer">
                        <span><strong>Hunyuan3D-2mv</strong> · {runtime_label}</span>
                        <span id="generation-console-clock">LOCAL STORAGE READY</span>
                    </div>
                </section>
                <section class="generation-details-card" aria-labelledby="generation-info-title">
                    <h3 id="generation-info-title" class="dashboard-card-title">Generation Info</h3>
                    <div class="generation-details-grid">
                        <div class="generation-detail"><span>Model</span><strong id="generation-info-model">—</strong></div>
                        <div class="generation-detail"><span>Views</span><strong id="generation-info-views">—</strong></div>
                        <div class="generation-detail"><span>Time</span><strong id="generation-info-time">—</strong></div>
                        <div class="generation-detail"><span>Resolution</span><strong id="generation-info-resolution">—</strong></div>
                        <div class="generation-detail"><span>Polygons</span><strong id="generation-info-polygons">—</strong></div>
                        <div class="generation-detail"><span>Vertices</span><strong id="generation-info-vertices">—</strong></div>
                    </div>
                </section>
                """)

                if MV_MODE:
                    with gr.Group(elem_id='generation-output-card'):
                        gr.HTML("""
                        <div class="dashboard-card-title">
                            <span>Outputs</span>
                            <span id="generation-output-meta" class="generation-output-meta" role="status" aria-live="polite">Awaiting generated mesh</span>
                        </div>
                        """, elem_classes='generation-output-heading')
                        file_out = gr.File(
                            label="Generated Mesh",
                            visible=True,
                            interactive=False,
                            show_label=False,
                            elem_classes='generation-output-file',
                        )
                        file_out2 = gr.File(label="File", visible=False)

            with gr.Column(scale=2, visible=not MV_MODE):
                with gr.Tabs(selected='tab_img_gallery'):
                    with gr.Tab('Image to 3D Gallery', 
                                id='tab_img_gallery', 
                                visible=not MV_MODE):
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)

        with gr.Column(
            elem_id='rtx3090-modal',
            visible=MV_MODE and args.device == 'cuda',
        ):
            with gr.Column(elem_classes='rtx3090-modal-panel'):
                gr.HTML("""
                <div class="rtx3090-modal-header">
                    <div class="rtx3090-header-main">
                        <span class="rtx3090-header-icon ui-icon-slot" data-ui-icon="zap" aria-hidden="true"></span>
                        <h2 id="rtx3090-modal-title">RTX 3090 · Cấu hình đề xuất</h2>
                        <span class="rtx3090-header-scope">1 ảnh &amp; 4 ảnh</span>
                    </div>
                    <div class="rtx3090-header-actions">
                        <span class="rtx3090-verified">
                            <i class="rtx3090-verified-dot"></i>
                            Đã kiểm tra
                        </span>
                        <span class="rtx3090-preset-count"><b>2</b> preset</span>
                        <button id="rtx3090-modal-close" type="button" aria-label="Đóng cửa sổ cấu hình">
                            <svg class="rtx3090-close-icon" width="100%" height="100%" viewBox="0 0 5 5" version="1.1" xmlns="http://www.w3.org/2000/svg" xml:space="preserve" style="fill: currentcolor; fill-rule: evenodd; clip-rule: evenodd; stroke-linejoin: round; stroke-miterlimit: 2;" aria-hidden="true" focusable="false">
                                <g>
                                    <path d="M3.789,0.09C3.903,-0.024 4.088,-0.024 4.202,0.09L4.817,0.705C4.931,0.819 4.931,1.004 4.817,1.118L1.118,4.817C1.004,4.931 0.819,4.931 0.705,4.817L0.09,4.202C-0.024,4.088 -0.024,3.903 0.09,3.789L3.789,0.09Z"></path>
                                    <path d="M4.825,3.797C4.934,3.907 4.934,4.084 4.825,4.193L4.193,4.825C4.084,4.934 3.907,4.934 3.797,4.825L0.082,1.11C-0.027,1.001 -0.027,0.823 0.082,0.714L0.714,0.082C0.823,-0.027 1.001,-0.027 1.11,0.082L4.825,3.797Z"></path>
                                </g>
                            </svg>
                        </button>
                    </div>
                </div>
                """, elem_classes='rtx3090-modal-header-block')
                gr.HTML("""
                <div class="rtx3090-api-intro">
                    <div class="rtx3090-context-tabs">
                        <span class="active"><i class="ui-icon-slot" data-ui-icon="zap" aria-hidden="true"></i>RTX 3090</span>
                        <span><i class="ui-icon-slot" data-ui-icon="memory" aria-hidden="true"></i>24 GB VRAM</span>
                        <span>1 ảnh</span>
                        <span>4 ảnh</span>
                        <span>FP16</span>
                    </div>
                    <p>
                        Chọn một trong các cấu hình dưới đây để tối ưu chất lượng mesh trên máy hiện tại.<br>
                        Các giá trị sẽ được cập nhật trực tiếp vào <strong>Advanced Options</strong>.
                    </p>
                </div>
                """, elem_classes='rtx3090-intro-block')
                gr.HTML("""
                <div class="rtx3090-section-heading">
                    <b>1. Xác nhận cấu hình máy.</b>
                    <span>Hai preset này dành riêng cho RTX 3090 24 GB đang chạy WebUI.</span>
                </div>
                """, elem_classes='rtx3090-section-one')
                gr.HTML("""
                <div class="rtx3090-machine-strip">
                    <div class="rtx3090-machine-badge">24 GB</div>
                    <div class="rtx3090-machine-copy">
                        <strong>NVIDIA GeForce RTX 3090 · CUDA · FP16</strong>
                        <span>
                            Hai mức dưới đây đã được kiểm tra end-to-end trên chính máy này,
                            không gặp lỗi thiếu VRAM với cả đầu vào 1 ảnh và 4 ảnh.
                        </span>
                    </div>
                    <span class="rtx3090-machine-check ui-icon-slot" data-ui-icon="check" aria-hidden="true"></span>
                </div>
                """, elem_classes='rtx3090-machine-block')
                gr.HTML("""
                <div class="rtx3090-section-heading">
                    <b>2. Chọn mức phù hợp rồi bấm Áp dụng.</b>
                    <span>Thông số bên dưới dùng chung cho cả chế độ 1 ảnh và 4 ảnh.</span>
                </div>
                """, elem_classes='rtx3090-section-two')
                gr.HTML("""
                <div class="rtx3090-profile-grid">
                    <article
                        class="rtx3090-profile-card safe is-selected"
                        data-profile="safe"
                        role="button"
                        tabindex="0"
                        aria-pressed="true"
                        aria-controls="advanced-settings-form"
                        aria-label="Chọn và áp dụng preset 256 mặc định an toàn"
                    >
                        <div class="rtx3090-profile-heading">
                            <h3>256 · Mặc định an toàn</h3>
                            <span class="rtx3090-profile-selector" aria-hidden="true"></span>
                        </div>
                        <p>Dùng cho lần chạy đầu hoặc khi cần ưu tiên ổn định và tốc độ.</p>
                        <div class="rtx3090-profile-values">
                            <span><b>30</b><small>Steps</small></span>
                            <span><b>5.0</b><small>Guidance</small></span>
                            <span><b>256</b><small>Octree</small></span>
                            <span><b>8000</b><small>Chunks</small></span>
                        </div>
                    </article>
                    <article
                        class="rtx3090-profile-card quality"
                        data-profile="quality"
                        role="button"
                        tabindex="0"
                        aria-pressed="false"
                        aria-controls="advanced-settings-form"
                        aria-label="Chọn và áp dụng preset 384 chất lượng cao"
                    >
                        <div class="rtx3090-profile-heading">
                            <h3>384 · Chất lượng cao</h3>
                            <span class="rtx3090-profile-selector" aria-hidden="true"></span>
                        </div>
                        <p>Dùng khi ảnh đầu vào đã đúng và cần mesh dày, mịn hơn.</p>
                        <div class="rtx3090-profile-values">
                            <span><b>30</b><small>Steps</small></span>
                            <span><b>5.0</b><small>Guidance</small></span>
                            <span><b>384</b><small>Octree</small></span>
                            <span><b>8000</b><small>Chunks</small></span>
                        </div>
                    </article>
                </div>
                """, elem_classes='rtx3090-profiles-block')
                with gr.Row(elem_classes='rtx-preset-actions'):
                    rtx_safe_preset = gr.Button(
                        value='Áp dụng · 256 an toàn',
                        min_width=160,
                        elem_id='rtx3090-safe-preset',
                    )
                    rtx_quality_preset = gr.Button(
                        value='Áp dụng · 384 chất lượng cao',
                        variant='primary',
                        min_width=180,
                        elem_id='rtx3090-quality-preset',
                    )
                rtx_preset_status = gr.HTML(
                    get_rtx3090_preset('safe')[-1],
                    elem_classes='rtx3090-status-block',
                )
                gr.HTML("""
                <div class="rtx3090-modal-note">
                    <span class="rtx3090-note-icon ui-icon-slot" data-ui-icon="info" aria-hidden="true"></span>
                    <p>
                        <strong>Lưu ý:</strong> Seed không làm tăng VRAM. Bật Randomize để thử biến thể,
                        tắt để tái tạo đúng kết quả.<br>
                        Chunks 8000 cân bằng tốc độ/bộ nhớ và không làm mesh đẹp hơn khi tăng quá cao.<br>
                        Không đặt Octree 512 làm mặc định trên RTX 3090 24 GB.
                    </p>
                </div>
                """, elem_classes='rtx3090-note-block')

        assert file_out is not None and file_out2 is not None

        tab_ip.select(
            fn=lambda: (
                'single',
                gr.update(value='Generate 3D · 1 Image'),
            ),
            outputs=[input_mode, btn],
            queue=False,
            api_name=False,
        )
        tab_mv.select(
            fn=lambda: (
                'four',
                gr.update(value='Generate 3D · 4 Images'),
            ),
            outputs=[input_mode, btn],
            queue=False,
            api_name=False,
        )

        demo.load(
            fn=restore_generation_from_request,
            outputs=[
                input_mode,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                file_out,
                html_gen_mesh,
                stats,
                seed,
                num_steps,
                cfg_scale,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                btn,
            ],
            queue=False,
            api_name=False,
        )

        rtx_preset_outputs = [
            num_steps,
            cfg_scale,
            octree_resolution,
            num_chunks,
            rtx_preset_status,
        ]
        rtx_quality_preset.click(
            fn=lambda: get_rtx3090_preset('quality'),
            outputs=rtx_preset_outputs,
            queue=False,
            api_name=False,
        )
        rtx_safe_preset.click(
            fn=lambda: get_rtx3090_preset('safe'),
            outputs=rtx_preset_outputs,
            queue=False,
            api_name=False,
        )
        #if HAS_T2I:
        #    tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                input_mode,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                input_mode,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], 
                           outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, 
                            reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=True)
            else:
                mesh = trimesh.load(file_out)
                floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker = get_postprocessors()
                mesh = floater_remove_worker(mesh)
                mesh = degenerate_face_remove_worker(mesh)
                if reduce_face:
                    mesh = face_reduce_worker(mesh, target_face_num)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=False)
            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mv')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mv')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='./save_dir')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--use_safetensors', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--variant', type=str, default='fp16')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    args = parser.parse_args()
    
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in f'{args.model_path}/{args.subfolder}'.lower()
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 820 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div class='viewer-empty-state' style='height: {HTML_HEIGHT}px;'>
      <div class='viewer-empty-copy'>
        <span class='viewer-empty-mark ui-icon-slot' data-ui-icon='box' aria-hidden='true'></span>
        <strong>3D preview is ready</strong>
        <p>Upload your input views and generate a mesh to begin.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; border-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            # Apply torchvision fix before importing basicsr/RealESRGAN
            print("Applying torchvision compatibility fix for texture generation...")
            try:
                from torchvision_fix import apply_fix
                fix_result = apply_fix()
                if not fix_result:
                    print("Warning: Torchvision fix may not have been applied successfully")
            except Exception as fix_error:
                print(f"Warning: Failed to apply torchvision fix: {fix_error}")
            
            # from hy3dgen.texgen import Hunyuan3DPaintPipeline
            # texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            # if args.low_vram_mode:
            #     texgen_worker.enable_model_cpu_offload()

            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            tex_pipeline = Hunyuan3DPaintPipeline(conf)
        
            # Not help much, ignore for now.
            # if args.compile:
            #     texgen_worker.models['delight_model'].pipeline.unet.compile()
            #     texgen_worker.models['delight_model'].pipeline.vae.compile()
            #     texgen_worker.models['multiview_model'].pipeline.unet.compile()
            #     texgen_worker.models['multiview_model'].pipeline.vae.compile()
            
            HAS_TEXTUREGEN = True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading texture generator: {e}")
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    HAS_T2I = True
    if args.enable_t23d:
        # Optional dependency required only when --enable_t23d is requested.
        from hy3dgen.text2image import HunyuanDiTPipeline  # pyright: ignore[reportMissingImports]

        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True

    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.pipelines import export_to_trimesh

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but PyTorch cannot access an NVIDIA GPU.')

    pipeline_dtype = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[args.dtype]
    print(
        f'Loading shape model {args.model_path}/{args.subfolder} '
        f'(safetensors={args.use_safetensors}, variant={args.variant}, dtype={args.dtype})'
    )
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=args.use_safetensors,
        variant=cast(str, args.variant or None),
        device=args.device,
        dtype=pipeline_dtype,
    )
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile:
        i23d_worker.compile()

    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()

    @app.get('/health')
    def health():
        return {
            'status': 'ready',
            'pid': os.getpid(),
            'model': args.model_path,
            'subfolder': args.subfolder,
            'multiview': MV_MODE,
            'device': args.device,
        }

    @app.get('/generation-viewer/{generation_uid}', response_class=HTMLResponse)
    def generation_viewer(generation_uid: str):
        try:
            generation_uid = str(uuid.UUID(generation_uid))
        except (ValueError, AttributeError, TypeError) as exc:
            raise HTTPException(status_code=404, detail='Generation not found') from exc

        mesh_path = stored_generation_file(
            os.path.join(SAVE_DIR, generation_uid),
            'white_mesh.glb',
        )
        if not mesh_path:
            raise HTTPException(status_code=404, detail='Generated mesh not found')

        document = render_model_viewer_document(
            f'/static/{generation_uid}/white_mesh.glb',
            HTML_HEIGHT - 10,
            HTML_WIDTH,
            textured=False,
        )
        return HTMLResponse(
            document,
            headers={'Cache-Control': 'no-store'},
        )
    
    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    demo = build_app()
    demo.queue(max_size=4, default_concurrency_limit=1)
    app = gr.mount_gradio_app(app, demo, path="/")
    print(f'Web UI ready at http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)
