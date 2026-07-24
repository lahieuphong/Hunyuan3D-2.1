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
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from glob import glob
from html import escape
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uuid
import numpy as np

from hy3dshape.utils import logger
from webui import load_ui_assets, render_history_modal, render_topbar
from webui.history import list_generation_history

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
RTX3090_GENERATION_MODES = {5: 'Turbo', 10: 'Fast', 30: 'Standard'}
RTX3090_DECODING_MODES = {196: 'Low', 256: 'Standard', 384: 'High'}
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


def resolve_rtx3090_mode(value, modes):
    """Map an exact integer control value to its named UI mode."""
    if isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        return None
    return modes.get(int(numeric_value))


def get_rtx3090_preset(profile, *, saved=False):
    """Return Gradio values and an explanatory status card for a GPU preset."""
    if profile not in RTX3090_PRESETS:
        raise ValueError(f"Unknown RTX 3090 preset: {profile}")

    preset = RTX3090_PRESETS[profile]
    is_quality = profile == 'quality'
    profile_name = 'Chất lượng cao' if is_quality else 'Mặc định an toàn'
    profile_class = 'quality' if is_quality else 'safe'
    status_label = 'Saved values' if saved else '&#272;ang d&#249;ng'
    status_html = f"""
    <div class="rtx-preset-status {profile_class}" data-profile="{profile_class}">
        <div class="rtx-preset-status-heading">
            <div class="rtx-preset-status-title">
                <span class="rtx-preset-status-check ui-icon-slot" data-ui-icon="check" aria-hidden="true"></span>
                <span>RTX 3090 · 1 ảnh &amp; 4 ảnh · {profile_name}</span>
            </div>
            <span class="rtx-preset-current">{status_label}</span>
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
        resolve_rtx3090_mode(preset['steps'], RTX3090_GENERATION_MODES),
        resolve_rtx3090_mode(
            preset['octree_resolution'],
            RTX3090_DECODING_MODES,
        ),
        status_html,
    )


def resolve_rtx3090_profile(steps, guidance_scale, octree_resolution, num_chunks):
    """Resolve a preset from its complete parameter tuple."""
    raw_values = (steps, guidance_scale, octree_resolution, num_chunks)
    if any(isinstance(value, bool) for value in raw_values):
        return None

    try:
        normalized_integers = []
        for value in (steps, octree_resolution, num_chunks):
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or not numeric_value.is_integer():
                return None
            normalized_integers.append(int(numeric_value))
        normalized_guidance = float(guidance_scale)
    except (OverflowError, TypeError, ValueError):
        return None

    if not math.isfinite(normalized_guidance):
        return None

    normalized_steps, normalized_octree, normalized_chunks = normalized_integers
    for profile, preset in RTX3090_PRESETS.items():
        if (
            normalized_steps == preset['steps']
            and math.isclose(
                normalized_guidance,
                float(preset['guidance_scale']),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and normalized_octree == preset['octree_resolution']
            and normalized_chunks == preset['num_chunks']
        ):
            return profile
    return None


def get_rtx3090_status(
    steps,
    guidance_scale,
    octree_resolution,
    num_chunks,
    *,
    saved=False,
):
    """Render the preset status from the actual values shown in the form."""
    profile = resolve_rtx3090_profile(
        steps,
        guidance_scale,
        octree_resolution,
        num_chunks,
    )
    if profile:
        return get_rtx3090_preset(profile, saved=saved)[-1]

    profile_name = 'Custom saved configuration' if saved else 'Custom configuration'
    status_label = 'Saved values' if saved else 'In use'
    display_values = tuple(
        '&mdash;' if value is None else escape(str(value))
        for value in (steps, guidance_scale, octree_resolution, num_chunks)
    )
    display_steps, display_guidance, display_octree, display_chunks = display_values
    return f"""
    <div class="rtx-preset-status custom" data-profile="custom">
        <div class="rtx-preset-status-heading">
            <div class="rtx-preset-status-title">
                <span class="rtx-preset-status-check ui-icon-slot" data-ui-icon="settings" aria-hidden="true"></span>
                <span>RTX 3090 - 1 &amp; 4 views - {profile_name}</span>
            </div>
            <span class="rtx-preset-current">{status_label}</span>
        </div>
        <div class="rtx-preset-values">
            <span><b>{display_steps}</b><small>Steps</small></span>
            <span><b>{display_guidance}</b><small>Guidance</small></span>
            <span><b>{display_octree}</b><small>Octree</small></span>
            <span><b>{display_chunks}</b><small>Chunks</small></span>
        </div>
    </div>
    """


def get_rtx3090_form_state(
    steps,
    guidance_scale,
    octree_resolution,
    num_chunks,
):
    """Return synchronized Turbo radios and RTX preset status markup."""
    return (
        resolve_rtx3090_mode(steps, RTX3090_GENERATION_MODES),
        resolve_rtx3090_mode(octree_resolution, RTX3090_DECODING_MODES),
        get_rtx3090_status(
            steps,
            guidance_scale,
            octree_resolution,
            num_chunks,
        ),
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
    raw_save_folder = os.path.abspath(save_folder)
    raw_candidate = os.path.abspath(os.path.join(raw_save_folder, str(filename)))
    if os.path.islink(raw_save_folder) or os.path.islink(raw_candidate):
        return None
    save_folder = os.path.realpath(raw_save_folder)
    candidate = os.path.realpath(raw_candidate)
    try:
        if (
            os.path.commonpath([save_folder, candidate]) != save_folder
            or os.path.dirname(candidate) != save_folder
        ):
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
    editable = (
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        '<span data-history-review-active="false"></span>',
    )
    try:
        generation_uid = generation_uid_query_from_request(request)
    except gr.Error as error:
        logger.warning("Could not restore generation URL: %s", error)
        return editable
    if not generation_uid:
        return editable

    save_folder = os.path.join(SAVE_DIR, generation_uid)
    manifest_path = stored_generation_file(save_folder, 'generation.json')
    if not manifest_path:
        logger.warning("Saved generation was not found: %s", generation_uid)
        return editable

    try:
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.exception("Could not restore generation manifest: %s", manifest_path)
        return editable

    if (
        not isinstance(manifest, dict)
        or manifest.get('schema_version') != 1
        or manifest.get('generation_uid') != generation_uid
        or not isinstance(manifest.get('events', []), list)
    ):
        logger.warning("Saved generation manifest is invalid: %s", manifest_path)
        return editable

    raw_params = manifest.get('params')
    raw_inputs = manifest.get('inputs')
    raw_outputs = manifest.get('outputs')
    raw_stats = manifest.get('stats')
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    inputs: dict[str, Any] = raw_inputs if isinstance(raw_inputs, dict) else {}
    outputs: dict[str, Any] = raw_outputs if isinstance(raw_outputs, dict) else {}
    stats: dict[str, Any] = raw_stats if isinstance(raw_stats, dict) else {}
    raw_input_mode = manifest.get('input_mode') or params.get('input_mode') or 'single'
    input_mode = (
        'four'
        if isinstance(raw_input_mode, str)
        and raw_input_mode in {'four', '4-view', 'multi-view'}
        else 'single'
    )
    if input_mode == 'four' and not MV_MODE:
        logger.warning(
            "Cannot restore a multi-view generation in single-view mode: %s",
            generation_uid,
        )
        return editable

    def numeric_param(name, default):
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        try:
            return value if math.isfinite(float(value)) else default
        except (OverflowError, TypeError, ValueError):
            return default

    def input_path(*filenames):
        for filename in filenames:
            path = stored_generation_file(save_folder, filename)
            if path:
                return path
        return None

    front_image = input_path(
        inputs.get('front'),
        inputs.get('image'),
        'input_front.png',
        'input_image.png',
    )
    back_image = input_path(inputs.get('back'), 'input_back.png')
    left_image = input_path(inputs.get('left'), 'input_left.png')
    right_image = input_path(inputs.get('right'), 'input_right.png')

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

    seed_value = numeric_param('seed', 1234)
    steps_value = numeric_param('steps', 30)
    guidance_value = numeric_param('guidance_scale', 5.0)
    octree_value = numeric_param('octree_resolution', 256)
    chunks_value = numeric_param('num_chunks', 8000)
    rembg_value = (
        params.get('check_box_rembg')
        if isinstance(params.get('check_box_rembg'), bool)
        else not MV_MODE
    )
    randomize_seed_value = (
        params.get('randomize_seed')
        if isinstance(params.get('randomize_seed'), bool)
        else False
    )
    preset_status = get_rtx3090_status(
        steps_value,
        guidance_value,
        octree_value,
        chunks_value,
        saved=True,
    )
    generation_mode_value = resolve_rtx3090_mode(
        steps_value,
        RTX3090_GENERATION_MODES,
    )
    decoding_mode_value = resolve_rtx3090_mode(
        octree_value,
        RTX3090_DECODING_MODES,
    )

    return (
        input_mode,
        gr.update(
            value=front_image if input_mode == 'single' else None,
            interactive=False,
        ),
        gr.update(
            value=front_image if input_mode == 'four' else None,
            interactive=False,
        ),
        gr.update(
            value=back_image if input_mode == 'four' else None,
            interactive=False,
        ),
        gr.update(
            value=left_image if input_mode == 'four' else None,
            interactive=False,
        ),
        gr.update(
            value=right_image if input_mode == 'four' else None,
            interactive=False,
        ),
        gr.update(value=mesh_path, interactive=False),
        viewer_html,
        stats,
        gr.update(value=seed_value, interactive=False),
        gr.update(value=steps_value, interactive=False),
        gr.update(value=guidance_value, interactive=False),
        gr.update(value=octree_value, interactive=False),
        gr.update(value=rembg_value, interactive=False),
        gr.update(value=chunks_value, interactive=False),
        gr.update(value=randomize_seed_value, interactive=False),
        gr.update(
            value=(
                'Generate 3D · 4 Images' if input_mode == 'four'
                else 'Generate 3D · 1 Image'
            ),
            interactive=False,
        ),
        preset_status,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=generation_mode_value, interactive=False),
        gr.update(value=decoding_mode_value, interactive=False),
        f'<span data-history-review-active="true" data-input-mode="{input_mode}"></span>',
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
            'randomize_seed': bool(randomize_seed),
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


def mount_gradio_at_root(app: FastAPI, demo: gr.Blocks) -> FastAPI:
    """Mount Gradio at the ASGI root without redirecting / to //.

    With Gradio 5.33, passing path="/" makes the mounted application issue a
    temporary redirect to a double-slash URL. An empty mount path is the
    Starlette-supported root mount and keeps browser URLs canonical. The
    middleware also redirects legacy root URLs made only of repeated slashes.
    """
    @app.middleware('http')
    async def canonicalize_root_url(request: Request, call_next):
        request_path = request.url.path
        if len(request_path) > 1 and not request_path.strip('/'):
            canonical_url = request.url.replace(path='/')
            return RedirectResponse(url=str(canonical_url), status_code=308)
        return await call_next(request)

    return gr.mount_gradio_app(app, demo, path="")


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

    title_html = (
        render_topbar(brand_name, workspace_title, rtx_profile_action)
        + render_history_modal()
    )
    custom_css, custom_js = load_ui_assets()

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
        history_review_state = gr.HTML(
            '<span data-history-review-active="false"></span>',
            visible=False,
            elem_id='history-review-state',
        )

        with gr.Row(elem_id='workspace-grid'):
            with gr.Column(scale=3, elem_id='input-panel'):
                gr.HTML('<div class="panel-heading">Input Views</div>')
                input_mode = gr.Textbox(value='single', visible=False, label='Input mode')
                with gr.Tabs(selected='tab_single_prompt', elem_id='prompt-mode-tabs'):
                    with gr.Tab('Single View', id='tab_single_prompt') as tab_ip:
                        gr.HTML("""
                        <div class="input-mode-guide">
                            <span class="input-mode-number ui-brand-mark" aria-hidden="true"><img class="app-context-logo" src="/favicon.ico" alt="" draggable="false"></span>
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
                            interactive=False,
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
                            <span class="input-mode-number ui-brand-mark" aria-hidden="true"><img class="app-context-logo" src="/favicon.ico" alt="" draggable="false"></span>
                            <div class="input-mode-copy">
                                <strong>Best results with four views</strong>
                                <span>Upload Front, Back, Left and Right views of the same object.</span>
                            </div>
                        </div>
                        """)
                        with gr.Row(elem_classes='mv-upload-row'):
                            mv_image_front = gr.Image(label='1 · Front', type='pil', image_mode='RGBA', height=180,
                                                      min_width=100, interactive=False, elem_id='mv-image-front',
                                                      elem_classes=['mv-image', 'ui-upload'])
                            mv_image_back = gr.Image(label='2 · Back', type='pil', image_mode='RGBA', height=180,
                                                     min_width=100, interactive=False, elem_id='mv-image-back',
                                                     elem_classes=['mv-image', 'ui-upload'])
                        with gr.Row(elem_classes='mv-upload-row'):
                            mv_image_left = gr.Image(label='3 · Left', type='pil', image_mode='RGBA', height=180,
                                                     min_width=100, interactive=False, elem_id='mv-image-left',
                                                     elem_classes=['mv-image', 'ui-upload'])
                            mv_image_right = gr.Image(label='4 · Right', type='pil', image_mode='RGBA', height=180,
                                                      min_width=100, interactive=False, elem_id='mv-image-right',
                                                      elem_classes=['mv-image', 'ui-upload'])
                        gr.HTML("""
                        <div class="input-upload-meta input-upload-meta--stacked">
                            <div class="input-upload-meta-title">
                                <span class="input-upload-brand-mark ui-brand-mark" aria-hidden="true"><img class="app-context-logo" src="/favicon.ico" alt="" draggable="false"></span>
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
                                        min_width=100,
                                        elem_id='generate-textured-3d-button')

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
                            value='Turbo',
                            elem_id='generation-mode')
                        decode_mode = gr.Radio(
                            label='Decoding Mode',
                            info='The resolution for exporting mesh from generated vectset',
                            choices=['Low', 'Standard', 'High'],
                            value='Standard',
                            elem_id='decoding-mode')
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
                    get_rtx3090_status(5 if TURBO_MODE else 30, 5.0, 256, 8000),
                    elem_classes='rtx3090-status-block',
                )
                gr.HTML("""
                <div class="rtx3090-modal-note">
                    <span class="rtx3090-note-icon ui-icon-slot" data-ui-icon="warning" aria-hidden="true"></span>
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

        restore_event = demo.load(
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
                rtx_preset_status,
                rtx_safe_preset,
                rtx_quality_preset,
                gen_mode,
                decode_mode,
                history_review_state,
            ],
            queue=False,
            show_progress='hidden',
            api_name=False,
        )

        rtx_preset_outputs = [
            num_steps,
            cfg_scale,
            octree_resolution,
            num_chunks,
            gen_mode,
            decode_mode,
            rtx_preset_status,
        ]
        rtx_quality_preset.click(
            fn=lambda: get_rtx3090_preset('quality'),
            outputs=rtx_preset_outputs,
            queue=False,
            show_progress='hidden',
            api_name=False,
        )
        rtx_safe_preset.click(
            fn=lambda: get_rtx3090_preset('safe'),
            outputs=rtx_preset_outputs,
            queue=False,
            show_progress='hidden',
            api_name=False,
        )

        for rtx_setting_control in (
            num_steps,
            cfg_scale,
            octree_resolution,
            num_chunks,
        ):
            rtx_setting_control.input(
                fn=get_rtx3090_form_state,
                inputs=[
                    num_steps,
                    cfg_scale,
                    octree_resolution,
                    num_chunks,
                ],
                outputs=[gen_mode, decode_mode, rtx_preset_status],
                queue=False,
                show_progress='hidden',
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
            outputs=[file_out, html_gen_mesh, stats, seed],
            show_progress='hidden' if MV_MODE else 'full',
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
            show_progress='hidden',
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
            show_progress='hidden',
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
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed],
            show_progress='full',
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
            show_progress='hidden',
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
            show_progress='hidden',
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.input(
            on_gen_mode_change,
            inputs=[gen_mode],
            outputs=[num_steps],
        ).then(
            fn=get_rtx3090_form_state,
            inputs=[num_steps, cfg_scale, octree_resolution, num_chunks],
            outputs=[gen_mode, decode_mode, rtx_preset_status],
            queue=False,
            show_progress='hidden',
            api_name=False,
        )

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.input(
            on_decode_mode_change,
            inputs=[decode_mode],
            outputs=[octree_resolution],
        ).then(
            fn=get_rtx3090_form_state,
            inputs=[num_steps, cfg_scale, octree_resolution, num_chunks],
            outputs=[gen_mode, decode_mode, rtx_preset_status],
            queue=False,
            show_progress='hidden',
            api_name=False,
        )

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
        <span class='viewer-empty-mark ui-brand-mark' aria-hidden='true'><img class='app-context-logo' src='/favicon.ico' alt='' draggable='false'></span>
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

    @app.get('/api/generation-history')
    def generation_history_api(limit: int = 200):
        return JSONResponse(
            list_generation_history(SAVE_DIR, limit=limit),
            headers={'Cache-Control': 'no-store'},
        )

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
    app = mount_gradio_at_root(app, demo)
    print(f'Web UI ready at http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)
