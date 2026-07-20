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
from urllib.parse import parse_qs, urlparse

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid
import numpy as np

from hy3dshape.utils import logger
MAX_SEED = 1e7
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
    import os, spaces, subprocess, sys, shlex
    print("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    os.system("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    print('install custom')
    subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"),
                   check=True)
else:
    """
    Define a dummy `spaces` module with a GPU decorator class for local environment.

    The GPU decorator is a no-op that simply returns the decorated function unchanged.
    This allows code that uses the `spaces.GPU` decorator to run without modification locally.
    """
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func 


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
    detail = (
        'Octree 384 tăng mật độ bề mặt. Đã chạy thành công trên máy này: '
        'khoảng 58 giây với 1 ảnh và 77 giây với 4 ảnh.'
        if is_quality else
        'Octree 256 là mức khởi đầu ổn định nhất, phù hợp để kiểm tra ảnh đầu vào '
        'trước khi xuất bản chất lượng cao.'
    )
    status_html = f"""
    <div class="rtx-preset-status {profile_class}">
        <div class="rtx-preset-status-title">
            <span class="rtx-preset-dot"></span>
            RTX 3090 · 1 ảnh &amp; 4 ảnh · {profile_name}
        </div>
        <div class="rtx-preset-values">
            <span><b>{preset['steps']}</b> Steps</span>
            <span><b>{preset['guidance_scale']}</b> Guidance</span>
            <span><b>{preset['octree_resolution']}</b> Octree</span>
            <span><b>{preset['num_chunks']}</b> Chunks</span>
        </div>
        <div class="rtx-preset-detail">{detail}</div>
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


def generation_uid_from_request(request=None):
    if request is not None:
        referer = request.headers.get('referer', '')
        generation_values = parse_qs(urlparse(referer).query).get('generation', [])
        if generation_values:
            return normalize_generation_uid(generation_values[0])
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
    'request_received': 3,
    'validating_input': 8,
    'input_validated': 15,
    'input_saved': 22,
    'preprocessing_input': 30,
    'input_ready': 38,
    'shape_generation': 45,
    'extracting_mesh': 78,
    'mesh_ready': 85,
    'exporting_glb': 90,
    'building_preview': 96,
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
    'shape_generation': 'Running Hunyuan3D diffusion inference',
    'extracting_mesh': 'Extracting mesh geometry',
    'mesh_ready': 'Mesh geometry is ready',
    'exporting_glb': 'Exporting binary GLB',
    'building_preview': 'Building interactive 3D preview',
    'completed': 'Generation completed successfully',
    'failed': 'Generation failed',
}


def update_generation_stage(save_folder, stage, message=None, **updates):
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
    events.append({
        'stage': stage,
        'message': message or GENERATION_STAGE_MESSAGES.get(stage, stage),
        'at': updated_at,
        'progress': GENERATION_STAGE_PROGRESS.get(stage, 0),
    })
    updates.update({
        'stage': stage,
        'progress': GENERATION_STAGE_PROGRESS.get(stage, 0),
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




def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
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


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', related_path)
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR).replace(os.sep, '/')
    iframe_tag = f'<iframe src="/static/{rel_path}" \
height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html file {output_html_path}, \
{os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

@spaces.GPU(duration=60)
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
    if caption: print('prompt is', caption)
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
            'views_used': list(image.keys()) if MV_MODE else ['image'],
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
        except Exception as e:
            raise gr.Error(f"Text to 3D is disable. \
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
        start_time = time.time()
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = get_background_remover()(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
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
    if tracking_enabled:
        update_generation_stage(
            save_folder,
            'shape_generation',
            message=f'Running {steps} diffusion steps at octree {octree_resolution}',
        )
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
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

@spaces.GPU(duration=60)
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

    text_path = os.path.join(save_folder, f'textured_mesh.obj')
    path_textured = tex_pipeline(mesh_path=path, image_path=image, output_mesh_path=text_path, save_glb=False)
        
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time

    tmp_time = time.time()
    # Convert textured OBJ to GLB using obj2gltf with PBR support
    glb_path_textured = os.path.join(save_folder, 'textured_mesh.glb')
    conversion_success = quick_convert_with_obj2gltf(path_textured, glb_path_textured)

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

@spaces.GPU(duration=60)
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
    request: gr.Request = None,
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
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'

    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">

    {title}
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    """
    custom_css = """
    .gradio-container {
        max-width: 1880px !important;
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
        scrollbar-color: #343a46 transparent;
        scrollbar-width: thin;
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

    body.rtx3090-modal-open {
        overflow: hidden;
    }

    #rtx3090-modal {
        align-items: stretch;
        background: rgba(3, 5, 9, 0.72);
        backdrop-filter: blur(5px);
        display: none !important;
        flex-direction: row !important;
        inset: 0;
        justify-content: flex-end;
        margin: 0 !important;
        max-width: none !important;
        padding: 0 16px 0 0 !important;
        position: fixed !important;
        z-index: 1001;
    }

    #rtx3090-modal.rtx-open {
        display: flex !important;
    }

    #rtx3090-modal .rtx3090-modal-panel {
        background: var(--body-background-fill);
        border: 1px solid var(--block-border-color);
        border-radius: 12px;
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.46);
        flex: 0 0 min(1020px, calc(100vw - 16px));
        gap: 14px;
        height: 100vh;
        max-width: 1020px;
        overflow-x: hidden;
        overflow-y: auto;
        padding: 0 22px 24px;
        width: min(1020px, calc(100vw - 16px));
    }

    #rtx3090-modal .rtx3090-modal-panel > *,
    #rtx3090-modal .html-container {
        min-width: 0;
        max-width: 100%;
    }

    #rtx3090-modal .html-container {
        overflow: visible !important;
    }

    .rtx3090-modal-header {
        align-items: center;
        background: var(--body-background-fill);
        border-bottom: 1px solid var(--border-color-primary);
        display: flex;
        gap: 16px;
        justify-content: space-between;
        margin: 0 -22px;
        min-height: 62px;
        padding: 12px 18px 12px 22px;
        position: sticky;
        top: 0;
        z-index: 2;
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
        color: #ff922b;
        display: inline-flex;
        flex: 0 0 auto;
        font-size: 18px;
        justify-content: center;
    }

    .rtx3090-modal-header h2 {
        color: var(--body-text-color);
        font-size: 18px;
        line-height: 1.2;
        margin: 0;
        white-space: nowrap;
    }

    .rtx3090-header-scope {
        color: var(--primary-500);
        font-size: 16px;
        white-space: nowrap;
    }

    .rtx3090-header-actions {
        flex: 0 0 auto;
        gap: 10px;
    }

    .rtx3090-verified {
        align-items: center;
        background: var(--button-secondary-background-fill);
        border: 1px solid var(--button-secondary-border-color);
        border-radius: 5px;
        display: inline-flex;
        font-size: 12px;
        gap: 5px;
        padding: 5px 7px;
        white-space: nowrap;
    }

    .rtx3090-verified-dot {
        background: #ff922b;
        border-radius: 50%;
        height: 8px;
        width: 8px;
    }

    .rtx3090-preset-count {
        font-size: 16px;
        white-space: nowrap;
    }

    .rtx3090-preset-count b {
        color: var(--primary-500);
        font-weight: 500;
    }

    #rtx3090-modal-close {
        align-items: center;
        background: transparent;
        border: 0;
        color: var(--body-text-color);
        cursor: pointer;
        display: flex;
        flex: 0 0 30px;
        font-size: 26px;
        font-weight: 700;
        height: 30px;
        justify-content: center;
        line-height: 1;
        padding: 0;
        width: 30px;
    }

    #rtx3090-modal-close:hover {
        color: var(--primary-500);
    }

    .rtx3090-api-intro {
        padding-top: 4px;
    }

    .rtx3090-api-intro p {
        color: var(--body-text-color);
        font-size: 13px;
        line-height: 1.5;
        margin: 0 0 13px;
    }

    .rtx3090-context-tabs {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
    }

    .rtx3090-context-tabs span {
        align-items: center;
        border: 1px solid var(--block-border-color);
        border-radius: 5px;
        color: var(--body-text-color-subdued);
        display: inline-flex;
        font-size: 11px;
        gap: 5px;
        padding: 4px 7px;
    }

    .rtx3090-context-tabs span.active {
        border-color: var(--body-text-color);
        color: var(--body-text-color);
    }

    .rtx3090-section-heading {
        align-items: baseline;
        color: var(--body-text-color);
        display: flex;
        font-size: 13px;
        gap: 5px;
        line-height: 1.5;
        margin: 1px 0 -4px;
    }

    .rtx3090-section-heading span {
        color: var(--body-text-color-subdued);
        font-size: 12px;
    }

    .rtx3090-machine-strip {
        align-items: center;
        background: var(--block-background-fill);
        border: 1px solid var(--block-border-color);
        border-radius: 8px;
        display: flex;
        gap: 11px;
        padding: 12px 14px;
    }

    .rtx3090-machine-badge {
        background: #4263eb;
        border-radius: 8px;
        color: white;
        flex: 0 0 auto;
        font-size: 12px;
        font-weight: 800;
        padding: 8px 10px;
    }

    .rtx3090-machine-strip strong,
    .rtx3090-machine-strip span {
        display: block;
    }

    .rtx3090-machine-strip strong {
        font-size: 13px;
        margin-bottom: 2px;
    }

    .rtx3090-machine-strip span {
        color: var(--body-text-color-subdued);
        font-size: 11px;
        line-height: 1.45;
    }

    .rtx3090-profile-grid {
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .rtx3090-profile-card {
        background: var(--block-background-fill);
        border: 1px solid var(--block-border-color);
        border-radius: 11px;
        padding: 14px;
    }

    .rtx3090-profile-card.quality {
        border-color: rgba(66, 99, 235, 0.72);
        box-shadow: inset 3px 0 0 #4263eb;
    }

    .rtx3090-profile-card.safe {
        box-shadow: inset 3px 0 0 #2f9e44;
    }

    .rtx3090-profile-card h3 {
        font-size: 13px;
        margin: 0 0 4px;
    }

    .rtx3090-profile-card p {
        color: var(--body-text-color-subdued);
        font-size: 11px;
        line-height: 1.45;
        margin: 0 0 11px;
    }

    .rtx3090-profile-values {
        display: grid;
        gap: 6px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }

    .rtx3090-profile-values span {
        background: var(--background-fill-secondary);
        border-radius: 7px;
        color: var(--body-text-color-subdued);
        font-size: 10px;
        padding: 7px 5px;
        text-align: center;
    }

    .rtx3090-profile-values b {
        color: var(--body-text-color);
        display: block;
        font-size: 12px;
        margin-bottom: 2px;
    }

    #rtx3090-modal .rtx-preset-actions {
        gap: 10px;
    }

    #rtx3090-modal .rtx-preset-actions button {
        min-height: 42px;
    }

    .rtx-preset-status {
        background: var(--block-background-fill);
        border: 1px solid var(--block-border-color);
        border-radius: 10px;
        margin-top: 8px;
        padding: 10px 12px;
    }

    .rtx-preset-status.quality {
        border-color: rgba(66, 99, 235, 0.72);
        box-shadow: inset 3px 0 0 #4263eb;
    }

    .rtx-preset-status.safe {
        box-shadow: inset 3px 0 0 #2f9e44;
    }

    .rtx-preset-status-title {
        align-items: center;
        display: flex;
        font-size: 12px;
        font-weight: 750;
        gap: 7px;
    }

    .rtx-preset-dot {
        background: #2f9e44;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(47, 158, 68, 0.15);
        height: 7px;
        width: 7px;
    }

    .rtx-preset-values {
        display: grid;
        gap: 6px;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin: 9px 0 7px;
    }

    .rtx-preset-values span {
        background: var(--background-fill-secondary);
        border-radius: 7px;
        color: var(--body-text-color-subdued);
        font-size: 10px;
        padding: 6px 7px;
        text-align: center;
    }

    .rtx-preset-values b {
        color: var(--body-text-color);
        display: block;
        font-size: 12px;
        margin-bottom: 1px;
    }

    .rtx-preset-detail {
        color: var(--body-text-color-subdued);
        font-size: 10px;
        line-height: 1.45;
    }

    .rtx3090-modal-note {
        background: var(--block-background-fill);
        border: 1px solid var(--block-border-color);
        border-radius: 10px;
        color: var(--body-text-color-subdued);
        font-size: 11px;
        line-height: 1.5;
        padding: 11px 13px;
    }

    @media (max-width: 720px) {
        #rtx3090-modal {
            padding: 0 !important;
        }

        #rtx3090-modal .rtx3090-modal-panel {
            border-radius: 0;
            flex-basis: 100vw;
            height: 100vh;
            max-width: 100vw;
            padding-left: 14px;
            padding-right: 14px;
            width: 100vw;
        }

        .rtx3090-modal-header {
            margin-left: -14px;
            margin-right: -14px;
            padding: 16px 14px;
        }

        .rtx3090-header-scope,
        .rtx3090-verified,
        .rtx3090-preset-count {
            display: none;
        }

        .rtx3090-modal-header h2 {
            font-size: 15px;
            white-space: normal;
        }

        .rtx3090-profile-grid {
            grid-template-columns: 1fr;
        }

        .rtx3090-profile-values,
        .rtx-preset-values {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    """

    custom_js = r"""
    () => {
        const modalId = "rtx3090-modal";
        const footerButtonId = "rtx3090-footer-trigger";
        const modal = () => document.getElementById(modalId);
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
            while (log.children.length > 80) log.firstElementChild?.remove();
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

        const stopGenerationConsolePolling = () => {
            if (generationConsoleTimer !== null) {
                window.clearInterval(generationConsoleTimer);
                generationConsoleTimer = null;
            }
        };

        const renderGenerationManifest = (manifest) => {
            if (!manifest || manifest.generation_uid !== generationConsoleUid) return;

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
                    appendGenerationConsoleLine(
                        "OUTPUT",
                        "hy3dshape/output_folder/webui/" + generationConsoleUid + "/white_mesh.glb",
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
                "Target: hy3dshape/output_folder/webui/" + uid,
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

        const setModalOpen = (isOpen) => {
            const element = modal();
            if (!element) return;
            element.classList.toggle("rtx-open", isOpen);
            element.setAttribute("aria-hidden", String(!isOpen));
            document.body.classList.toggle("rtx3090-modal-open", isOpen);
            if (isOpen) {
                document.getElementById("rtx3090-modal-close")?.focus();
            }
        };

        const syncFromUrl = () => {
            const url = currentAppUrl();
            setModalOpen(url.searchParams.get("view") === "rtx3090");
        };

        const openModal = () => {
            const url = currentAppUrl();
            url.searchParams.set("view", "rtx3090");
            window.history.pushState({}, "", url);
            setModalOpen(true);
        };

        const closeModal = () => {
            const url = currentAppUrl();
            if (url.searchParams.get("view") === "rtx3090") {
                url.searchParams.delete("view");
                window.history.pushState({}, "", url);
            }
            setModalOpen(false);
        };

        const installFooterItem = () => {
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
            trigger.innerHTML = '<span class="rtx3090-footer-icon">⚡</span><span>RTX 3090 - Cấu hình đề xuất</span>';
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
            wireModal();
            installTabRouting();
            installGenerationRouting();
            syncGenerationConsoleFromUrl();
        });
        observer.observe(document.body, {childList: true, subtree: true});

        installFooterItem();
        wireModal();
        installTabRouting();
        installGenerationRouting();
        syncFromUrl();
        syncGenerationConsoleFromUrl();

        window.addEventListener("popstate", () => {
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
    """ if MV_MODE and args.device == 'cuda' else "() => {}"

    with gr.Blocks(
        theme=gr.themes.Base(),
        title=title,
        analytics_enabled=False,
        css=custom_css,
        js=custom_js,
    ) as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=3):
                input_mode = gr.Textbox(value='single', visible=False, label='Input mode')
                with gr.Tabs(selected='tab_single_prompt', elem_id='prompt-mode-tabs') as tabs_prompt:
                    with gr.Tab('1 ẢNH · Single View', id='tab_single_prompt') as tab_ip:
                        gr.HTML("""
                        <div class="input-mode-guide">
                            <div class="input-mode-number">1</div>
                            <div class="input-mode-copy">
                                <strong>Một ảnh chính diện</strong>
                                <span>Nhanh và đơn giản. Dùng ảnh Front rõ nét, nền trong suốt.</span>
                            </div>
                        </div>
                        """)
                        image = gr.Image(
                            label='Ảnh chính diện · Front',
                            type='pil',
                            image_mode='RGBA',
                            height=300,
                            elem_classes='single-image',
                        )
                        caption = gr.State(None)
#                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
#                        caption = gr.Textbox(label='Text Prompt',
#                                             placeholder='HunyuanDiT will be used to generate image.',
#                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('4 ẢNH · Multi View', id='tab_mv_prompt', visible=MV_MODE) as tab_mv:
                        gr.HTML("""
                        <div class="input-mode-guide">
                            <div class="input-mode-number">4</div>
                            <div class="input-mode-copy">
                                <strong>Bốn hướng đồng bộ</strong>
                                <span>Đưa đủ Front, Back, Left và Right để hình học nhất quán hơn.</span>
                            </div>
                        </div>
                        """)
                        with gr.Row():
                            mv_image_front = gr.Image(label='1 · Mặt trước · Front', type='pil', image_mode='RGBA', height=150,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='2 · Mặt sau · Back', type='pil', image_mode='RGBA', height=150,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='3 · Bên trái · Left', type='pil', image_mode='RGBA', height=150,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='4 · Bên phải · Right', type='pil', image_mode='RGBA', height=150,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row():
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

                with gr.Group():
                    file_out = gr.File(label="Generated mesh (direct download)", visible=True,
                                       interactive=False)
                    file_out2 = gr.File(label="File", visible=False)

                selected_options_tab = (
                    'tab_options' if TURBO_MODE
                    else 'tab_export' if HAS_PYMESHLAB
                    else 'tab_advanced_options'
                )
                with gr.Tabs(selected=selected_options_tab):
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
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(
                                value=not MV_MODE,
                                label='Remove Background',
                                visible=HAS_REMBG,
                                min_width=100)
                            randomize_seed = gr.Checkbox(
                                label="Randomize seed", 
                                value=True, 
                                min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, 
                                                          minimum=16, 
                                                          value=256, 
                                                          label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
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

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3, visible=MV_MODE, elem_id='generation-console-panel'):
                gr.HTML("""
                <section id="generation-console" class="generation-console" data-state="idle">
                    <header class="generation-console-windowbar">
                        <span class="generation-console-dots" aria-hidden="true"><i></i><i></i><i></i></span>
                        <span class="generation-console-title">
                            <strong>Generation Console</strong>
                            <span>REAL-TIME INFERENCE PIPELINE</span>
                        </span>
                        <span id="generation-console-status" class="generation-console-status">IDLE</span>
                    </header>
                    <div class="generation-console-jobbar">
                        <span aria-hidden="true">›_</span>
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
                    <div class="generation-console-progress-wrap">
                        <div class="generation-console-progress-meta">
                            <span id="generation-console-stage">Waiting for a generation request</span>
                            <span id="generation-console-percent">0%</span>
                        </div>
                        <div class="generation-console-progress-track">
                            <div id="generation-console-progress" class="generation-console-progress-bar"></div>
                        </div>
                    </div>
                    <div class="generation-console-footer">
                        <span><strong>Hunyuan3D-2mv</strong> · CUDA FP16</span>
                        <span id="generation-console-clock">LOCAL STORAGE READY</span>
                    </div>
                </section>
                """)

            with gr.Column(scale=2, visible=not MV_MODE):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', 
                                id='tab_img_gallery', 
                                visible=not MV_MODE) as tab_gi:
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
                        <span class="rtx3090-header-icon">⚡</span>
                        <h2 id="rtx3090-modal-title">RTX 3090 · Cấu hình đề xuất</h2>
                        <span class="rtx3090-header-scope">1 ảnh &amp; 4 ảnh</span>
                    </div>
                    <div class="rtx3090-header-actions">
                        <span class="rtx3090-verified">
                            <i class="rtx3090-verified-dot"></i>
                            Đã kiểm tra
                        </span>
                        <span class="rtx3090-preset-count"><b>2</b> preset</span>
                        <button id="rtx3090-modal-close" type="button" aria-label="Đóng">×</button>
                    </div>
                </div>
                """)
                gr.HTML("""
                <div class="rtx3090-api-intro">
                    <p>
                        Chọn một trong các cấu hình dưới đây để tối ưu chất lượng mesh trên
                        máy hiện tại. Các giá trị sẽ được cập nhật trực tiếp vào
                        <strong>Advanced Options</strong>.
                    </p>
                    <div class="rtx3090-context-tabs">
                        <span class="active">⚡ RTX 3090</span>
                        <span>▣ 24 GB VRAM</span>
                        <span>1 ảnh</span>
                        <span>4 ảnh</span>
                        <span>FP16</span>
                    </div>
                </div>
                """)
                gr.HTML("""
                <div class="rtx3090-section-heading">
                    <b>1. Xác nhận cấu hình máy.</b>
                    <span>Hai preset này dành riêng cho RTX 3090 24 GB đang chạy WebUI.</span>
                </div>
                """)
                gr.HTML("""
                <div class="rtx3090-machine-strip">
                    <div class="rtx3090-machine-badge">24 GB</div>
                    <div>
                        <strong>NVIDIA GeForce RTX 3090 · CUDA · FP16</strong>
                        <span>
                            Hai mức dưới đây đã được kiểm tra end-to-end trên chính máy này,
                            không gặp lỗi thiếu VRAM với cả đầu vào 1 ảnh và 4 ảnh.
                        </span>
                    </div>
                </div>
                """)
                gr.HTML("""
                <div class="rtx3090-section-heading">
                    <b>2. Chọn mức phù hợp rồi bấm Áp dụng.</b>
                    <span>Thông số bên dưới dùng chung cho cả chế độ 1 ảnh và 4 ảnh.</span>
                </div>
                """)
                gr.HTML("""
                <div class="rtx3090-profile-grid">
                    <article class="rtx3090-profile-card safe">
                        <h3>256 · Mặc định an toàn</h3>
                        <p>Dùng cho lần chạy đầu hoặc khi cần ưu tiên ổn định và tốc độ.</p>
                        <div class="rtx3090-profile-values">
                            <span><b>30</b>Steps</span>
                            <span><b>5.0</b>Guidance</span>
                            <span><b>256</b>Octree</span>
                            <span><b>8000</b>Chunks</span>
                        </div>
                    </article>
                    <article class="rtx3090-profile-card quality">
                        <h3>384 · Chất lượng cao</h3>
                        <p>Dùng khi ảnh đầu vào đã đúng và cần mesh dày, mịn hơn.</p>
                        <div class="rtx3090-profile-values">
                            <span><b>30</b>Steps</span>
                            <span><b>5.0</b>Guidance</span>
                            <span><b>384</b>Octree</span>
                            <span><b>8000</b>Chunks</span>
                        </div>
                    </article>
                </div>
                """)
                with gr.Row(elem_classes='rtx-preset-actions'):
                    rtx_safe_preset = gr.Button(
                        value='Áp dụng · 256 an toàn',
                        min_width=160,
                    )
                    rtx_quality_preset = gr.Button(
                        value='Áp dụng · 384 chất lượng cao',
                        variant='primary',
                        min_width=180,
                    )
                rtx_preset_status = gr.HTML(get_rtx3090_preset('safe')[-1])
                gr.HTML("""
                <div class="rtx3090-modal-note">
                    <strong>Lưu ý:</strong> Seed không làm tăng VRAM. Bật Randomize để thử biến
                    thể, tắt để tái tạo đúng kết quả. Chunks 8000 cân bằng tốc độ/bộ nhớ và
                    không làm mesh đẹp hơn khi tăng quá cao. Không đặt Octree 512 làm mặc định
                    trên RTX 3090 24 GB.
                </div>
                """)

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

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
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
        from hy3dgen.text2image import HunyuanDiTPipeline

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
        variant=args.variant or None,
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
