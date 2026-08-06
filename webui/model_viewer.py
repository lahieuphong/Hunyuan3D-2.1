"""Shared generation-asset resolution and model-viewer rendering helpers.

This module deliberately has no Gradio or FastAPI dependency.  History, saved
generation restoration, and the standalone viewer can therefore share the
same filename validation and variant-selection rules.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import struct
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .i18n import (
    DEFAULT_UI_LOCALE,
    ENABLED_UI_LOCALES,
    SUPPORTED_UI_LOCALES,
    normalize_ui_locale,
)


VARIANT_ORDER = ("original", "white", "wireframe")

_ASSET_DIRECTORY = Path(__file__).resolve().parent.parent / "assets"
_VIEWER_TEMPLATE_PATH = _ASSET_DIRECTORY / "modelviewer-template.html"
_VIEWER_CSS_PATH = _ASSET_DIRECTORY / "modelviewer.css"
_VIEWER_JS_PATH = _ASSET_DIRECTORY / "modelviewer.js"
_VIEWER_CSS_SENTINEL = "/* #viewer-css# */"
_VIEWER_JS_SENTINEL = "// #viewer-js#"
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_GLB_JSON_BYTES = 16 * 1024 * 1024
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_LINES_MODE = 1
_RENDER_MODE_DEFAULTS = {
    "original": "embedded",
    "white": "clay",
    "wireframe": "lines",
}
_RENDER_MODE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_MODE_ICONS = {
    "original": "palette",
    "white": "box",
    "wireframe": "wireframe",
}
VIEWER_LOCALES = SUPPORTED_UI_LOCALES
_VIEWER_MESSAGES = {
    "en": {
        "title": "Hunyuan3D-2mv · 3D Model Viewer",
        "modelPreview": "Generated 3D model preview",
        "preview": "Preview",
        "clickDrag": "Click and drag to rotate",
        "scrollZoom": "Scroll to zoom",
        "viewerControls": "Viewer controls",
        "fullscreen": "Fullscreen",
        "resetCamera": "Reset camera",
        "toggleAutoRotate": "Toggle auto rotate",
        "toggleFloorGrid": "Toggle floor grid",
        "loadingModel": "Loading model…",
        "displayModes": "Model display modes",
        "modeOriginal": "Original",
        "modeWhite": "White",
        "modeWireframe": "Wireframe",
        "modeFailedTitle": "{mode} model could not be loaded",
        "modeUnavailableTitle": "{mode} model is unavailable",
        "showModeTitle": "Show {mode} model",
        "loadingMode": "Loading {mode}…",
        "modeReady": "{mode} model ready",
        "requestedMode": "Requested",
        "fallbackMode": "{mode} is unavailable. Showing {fallback} instead.",
        "modeFailed": "{mode} model could not be loaded.",
        "fullscreenUnavailable": "Fullscreen is unavailable in this browser.",
        "configurationUnreadable": "Viewer configuration could not be read.",
        "generationNotFound": "Generation not found",
        "meshNotFound": "Generated mesh not found",
        "variantNotFound": "Model variant not found",
    },
    "zh-CN": {
        "title": "Hunyuan3D-2mv · 3D 模型查看器",
        "modelPreview": "生成的 3D 模型预览",
        "preview": "预览",
        "clickDrag": "点击并拖动以旋转",
        "scrollZoom": "滚动以缩放",
        "viewerControls": "查看器控件",
        "fullscreen": "全屏",
        "resetCamera": "重置相机",
        "toggleAutoRotate": "切换自动旋转",
        "toggleFloorGrid": "切换地面网格",
        "loadingModel": "正在加载模型…",
        "displayModes": "模型显示模式",
        "modeOriginal": "原始模型",
        "modeWhite": "白模",
        "modeWireframe": "线框",
        "modeFailedTitle": "无法加载{mode}",
        "modeUnavailableTitle": "{mode}不可用",
        "showModeTitle": "显示{mode}",
        "loadingMode": "正在加载{mode}…",
        "modeReady": "{mode}已就绪",
        "requestedMode": "请求的模型",
        "fallbackMode": "{mode}不可用，改为显示{fallback}。",
        "modeFailed": "无法加载{mode}。",
        "fullscreenUnavailable": "此浏览器不支持全屏。",
        "configurationUnreadable": "无法读取查看器配置。",
        "generationNotFound": "未找到生成记录",
        "meshNotFound": "未找到生成的网格",
        "variantNotFound": "未找到模型变体",
    },
}
_MODE_MESSAGE_KEYS = {
    "original": "modeOriginal",
    "white": "modeWhite",
    "wireframe": "modeWireframe",
}
_WIREFRAME_EXPORT_LOCK = threading.Lock()
# Bump this whenever the generated line topology or styling changes so cached
# previews from older releases are rebuilt automatically.
_WIREFRAME_GENERATOR_VERSION = 2
_WIREFRAME_MAX_FACES = 8_000
_WIREFRAME_GRID_RESOLUTIONS = (
    256,
    224,
    192,
    176,
    160,
    144,
    128,
    112,
    96,
    80,
    72,
    64,
    56,
    48,
    40,
    32,
)


@dataclass(frozen=True)
class ViewerVariant:
    """One validated GLB that can be selected in the model viewer."""

    mode: str
    filename: str
    path: Path
    render_mode: str

    @property
    def cache_key(self) -> str:
        """Return a stable browser cache key for the current file contents."""

        stat = self.path.stat()
        return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


@dataclass(frozen=True)
class ViewerAssets:
    """The available variants for one generation folder."""

    variants: dict[str, ViewerVariant]
    default_mode: str

    @property
    def primary(self) -> ViewerVariant:
        return self.variants[self.default_mode]

    @property
    def cache_key(self) -> str:
        """Return one compact key that changes when any available GLB changes."""

        state = "|".join(
            f"{mode}:{variant.cache_key}"
            for mode, variant in self.variants.items()
        )
        return hashlib.sha256(state.encode("utf-8")).hexdigest()[:20]


def stored_generation_file(
    folder: str | os.PathLike[str],
    filename: object,
    suffix: str | None = None,
) -> Path | None:
    """Resolve a direct, non-symlink file inside a generation folder.

    Both slash styles are rejected explicitly so the check remains safe when
    manifests created on Windows are read on POSIX, or vice versa.
    """

    if not isinstance(filename, str):
        return None
    if filename != filename.strip():
        return None
    if (
        not filename
        or filename in {".", ".."}
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        return None

    if suffix is not None:
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        if Path(filename).suffix.lower() != normalized_suffix.lower():
            return None

    raw_folder = Path(folder).absolute()
    try:
        if raw_folder.is_symlink() or not raw_folder.is_dir():
            return None
        resolved_folder = raw_folder.resolve(strict=True)
        candidate = raw_folder / filename
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    if (
        resolved_candidate.parent != resolved_folder
        or resolved_candidate.name != filename
    ):
        return None
    return resolved_candidate


def read_generation_manifest(
    folder: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read a small, regular ``generation.json`` or return an empty mapping."""

    path = stored_generation_file(folder, "generation.json", suffix=".json")
    if path is None:
        return {}
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _glb_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read only the JSON chunk of a GLB after validating its container."""

    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        file_size = candidate.stat().st_size
        with candidate.open("rb") as glb_file:
            header = glb_file.read(20)
            if len(header) != 20:
                return None
            magic, version, declared_size, chunk_size, chunk_type = struct.unpack(
                "<4sIIII",
                header,
            )
            if (
                magic != b"glTF"
                or version != 2
                or declared_size != file_size
                or chunk_type != _GLB_JSON_CHUNK
                or chunk_size <= 0
                or chunk_size > _MAX_GLB_JSON_BYTES
                or 20 + chunk_size > file_size
            ):
                return None
            raw_document = glb_file.read(chunk_size)
        document = json.loads(raw_document.decode("utf-8").rstrip("\x00 \t\r\n"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        struct.error,
        ValueError,
    ):
        return None
    return document if isinstance(document, dict) else None


def is_wireframe_glb(path: str | os.PathLike[str]) -> bool:
    """Return whether a GLB contains one or more line-only mesh primitives."""

    document = _glb_json(path)
    if not document:
        return False

    primitive_modes: list[object] = []
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        return False
    for mesh in meshes:
        if not isinstance(mesh, dict):
            return False
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            return False
        for primitive in primitives:
            if not isinstance(primitive, dict):
                return False
            primitive_modes.append(primitive.get("mode", 4))
    return bool(primitive_modes) and all(
        mode == _GLB_LINES_MODE for mode in primitive_modes
    )


def _wireframe_cache_is_current(path: Path) -> bool:
    document = _glb_json(path)
    if not document:
        return False
    scene_index = document.get("scene", 0)
    scenes = document.get("scenes")
    if (
        not isinstance(scene_index, int)
        or not isinstance(scenes, list)
        or scene_index < 0
        or scene_index >= len(scenes)
        or not isinstance(scenes[scene_index], dict)
    ):
        return False
    extras = scenes[scene_index].get("extras")
    return (
        isinstance(extras, dict)
        and extras.get("wireframe_generator_version")
        == _WIREFRAME_GENERATOR_VERSION
    )


def _glb_has_texture(path: Path) -> bool:
    """Best-effort legacy classification for a custom primary GLB."""

    document = _glb_json(path)
    if not document:
        return False
    images = document.get("images")
    textures = document.get("textures")
    return (
        isinstance(images, list)
        and bool(images)
        and isinstance(textures, list)
        and bool(textures)
    )


def _render_mode(mode: str, value: object = None) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower()
        if _RENDER_MODE_PATTERN.fullmatch(candidate):
            return candidate
    return _RENDER_MODE_DEFAULTS[mode]


def _variant_from_filename(
    folder: Path,
    mode: str,
    filename: object,
    render_mode: object = None,
) -> ViewerVariant | None:
    path = stored_generation_file(folder, filename, suffix=".glb")
    if path is None:
        return None
    if mode == "wireframe" and not is_wireframe_glb(path):
        return None
    return ViewerVariant(
        mode=mode,
        filename=path.name,
        path=path,
        render_mode=_render_mode(mode, render_mode),
    )


def _explicit_variant(
    folder: Path,
    mode: str,
    raw_variant: object,
) -> ViewerVariant | None:
    if isinstance(raw_variant, str):
        return _variant_from_filename(folder, mode, raw_variant)
    if not isinstance(raw_variant, Mapping):
        return None
    filename = (
        raw_variant.get("file")
        or raw_variant.get("filename")
        or raw_variant.get("mesh")
    )
    return _variant_from_filename(
        folder,
        mode,
        filename,
        raw_variant.get("render_mode"),
    )


def _first_variant(
    folder: Path,
    mode: str,
    filenames: tuple[object, ...],
) -> ViewerVariant | None:
    for filename in filenames:
        variant = _variant_from_filename(folder, mode, filename)
        if variant is not None:
            return variant
    return None


def resolve_generation_assets(
    folder: str | os.PathLike[str],
    manifest: Mapping[str, Any] | None = None,
    ensure_wireframe: bool = False,
) -> ViewerAssets | None:
    """Resolve explicit and legacy model variants for a generation.

    Explicit, valid entries win per mode.  Missing or invalid modes are filled
    only from the named legacy files below; arbitrary ``*.glb`` files are
    intentionally never scanned.
    """

    raw_folder = Path(folder).absolute()
    try:
        if raw_folder.is_symlink() or not raw_folder.is_dir():
            return None
        resolved_folder = raw_folder.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    if manifest is None:
        manifest_value: Mapping[str, Any] = read_generation_manifest(
            resolved_folder
        )
    else:
        manifest_value = manifest if isinstance(manifest, Mapping) else {}

    raw_outputs = manifest_value.get("outputs")
    outputs = raw_outputs if isinstance(raw_outputs, Mapping) else {}
    raw_variants = outputs.get("variants")
    explicit_variants = (
        raw_variants if isinstance(raw_variants, Mapping) else {}
    )

    resolved: dict[str, ViewerVariant] = {}
    for mode in VARIANT_ORDER:
        variant = _explicit_variant(
            resolved_folder,
            mode,
            explicit_variants.get(mode),
        )
        if variant is not None:
            resolved[mode] = variant

    primary_filename = outputs.get("mesh")
    primary_path = stored_generation_file(
        resolved_folder,
        primary_filename,
        suffix=".glb",
    )
    primary_is_textured = (
        primary_path is not None and _glb_has_texture(primary_path)
    )

    if "original" not in resolved:
        resolved_original = _first_variant(
            resolved_folder,
            "original",
            ("textured_mesh.glb", "textured_preview.glb"),
        )
        if resolved_original is None:
            if primary_is_textured and primary_path is not None:
                resolved_original = _variant_from_filename(
                    resolved_folder,
                    "original",
                    primary_path.name,
                )
        if resolved_original is not None:
            resolved["original"] = resolved_original

    if "white" not in resolved:
        resolved_white = _first_variant(
            resolved_folder,
            "white",
            ("white_mesh.glb",),
        )
        primary_is_original = (
            primary_path is not None
            and "original" in resolved
            and resolved["original"].path == primary_path
        )
        if (
            resolved_white is None
            and primary_path is not None
            and not primary_is_textured
            and not primary_is_original
        ):
            resolved_white = _variant_from_filename(
                resolved_folder,
                "white",
                primary_path.name,
            )
        if resolved_white is not None:
            resolved["white"] = resolved_white

    if "wireframe" not in resolved:
        resolved_wireframe = _variant_from_filename(
            resolved_folder,
            "wireframe",
            "wireframe_mesh.glb",
        )
        if resolved_wireframe is not None:
            resolved["wireframe"] = resolved_wireframe

    canonical_wireframe_path = resolved_folder / "wireframe_mesh.glb"
    existing_wireframe = resolved.get("wireframe")
    should_ensure_wireframe = (
        ensure_wireframe
        and "white" in resolved
        and (
            existing_wireframe is None
            or existing_wireframe.path == canonical_wireframe_path
        )
    )
    if should_ensure_wireframe:
        try:
            wireframe_path = export_wireframe_glb(
                resolved["white"].path,
                canonical_wireframe_path,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            wireframe_path = None
        if wireframe_path is not None:
            resolved_wireframe = _variant_from_filename(
                resolved_folder,
                "wireframe",
                wireframe_path.name,
            )
            if resolved_wireframe is not None:
                resolved["wireframe"] = resolved_wireframe
        elif existing_wireframe is not None:
            # Do not keep serving a canonical cache that failed its refresh.
            resolved.pop("wireframe", None)

    ordered = {
        mode: resolved[mode]
        for mode in VARIANT_ORDER
        if mode in resolved
    }
    if not ordered:
        return None

    requested_default = outputs.get("default_variant")
    default_mode = (
        requested_default
        if isinstance(requested_default, str) and requested_default in ordered
        else next(mode for mode in VARIANT_ORDER if mode in ordered)
    )
    return ViewerAssets(variants=ordered, default_mode=default_mode)


def _mesh_for_wireframe(mesh_or_path: object, trimesh_module: Any) -> tuple[Any, Path | None]:
    source_path: Path | None = None
    if isinstance(mesh_or_path, (str, os.PathLike)):
        source_path = Path(mesh_or_path).resolve(strict=True)
        loaded = trimesh_module.load(
            str(source_path),
            force="mesh",
            process=False,
        )
    else:
        loaded = mesh_or_path

    if isinstance(loaded, trimesh_module.Scene):
        geometries = tuple(
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh_module.Trimesh)
        )
        if not geometries:
            raise ValueError("The source scene does not contain a triangle mesh")
        loaded = trimesh_module.util.concatenate(geometries)
    if not isinstance(loaded, trimesh_module.Trimesh):
        raise TypeError("mesh_or_path must resolve to a trimesh.Trimesh")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("Cannot create a wireframe from an empty mesh")
    return loaded, source_path


def _display_wireframe_mesh(mesh: Any, trimesh_module: Any, np: Any) -> Any:
    """Cluster very dense meshes so their line grid remains readable zoomed out."""

    if len(mesh.faces) <= _WIREFRAME_MAX_FACES:
        return mesh

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    minimum = vertices.min(axis=0)
    span = float((vertices.max(axis=0) - minimum).max())
    if not np.isfinite(span) or span <= 0:
        return mesh

    selected = None
    for resolution in _WIREFRAME_GRID_RESOLUTIONS:
        cells = np.floor(
            (vertices - minimum) * (resolution - 1) / span
        ).astype(np.int64)
        cell_keys = (
            cells[:, 0]
            + resolution * (cells[:, 1] + resolution * cells[:, 2])
        )
        _, inverse = np.unique(cell_keys, return_inverse=True)
        remapped_faces = inverse[faces]
        non_degenerate = (
            (remapped_faces[:, 0] != remapped_faces[:, 1])
            & (remapped_faces[:, 1] != remapped_faces[:, 2])
            & (remapped_faces[:, 0] != remapped_faces[:, 2])
        )
        remapped_faces = remapped_faces[non_degenerate]
        if len(remapped_faces) == 0:
            continue

        _, first_indices = np.unique(
            np.sort(remapped_faces, axis=1),
            axis=0,
            return_index=True,
        )
        remapped_faces = remapped_faces[np.sort(first_indices)]
        selected = (inverse, remapped_faces)
        if len(remapped_faces) <= _WIREFRAME_MAX_FACES:
            break

    if selected is None:
        return mesh

    inverse, remapped_faces = selected
    cluster_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=cluster_count)
    clustered_vertices = np.column_stack([
        np.bincount(inverse, weights=vertices[:, axis], minlength=cluster_count)
        for axis in range(3)
    ]) / counts[:, None]

    used_vertices = np.unique(remapped_faces)
    compact_indices = np.full(cluster_count, -1, dtype=np.int64)
    compact_indices[used_vertices] = np.arange(len(used_vertices))
    return trimesh_module.Trimesh(
        vertices=clustered_vertices[used_vertices],
        faces=compact_indices[remapped_faces],
        process=False,
        validate=False,
    )


def _replace_file(source: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.01 * (attempt + 1))


def export_wireframe_glb(
    mesh_or_path: object,
    target: str | os.PathLike[str],
) -> Path:
    """Atomically export a readable topology preview as glTF ``LINES``."""

    # History refreshes and a restored viewer can resolve the same generation
    # concurrently. Serialize the cache check and build so only one request
    # performs the relatively expensive edge export.
    with _WIREFRAME_EXPORT_LOCK:
        return _export_wireframe_glb_unlocked(mesh_or_path, target)


def _export_wireframe_glb_unlocked(
    mesh_or_path: object,
    target: str | os.PathLike[str],
) -> Path:

    target_path = Path(target).absolute()
    if target_path.suffix.lower() != ".glb":
        raise ValueError("Wireframe output must use the .glb suffix")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.is_symlink():
        raise ValueError("Refusing to replace a symlink wireframe target")

    source_path: Path | None = None
    if isinstance(mesh_or_path, (str, os.PathLike)):
        source_path = Path(mesh_or_path).resolve(strict=True)
        if source_path == target_path.resolve(strict=False):
            raise ValueError("Wireframe source and target must be different files")

    if is_wireframe_glb(target_path) and _wireframe_cache_is_current(target_path):
        if source_path is not None:
            try:
                if target_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
                    return target_path.resolve(strict=True)
            except OSError:
                pass

    # Trimesh is intentionally imported only when a wireframe must be built.
    import numpy as np
    import trimesh

    mesh, loaded_source_path = _mesh_for_wireframe(mesh_or_path, trimesh)
    mesh = _display_wireframe_mesh(mesh, trimesh, np)
    source_path = source_path or loaded_source_path
    edges = mesh.edges_unique
    if len(edges) == 0:
        raise ValueError("The source mesh does not contain any edges")

    segments = mesh.vertices[edges]
    line_geometry = trimesh.load_path(segments)
    line_geometry.colors = np.tile(
        np.array([126, 139, 255, 255], dtype=np.uint8),
        (len(line_geometry.entities), 1),
    )
    wireframe_scene = trimesh.Scene(line_geometry)
    wireframe_scene.metadata[
        "wireframe_generator_version"
    ] = _WIREFRAME_GENERATOR_VERSION
    payload = trimesh.exchange.gltf.export_glb(wireframe_scene)
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("Trimesh did not return a binary GLB payload")

    temporary_path = target_path.with_name(
        f".{target_path.stem}.{uuid.uuid4().hex}.tmp.glb"
    )
    try:
        temporary_path.write_bytes(bytes(payload))
        if not is_wireframe_glb(temporary_path):
            raise RuntimeError("Exported wireframe is not a line-only GLB")
        _replace_file(temporary_path, target_path)
    finally:
        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass
    return target_path.resolve(strict=True)


def _safe_json(value: object) -> str:
    """Serialize JSON for an inline non-executable ``<script>`` element."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def normalize_viewer_locale(locale: object) -> str:
    """Return an enabled viewer locale, defaulting to Simplified Chinese."""

    return normalize_ui_locale(locale if isinstance(locale, str) else None)


def _viewer_message(locale: str, key: str) -> str:
    return _VIEWER_MESSAGES[locale][key]


def viewer_message(key: str, locale: object = None, **parameters: object) -> str:
    """Translate one model-viewer message for route and renderer callers."""

    locale_value = normalize_viewer_locale(locale)
    template = _viewer_message(locale_value, key)
    return template.format_map(parameters) if parameters else template


def _mode_buttons(
    available_modes: set[str],
    default_mode: str | None,
    locale: str,
) -> str:
    buttons: list[str] = []
    for mode in VARIANT_ORDER:
        available = mode in available_modes
        active = available and mode == default_mode
        unavailable_attributes = (
            ""
            if available
            else ' disabled aria-disabled="true"'
        )
        active_class = " is-active" if active else ""
        buttons.append(
            f'<button class="viewer-mode{active_class}" type="button" '
            f'data-view-mode="{mode}" data-icon="{_MODE_ICONS[mode]}" '
            f'aria-pressed="{str(active).lower()}"'
            f"{unavailable_attributes}>"
            f'<span data-viewer-mode-label="{mode}">'
            f'{html.escape(_viewer_message(locale, _MODE_MESSAGE_KEYS[mode]))}'
            f"</span></button>"
        )
    return "\n".join(buttons)


def _positive_dimension(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        dimension = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return dimension if dimension > 0 else default


def render_model_viewer_document(
    variant_sources: Mapping[str, str | None],
    default_mode: str | None,
    height: int,
    width: int,
    locale: str | None = None,
) -> str:
    """Render the standalone viewer with safe, server-resolved variant URLs."""

    locale_value = normalize_viewer_locale(locale)
    messages = _VIEWER_MESSAGES[locale_value]
    sources: dict[str, str] = {}
    for mode in VARIANT_ORDER:
        source = variant_sources.get(mode)
        if isinstance(source, str) and source.strip():
            # Preserve the exact server-generated URL. Trimming here would
            # silently alter valid percent-decoded Unicode such as U+2028.
            sources[mode] = source

    if default_mode not in sources:
        default_mode = next(
            (mode for mode in VARIANT_ORDER if mode in sources),
            None,
        )

    template = _VIEWER_TEMPLATE_PATH.read_text(encoding="utf-8")
    css = (
        _VIEWER_CSS_PATH.read_text(encoding="utf-8")
        if _VIEWER_CSS_PATH.is_file()
        else ""
    )
    javascript = (
        _VIEWER_JS_PATH.read_text(encoding="utf-8")
        if _VIEWER_JS_PATH.is_file()
        else ""
    )
    viewer_height = _positive_dimension(height, 650)
    viewer_width = _positive_dimension(width, 790)
    initial_source = sources.get(default_mode or "", "")
    config = {
        "defaultMode": default_mode,
        "defaultLocale": DEFAULT_UI_LOCALE,
        "enabledLocales": ENABLED_UI_LOCALES,
        "locale": locale_value,
        "messages": _VIEWER_MESSAGES,
        "variants": {
            mode: {"src": source}
            for mode, source in sources.items()
        },
    }
    buttons = _mode_buttons(set(sources), default_mode, locale_value)

    css_placeholder = (
        _VIEWER_CSS_SENTINEL
        if _VIEWER_CSS_SENTINEL in template
        else "#viewer-css#"
    )
    js_placeholder = (
        _VIEWER_JS_SENTINEL
        if _VIEWER_JS_SENTINEL in template
        else "#viewer-js#"
    )
    had_css_placeholder = css_placeholder in template
    had_js_placeholder = js_placeholder in template
    had_config_placeholder = "#viewer-config#" in template
    had_buttons_placeholder = "#mode-buttons#" in template

    document = (
        template
        .replace(css_placeholder, css)
        .replace(js_placeholder, javascript)
        .replace("#viewer-config#", _safe_json(config))
        .replace("#mode-buttons#", buttons)
        .replace("#document-lang#", locale_value)
        .replace("#viewer-title#", html.escape(messages["title"], quote=True))
        .replace("#model-preview#", html.escape(messages["modelPreview"], quote=True))
        .replace("#preview#", html.escape(messages["preview"], quote=True))
        .replace("#click-drag#", html.escape(messages["clickDrag"], quote=True))
        .replace("#scroll-zoom#", html.escape(messages["scrollZoom"], quote=True))
        .replace("#viewer-controls#", html.escape(messages["viewerControls"], quote=True))
        .replace("#fullscreen#", html.escape(messages["fullscreen"], quote=True))
        .replace("#reset-camera#", html.escape(messages["resetCamera"], quote=True))
        .replace("#toggle-auto-rotate#", html.escape(messages["toggleAutoRotate"], quote=True))
        .replace("#toggle-floor-grid#", html.escape(messages["toggleFloorGrid"], quote=True))
        .replace("#loading-model#", html.escape(messages["loadingModel"], quote=True))
        .replace("#display-modes#", html.escape(messages["displayModes"], quote=True))
        .replace("#src#", html.escape(initial_source, quote=True))
        .replace("#width#", str(viewer_width))
        .replace("var(--viewer-height, 650px)", f"{viewer_height}px")
    )

    # These fallbacks keep the renderer useful while an older monolithic
    # template is present during upgrades.
    if not had_buttons_placeholder:
        mode_strip = (
            '<div class="viewer-mode-strip" role="group" '
            f'aria-label="{html.escape(messages["displayModes"], quote=True)}">'
            f"{buttons}</div>"
        )
        document, replacements = re.subn(
            r'<div class="camera-strip"[^>]*>.*?</div>',
            mode_strip,
            document,
            count=1,
            flags=re.DOTALL,
        )
        if replacements == 0:
            document = document.replace("</main>", f"{mode_strip}</main>", 1)
    if css and not had_css_placeholder:
        document = document.replace(
            "</head>",
            f"<style>{css}</style></head>",
            1,
        )
    if not had_config_placeholder:
        config_element = (
            '<script type="application/json" id="viewer-config">'
            f"{_safe_json(config)}</script>"
        )
        document = document.replace("</body>", f"{config_element}</body>", 1)
    if javascript and not had_js_placeholder:
        document = document.replace(
            "</body>",
            f"<script>{javascript}</script></body>",
            1,
        )
    return document
