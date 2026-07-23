"""Read-only generation history summaries for the WebUI."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

_MAX_HISTORY_ITEMS = 200
_MAX_MANIFEST_BYTES = 1024 * 1024
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return None
    if not finite:
        return None
    return value


def _text(value: Any, max_length: int = 180) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:max_length] if value else None


def _read_manifest(folder: Path) -> dict[str, Any]:
    path = folder / "generation.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return _mapping(value)


def _regular_file(folder: Path, filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    candidate = folder / filename
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.parent == folder else None


def _thumbnail_filename(folder: Path, manifest: dict[str, Any]) -> str | None:
    inputs = _mapping(manifest.get("inputs"))
    candidates: list[str] = []
    for key in ("front", "image"):
        value = inputs.get(key)
        if isinstance(value, str):
            candidates.append(value)
    candidates.extend(value for value in inputs.values() if isinstance(value, str))
    candidates.extend(("input_front.png", "input_image.png", "input.png"))
    try:
        candidates.extend(path.name for path in sorted(folder.glob("input_*")))
    except OSError:
        pass

    seen: set[str] = set()
    for filename in candidates:
        if filename in seen or Path(filename).suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        seen.add(filename)
        if _regular_file(folder, filename):
            return filename
    return None


def _timestamp(value: Any, fallback: float) -> tuple[str, float]:
    text = _text(value)
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat(), parsed.timestamp()
        except ValueError:
            pass
    parsed = datetime.fromtimestamp(fallback, tz=timezone.utc)
    return parsed.isoformat(), fallback


def _model_name(manifest: dict[str, Any], stats: dict[str, Any]) -> str | None:
    model = _mapping(manifest.get("model")) or _mapping(stats.get("model"))
    shapegen = _text(model.get("shapegen"))
    return shapegen.rsplit("/", 1)[-1] if shapegen else None


def _history_item(folder: Path) -> tuple[dict[str, Any], float] | None:
    mesh = _regular_file(folder, "white_mesh.glb")
    if not mesh:
        return None
    mesh_stat = mesh.stat()

    manifest = _read_manifest(folder)
    params = _mapping(manifest.get("params"))
    stats = _mapping(manifest.get("stats"))
    if not params:
        params = _mapping(stats.get("params"))

    input_mode_value = _text(manifest.get("input_mode") or params.get("input_mode"))
    input_mode = (
        "four" if input_mode_value in {"four", "4-view", "multi-view"}
        else "single" if input_mode_value in {"single", "1-view", "single-view"}
        else None
    )
    views_used = params.get("views_used")
    view_count = len(views_used) if isinstance(views_used, list) else None
    if not view_count and input_mode:
        view_count = 4 if input_mode == "four" else 1

    stats_time = _mapping(stats.get("time"))
    completed_at, sort_timestamp = _timestamp(
        manifest.get("completed_at") or manifest.get("updated_at") or manifest.get("created_at"),
        mesh_stat.st_mtime,
    )
    created_at, _ = _timestamp(manifest.get("created_at"), sort_timestamp)
    thumbnail = _thumbnail_filename(folder, manifest)
    generation_uid = folder.name
    tracked = (
        manifest.get("schema_version") == 1
        and manifest.get("generation_uid") == generation_uid
        and (manifest.get("events") is None or isinstance(manifest.get("events"), list))
        and all(
            value is None or isinstance(value, dict)
            for value in (
                manifest.get("params"),
                manifest.get("inputs"),
                manifest.get("outputs"),
                manifest.get("stats"),
            )
        )
    )
    status = _text(manifest.get("status"), 24) or "completed"
    if status not in {"completed", "processing", "failed"}:
        status = "completed"

    faces = _number(manifest.get("number_of_faces"))
    vertices = _number(manifest.get("number_of_vertices"))
    if faces is None:
        faces = _number(stats.get("number_of_faces"))
    if vertices is None:
        vertices = _number(stats.get("number_of_vertices"))

    item = {
        "generation_uid": generation_uid,
        "status": status,
        "legacy": not tracked,
        "created_at": created_at,
        "completed_at": completed_at,
        "input_mode": input_mode,
        "view_count": view_count,
        "model": _model_name(manifest, stats),
        "parameters": {
            "steps": _number(params.get("steps")),
            "guidance_scale": _number(params.get("guidance_scale")),
            "seed": _number(params.get("seed")),
            "octree_resolution": _number(params.get("octree_resolution")),
            "num_chunks": _number(params.get("num_chunks")),
        },
        "statistics": {
            "seconds": _number(stats_time.get("total")),
            "faces": faces,
            "vertices": vertices,
            "mesh_bytes": mesh_stat.st_size,
        },
        "assets": {
            "thumbnail_url": (
                f"/static/{generation_uid}/{quote(thumbnail)}" if thumbnail else None
            ),
            "viewer_url": f"/generation-viewer/{generation_uid}",
            "download_url": f"/static/{generation_uid}/white_mesh.glb",
        },
    }
    return item, sort_timestamp


def list_generation_history(save_directory: str | Path, limit: int = _MAX_HISTORY_ITEMS) -> dict[str, Any]:
    """Return newest-first summaries for UUID folders containing a generated GLB."""
    root = Path(save_directory).resolve()
    limit = max(1, min(int(limit), _MAX_HISTORY_ITEMS))
    entries: list[tuple[dict[str, Any], float]] = []
    try:
        candidates = tuple(root.iterdir())
    except OSError:
        candidates = ()

    for candidate in candidates:
        try:
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            generation_uid = str(uuid.UUID(candidate.name))
            folder = candidate.resolve(strict=True)
        except (OSError, ValueError):
            continue
        if generation_uid != candidate.name or folder.parent != root:
            continue
        try:
            item = _history_item(folder)
        except OSError:
            continue
        if item:
            entries.append(item)

    entries.sort(key=lambda entry: (entry[1], entry[0]["generation_uid"]), reverse=True)
    total = len(entries)
    return {
        "items": [entry[0] for entry in entries[:limit]],
        "total": total,
        "has_more": total > limit,
    }
