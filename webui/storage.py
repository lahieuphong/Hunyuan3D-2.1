"""Canonical filesystem layout and legacy migration for WebUI artifacts.

Historically, generation UUID folders were written directly below the WebUI
output root alongside logs and copied static assets.  This module keeps the
layout definition in one place and moves only unmistakable legacy generation
directories into the dedicated ``generations`` directory.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WebUIStorageLayout:
    """Resolved paths owned by one WebUI output root."""

    root: Path
    generations: Path
    logs: Path
    runtime: Path
    training: Path
    inference: Path
    quality_tests: Path
    projects: Path
    archive: Path

    @classmethod
    def from_root(cls, root: str | os.PathLike[str]) -> "WebUIStorageLayout":
        """Build a canonical layout without creating or moving anything."""

        resolved_root = Path(root).expanduser().resolve(strict=False)
        return cls(
            root=resolved_root,
            generations=resolved_root / "generations",
            logs=resolved_root / "logs",
            runtime=resolved_root / "runtime",
            training=resolved_root / "training",
            inference=resolved_root / "inference",
            quality_tests=resolved_root / "quality_tests",
            projects=resolved_root / "projects",
            archive=resolved_root / "archive",
        )

    def directories(self) -> tuple[Path, ...]:
        """Return every canonical child path in stable order."""

        return (
            self.generations,
            self.logs,
            self.runtime,
            self.training,
            self.inference,
            self.quality_tests,
            self.projects,
            self.archive,
        )

    def runtime_directories(self) -> tuple[Path, ...]:
        """Return directories the WebUI itself owns and must create eagerly."""

        return (self.generations, self.logs, self.runtime)


class StorageMigrationCollisionError(FileExistsError):
    """Raised before migration when a destination would be overwritten."""

    def __init__(self, collisions: tuple[Path, ...]):
        self.collisions = collisions
        rendered = ", ".join(str(path) for path in collisions)
        super().__init__(
            "Cannot migrate legacy WebUI generations because destination "
            f"paths already exist: {rendered}"
        )


def prepare_webui_storage(
    root: str | os.PathLike[str],
    migrate_legacy: bool = True,
) -> WebUIStorageLayout:
    """Create the canonical layout and optionally migrate legacy generations.

    Migration is deliberately narrow: only direct child directories whose
    names are canonical lowercase UUID strings are considered.  Files,
    symlinks, Windows reparse points, non-canonical UUID spellings, and all
    other directories remain untouched.

    All destination collisions are detected before the first directory is
    moved, so migration never merges with or overwrites existing data.
    Re-running this function after a successful migration is a no-op.
    """

    # Validate the caller's absolute, non-resolved path first.  Resolving it
    # earlier would hide a symlink/reparse-point root; resolving it afterwards
    # gives the rest of the application one stable canonical spelling (also
    # avoiding Windows 8.3 short-path mismatches).
    unchecked_root = Path(os.path.abspath(Path(root).expanduser()))
    _ensure_directory(unchecked_root, parents=True)
    layout = WebUIStorageLayout.from_root(unchecked_root)
    # Offline workflow folders are created by their own producer on demand.
    # Eagerly creating empty training/projects/archive folders makes the output
    # tree look populated even after the user has intentionally cleaned it.
    for directory in layout.runtime_directories():
        _ensure_directory(directory)

    if migrate_legacy:
        _migrate_legacy_generations(layout)
    _normalize_generation_manifests(layout.generations)
    return layout


def _ensure_directory(path: Path, *, parents: bool = False) -> None:
    """Create a real directory without accepting a link as storage."""

    if os.path.lexists(path):
        if _is_link_or_reparse_point(path):
            raise OSError(f"Storage path must not be a link or reparse point: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Storage path is not a directory: {path}")
        return
    path.mkdir(parents=parents, exist_ok=False)


def _migrate_legacy_generations(layout: WebUIStorageLayout) -> None:
    candidates = tuple(_legacy_generation_directories(layout.root))
    collisions = tuple(
        layout.generations / source.name
        for source in candidates
        if os.path.lexists(layout.generations / source.name)
    )
    if collisions:
        raise StorageMigrationCollisionError(collisions)

    for source in candidates:
        destination = layout.generations / source.name
        os.replace(source, destination)


def _normalize_generation_manifests(generations: Path) -> None:
    """Repair manifests after both completed and interrupted migrations."""

    for generation_folder in _canonical_generation_directories(generations):
        _update_generation_manifest(generation_folder)


def _legacy_generation_directories(root: Path):
    yield from _canonical_generation_directories(root)


def _canonical_generation_directories(root: Path):
    resolved_root = root.resolve(strict=True)
    for candidate in sorted(root.iterdir(), key=lambda path: path.name):
        if not _is_canonical_uuid(candidate.name):
            continue
        if _is_link_or_reparse_point(candidate) or not candidate.is_dir():
            continue
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved_candidate.parent != resolved_root:
            continue
        yield candidate


def _is_canonical_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except (AttributeError, TypeError, ValueError):
        return False


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(metadata.st_mode):
        return True
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(file_attributes & reparse_attribute)


def _update_generation_manifest(generation_folder: Path) -> None:
    manifest_path = generation_folder / "generation.json"
    if not os.path.lexists(manifest_path):
        return
    if _is_link_or_reparse_point(manifest_path) or not manifest_path.is_file():
        return

    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A malformed historical manifest is user data.  Moving its enclosing
        # generation is safe; guessing how to rewrite the file is not.
        return

    storage_folder = _display_storage_path(generation_folder)
    if not _replace_storage_folder_fields(manifest, storage_folder):
        return
    _atomic_write_json(manifest_path, manifest)


def _replace_storage_folder_fields(value: Any, storage_folder: str) -> int:
    replacements = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "storage_folder":
                if child != storage_folder:
                    value[key] = storage_folder
                    replacements += 1
            else:
                replacements += _replace_storage_folder_fields(
                    child,
                    storage_folder,
                )
    elif isinstance(value, list):
        for child in value:
            replacements += _replace_storage_folder_fields(child, storage_folder)
    return replacements


def _display_storage_path(path: Path) -> str:
    """Match the portable path format used by WebUI generation manifests."""

    try:
        value = os.path.relpath(path, start=Path.cwd())
    except ValueError:
        value = str(path.resolve(strict=False))
    return value.replace(os.sep, "/")


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(value, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "StorageMigrationCollisionError",
    "WebUIStorageLayout",
    "prepare_webui_storage",
]
