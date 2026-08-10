#!/usr/bin/env python
"""Safely migrate the legacy Hunyuan3D output tree into WebUI storage.

The command is a dry run unless ``--apply`` is supplied.  It only renames
whole files or directories, never merges directories, overwrites a target,
or removes duplicate data.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SHAPE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SHAPE_ROOT.parent
DEFAULT_OUTPUT_ROOT = SHAPE_ROOT / "output_folder"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui.storage import WebUIStorageLayout, prepare_webui_storage  # noqa: E402


_CANONICAL_WEBUI_NAMES = frozenset(
    {
        "generations",
        "logs",
        "runtime",
        "training",
        "inference",
        "quality_tests",
        "projects",
        "archive",
    }
)


class OutputMigrationError(RuntimeError):
    """Base error for migrations that must stop without moving data."""


class OutputRootValidationError(OutputMigrationError, ValueError):
    """Raised when the supplied path is not a safe legacy output root."""


class OutputMigrationCollisionError(OutputMigrationError, FileExistsError):
    """Raised when at least one planned destination already exists."""

    def __init__(self, collisions: Sequence[Path]):
        self.collisions = tuple(collisions)
        rendered = ", ".join(str(path) for path in self.collisions)
        super().__init__(
            f"Refusing to overwrite or merge existing migration targets: {rendered}"
        )


class LiveWebUIProcessError(OutputMigrationError):
    """Raised when a legacy or canonical WebUI PID is still alive."""

    def __init__(self, live_processes: Sequence[tuple[int, Path]]):
        self.live_processes = tuple(live_processes)
        rendered = ", ".join(
            f"PID {pid} ({pid_file})" for pid, pid_file in self.live_processes
        )
        super().__init__(
            f"Refusing to migrate while a managed WebUI process is alive: {rendered}"
        )


@dataclass(frozen=True, slots=True)
class MigrationAction:
    """One no-overwrite rename in a deterministic migration phase."""

    source: Path
    destination: Path
    phase: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """A validated, read-only snapshot of the work to perform."""

    output_root: Path
    layout: WebUIStorageLayout
    before_storage: tuple[MigrationAction, ...]
    generations: tuple[MigrationAction, ...]
    after_storage: tuple[MigrationAction, ...]

    @property
    def actions(self) -> tuple[MigrationAction, ...]:
        return self.before_storage + self.generations + self.after_storage


def validate_output_root(output_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Return real output/WebUI roots, rejecting ambiguous or linked paths."""

    candidate = Path(output_root).expanduser()
    if not os.path.lexists(candidate):
        raise OutputRootValidationError(f"Output root does not exist: {candidate}")
    if _is_link_or_reparse_point(candidate) or not candidate.is_dir():
        raise OutputRootValidationError(
            f"Output root must be a real directory, not a link: {candidate}"
        )

    resolved_root = candidate.resolve(strict=True)
    if resolved_root.parent == resolved_root:
        raise OutputRootValidationError(
            f"Refusing to use a filesystem root as the output root: {resolved_root}"
        )

    webui_root = resolved_root / "webui"
    if not os.path.lexists(webui_root):
        raise OutputRootValidationError(
            f"Expected an existing WebUI directory below the output root: {webui_root}"
        )
    if _is_link_or_reparse_point(webui_root) or not webui_root.is_dir():
        raise OutputRootValidationError(
            f"WebUI root must be a real directory, not a link: {webui_root}"
        )
    if webui_root.resolve(strict=True).parent != resolved_root:
        raise OutputRootValidationError(
            f"WebUI root escapes the output root: {webui_root}"
        )
    return resolved_root, webui_root


def build_migration_plan(
    output_root: str | os.PathLike[str],
    *,
    process_is_alive: Callable[[int], bool] | None = None,
) -> MigrationPlan:
    """Inspect and validate a migration without changing the filesystem."""

    resolved_root, webui_root = validate_output_root(output_root)
    layout = WebUIStorageLayout.from_root(webui_root)
    _validate_canonical_parents(layout)
    _refuse_live_webui_processes(
        layout,
        process_is_alive=process_is_alive or _process_is_alive,
    )

    before_storage: list[MigrationAction] = []
    generations: list[MigrationAction] = []
    after_storage: list[MigrationAction] = []

    sibling_moves = (
        ("dit", layout.training),
        ("inference", layout.inference),
        ("quality_tests", layout.quality_tests),
        ("cau_mong_multiview", layout.projects / "cau_mong_multiview"),
    )
    for source_name, destination in sibling_moves:
        source = resolved_root / source_name
        action = _optional_typed_move(
            source, destination, "before-storage", directory=True
        )
        if action is not None:
            before_storage.append(action)

    runlogs_action = _optional_typed_move(
        resolved_root / "_runlogs",
        layout.logs / "training_smoke",
        "after-storage",
        directory=True,
    )
    if runlogs_action is not None:
        after_storage.append(runlogs_action)

    generation_names: set[str] = set()
    for candidate in sorted(webui_root.iterdir(), key=lambda path: path.name):
        if not _is_canonical_uuid(candidate.name):
            continue
        if _is_link_or_reparse_point(candidate):
            raise OutputRootValidationError(
                f"Legacy generation must not be a link or reparse point: {candidate}"
            )
        if not candidate.is_dir():
            continue
        generation_names.add(candidate.name)
        generations.append(
            MigrationAction(
                source=candidate,
                destination=layout.generations / candidate.name,
                phase="generation",
            )
        )

    direct_stderr = webui_root / "webui.stderr.log"
    stderr_action = _optional_typed_move(
        direct_stderr,
        layout.logs / "legacy" / direct_stderr.name,
        "after-storage",
        directory=False,
    )
    if stderr_action is not None:
        after_storage.append(stderr_action)

    env_maps = webui_root / "env_maps"
    env_maps_action = _optional_typed_move(
        env_maps,
        layout.archive / "legacy" / env_maps.name,
        "after-storage",
        directory=True,
    )
    if env_maps_action is not None:
        after_storage.append(env_maps_action)

    handled_names = (
        _CANONICAL_WEBUI_NAMES
        | generation_names
        | {
            "env_maps",
            "webui.stderr.log",
        }
    )
    for candidate in sorted(webui_root.iterdir(), key=lambda path: path.name):
        if candidate.name in handled_names:
            continue
        _validate_move_source(candidate)
        after_storage.append(
            MigrationAction(
                source=candidate,
                destination=layout.archive / "legacy" / candidate.name,
                phase="after-storage",
            )
        )

    plan = MigrationPlan(
        output_root=resolved_root,
        layout=layout,
        before_storage=tuple(before_storage),
        generations=tuple(generations),
        after_storage=tuple(after_storage),
    )
    _validate_destinations(plan.actions)
    return plan


def migrate_output_layout(
    output_root: str | os.PathLike[str] = DEFAULT_OUTPUT_ROOT,
    *,
    apply: bool = False,
    process_is_alive: Callable[[int], bool] | None = None,
) -> MigrationPlan:
    """Build a plan and optionally apply it using no-overwrite renames."""

    checker = process_is_alive or _process_is_alive
    plan = build_migration_plan(output_root, process_is_alive=checker)
    if not apply:
        return plan

    # Recheck immediately before the first mutation in case the service was
    # started while the plan was being built.
    _refuse_live_webui_processes(plan.layout, process_is_alive=checker)
    _validate_destinations(plan.actions)

    for action in plan.before_storage:
        _ensure_real_directory(action.destination.parent)
        _rename_without_overwrite(action)

    # This creates the canonical directories, migrates direct UUID folders,
    # and updates their storage_folder manifest fields atomically.
    prepare_webui_storage(plan.layout.root, migrate_legacy=True)

    for action in plan.after_storage:
        _ensure_real_directory(action.destination.parent)
        _rename_without_overwrite(action)

    return plan


def _optional_typed_move(
    source: Path,
    destination: Path,
    phase: str,
    *,
    directory: bool,
) -> MigrationAction | None:
    if not os.path.lexists(source):
        return None
    _validate_move_source(source)
    if directory and not source.is_dir():
        raise OutputRootValidationError(f"Expected a directory: {source}")
    if not directory and not source.is_file():
        raise OutputRootValidationError(f"Expected a file: {source}")
    return MigrationAction(source, destination, phase)


def _validate_move_source(source: Path) -> None:
    if _is_link_or_reparse_point(source):
        raise OutputRootValidationError(
            f"Migration source must not be a link or reparse point: {source}"
        )
    if not source.is_dir() and not source.is_file():
        raise OutputRootValidationError(f"Unsupported migration source: {source}")


def _validate_canonical_parents(layout: WebUIStorageLayout) -> None:
    for directory in layout.directories():
        if not os.path.lexists(directory):
            continue
        if _is_link_or_reparse_point(directory) or not directory.is_dir():
            raise OutputRootValidationError(
                f"Canonical storage path must be a real directory: {directory}"
            )

    for directory in (layout.logs / "legacy", layout.archive / "legacy"):
        if not os.path.lexists(directory):
            continue
        if _is_link_or_reparse_point(directory) or not directory.is_dir():
            raise OutputRootValidationError(
                f"Migration destination parent must be a real directory: {directory}"
            )


def _validate_destinations(actions: Sequence[MigrationAction]) -> None:
    destinations: set[Path] = set()
    collisions: list[Path] = []
    for action in actions:
        if action.destination in destinations or os.path.lexists(action.destination):
            collisions.append(action.destination)
        destinations.add(action.destination)
    if collisions:
        raise OutputMigrationCollisionError(tuple(sorted(set(collisions))))


def _rename_without_overwrite(action: MigrationAction) -> None:
    if not os.path.lexists(action.source):
        raise OutputMigrationError(
            f"Migration source disappeared before it could be moved: {action.source}"
        )
    if os.path.lexists(action.destination):
        raise OutputMigrationCollisionError((action.destination,))
    action.source.rename(action.destination)


def _ensure_real_directory(directory: Path) -> None:
    if os.path.lexists(directory):
        if _is_link_or_reparse_point(directory) or not directory.is_dir():
            raise OutputRootValidationError(
                f"Migration destination parent must be a real directory: {directory}"
            )
        return
    parent = directory.parent
    if parent != directory and not os.path.lexists(parent):
        _ensure_real_directory(parent)
    directory.mkdir(parents=False, exist_ok=False)


def _managed_pid_files(layout: WebUIStorageLayout) -> tuple[Path, ...]:
    result: list[Path] = []
    for directory in (layout.logs, layout.runtime):
        if not directory.is_dir() or _is_link_or_reparse_point(directory):
            continue
        result.extend(directory.glob("webui-*.pid"))
    return tuple(sorted(set(result)))


def _refuse_live_webui_processes(
    layout: WebUIStorageLayout,
    *,
    process_is_alive: Callable[[int], bool],
) -> None:
    live_processes: list[tuple[int, Path]] = []
    for pid_file in _managed_pid_files(layout):
        if _is_link_or_reparse_point(pid_file) or not pid_file.is_file():
            raise OutputRootValidationError(f"Unsafe managed PID file: {pid_file}")
        try:
            pid = int(pid_file.read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            continue
        if pid > 0 and process_is_alive(pid):
            live_processes.append((pid, pid_file))
    if live_processes:
        raise LiveWebUIProcessError(live_processes)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _windows_process_is_alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Legacy output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the displayed no-overwrite moves. Without this flag, only plan.",
    )
    return parser


def _print_plan(plan: MigrationPlan, *, apply: bool) -> None:
    mode = "APPLIED" if apply else "DRY-RUN"
    print(f"{mode}: {plan.output_root}")
    for action in plan.actions:
        print(f"MOVE [{action.phase}] {action.source} -> {action.destination}")
    if not plan.actions:
        print("No legacy output items need migration.")
    elif not apply:
        print("No files were changed. Re-run with --apply to execute this plan.")
    else:
        print(f"Moved {len(plan.actions)} legacy item(s) without overwrite or merge.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = migrate_output_layout(args.output_root, apply=args.apply)
    except OutputMigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_plan(plan, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiveWebUIProcessError",
    "MigrationAction",
    "MigrationPlan",
    "OutputMigrationCollisionError",
    "OutputMigrationError",
    "OutputRootValidationError",
    "build_migration_plan",
    "main",
    "migrate_output_layout",
    "validate_output_root",
]
