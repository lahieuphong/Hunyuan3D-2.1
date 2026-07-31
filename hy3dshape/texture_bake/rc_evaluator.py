"""Production adapter for the Blender face/hair RC render worker.

The texture generation module accepts a small ``CandidateQualityEvaluator``
callable.  This module implements that callable without importing generation
itself (which keeps the dependency direction acyclic):

1. save the ten reference cut-outs beside the private candidate;
2. render the candidate with the same ten calibrated orthographic cameras;
3. crop both sides to an identical head region; and
4. pass the aligned crops to :func:`score_face_hair_rc`.

All evidence is retained below the candidate generation folder so a rejected
candidate can be inspected instead of leaving only a scalar score.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .face_hair_rc import (
    FaceHairRCConfig,
    FaceHairRCResult,
    score_face_hair_rc,
)
from .ten_view_consensus import TEN_VIEW_ANGLES


TEN_VIEW_NAMES = tuple(TEN_VIEW_ANGLES)
DEFAULT_RC_TIMEOUT_SECONDS = 600
_BLENDER_RC_LOCK = threading.Lock()

ReferenceImage = Image.Image | np.ndarray | str | os.PathLike[str]
CommandRunner = Callable[..., subprocess.CompletedProcess[str] | object]


def resolve_blender_executable(
    explicit: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve Blender without importing ``texture_bake.generation``.

    Keeping this resolver local is intentional: generation imports/constructs
    this evaluator, so importing generation here would create a cycle.
    """

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for variable in ("HUNYUAN3D_BLENDER_PATH", "BLENDER_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    discovered = shutil.which("blender")
    if discovered:
        candidates.append(Path(discovered))
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        blender_root = program_files / "Blender Foundation"
        candidates.extend(
            blender_root / version / "blender.exe"
            for version in (
                "Blender 5.0",
                "Blender 4.5",
                "Blender 4.4",
                "Blender 4.3",
                "Blender 4.2",
                "Blender 4.1",
                "Blender 4.0",
            )
        )
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Blender was not found. Set HUNYUAN3D_BLENDER_PATH to blender.exe."
    )


def _reference_rgba(value: ReferenceImage, *, name: str) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value.convert("RGBA")
    elif isinstance(value, (str, os.PathLike)):
        with Image.open(value) as loaded:
            image = loaded.convert("RGBA")
    else:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError(f"reference view {name!r} must be HxWx4 RGBA")
        if np.issubdtype(array.dtype, np.floating):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"reference view {name!r} contains non-finite values"
                )
            maximum = float(array.max()) if array.size else 0.0
            if maximum <= 1.0 + 1.0e-6:
                array = array * 255.0
        image = Image.fromarray(
            np.clip(np.rint(array), 0, 255).astype(np.uint8),
            mode="RGBA",
        )
    if image.width < 2 or image.height < 2:
        raise ValueError(f"reference view {name!r} is too small")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    foreground = int(np.count_nonzero(alpha >= 26))
    background = int(np.count_nonzero(alpha < 26))
    if foreground < 32:
        raise ValueError(
            f"reference view {name!r} has no usable foreground alpha"
        )
    if background < max(4, int(round(alpha.size * 0.002))):
        raise ValueError(
            f"reference view {name!r} needs a transparent background for RC"
        )
    return image


def _safe_candidate_id(candidate: Path) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.stem).strip("._")
    return identifier or "candidate"


def _head_roi(
    reference: Image.Image,
    *,
    head_fraction: float,
    margin_fraction: float,
) -> tuple[int, int, int, int]:
    alpha = np.asarray(reference.getchannel("A"), dtype=np.uint8)
    rows, columns = np.nonzero(alpha >= 26)
    if rows.size < 32:
        raise ValueError("reference does not contain a usable alpha silhouette")
    top = int(rows.min())
    bottom = int(rows.max()) + 1
    object_height = bottom - top
    head_bottom = min(
        bottom,
        max(top + 2, int(round(top + object_height * head_fraction))),
    )

    upper = alpha[top:head_bottom] >= 26
    upper_rows, upper_columns = np.nonzero(upper)
    if upper_rows.size < 32:
        raise ValueError("reference head region is too small for RC")
    head_left = int(upper_columns.min())
    head_right = int(upper_columns.max()) + 1
    head_top = top + int(upper_rows.min())
    head_bottom = top + int(upper_rows.max()) + 1
    margin = max(
        2,
        int(
            round(
                max(head_right - head_left, head_bottom - head_top)
                * margin_fraction
            )
        ),
    )
    return (
        max(0, head_left - margin),
        max(0, head_top - margin),
        min(reference.width, head_right + margin),
        min(reference.height, head_bottom + margin),
    )


def _load_render(path: Path, *, name: str) -> Image.Image:
    if not path.is_file():
        raise RuntimeError(f"Blender did not render RC view {name!r}: {path}")
    with Image.open(path) as loaded:
        rendered = loaded.convert("RGBA")
    if rendered.width < 2 or rendered.height < 2:
        raise RuntimeError(f"rendered RC view {name!r} is too small")
    return rendered


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, path)


@dataclass
class FaceHairRCCandidateEvaluator:
    """Callable adapter matching generation's candidate evaluator contract."""

    blender_path: str | os.PathLike[str] | None = None
    resolution_scale: float = 0.5
    timeout: int = DEFAULT_RC_TIMEOUT_SECONDS
    head_fraction: float = 0.38
    head_margin_fraction: float = 0.06
    config: FaceHairRCConfig | None = None
    runner: CommandRunner | None = None

    def __post_init__(self) -> None:
        if not 0.1 <= self.resolution_scale <= 1.0:
            raise ValueError("resolution_scale must be between 0.1 and 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not 0.2 <= self.head_fraction <= 0.6:
            raise ValueError("head_fraction must be between 0.2 and 0.6")
        if not 0.0 <= self.head_margin_fraction <= 0.25:
            raise ValueError(
                "head_margin_fraction must be between zero and 0.25"
            )
        if self.config is not None:
            self.config.validate()

    def _validate_inputs(
        self,
        view_names: Sequence[str],
        images: Mapping[str, ReferenceImage],
    ) -> dict[str, Image.Image]:
        names = tuple(str(name) for name in view_names)
        expected = set(TEN_VIEW_NAMES)
        if len(names) != len(TEN_VIEW_NAMES) or set(names) != expected:
            missing = sorted(expected - set(names))
            extra = sorted(set(names) - expected)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            raise ValueError(
                "face/hair RC requires all ten canonical views"
                + (": " + "; ".join(detail) if detail else "")
            )
        if len(set(names)) != len(names):
            raise ValueError("face/hair RC view_names must be unique")
        missing_images = [name for name in TEN_VIEW_NAMES if name not in images]
        if missing_images:
            raise ValueError(
                "face/hair RC requires all ten reference images: missing "
                + ", ".join(missing_images)
            )
        return {
            name: _reference_rgba(images[name], name=name)
            for name in TEN_VIEW_NAMES
        }

    def __call__(
        self,
        candidate_path: str | os.PathLike[str],
        view_names: Sequence[str],
        images: Mapping[str, ReferenceImage],
    ) -> FaceHairRCResult:
        candidate = Path(candidate_path).expanduser().resolve(strict=True)
        if not candidate.is_file() or candidate.suffix.casefold() != ".glb":
            raise ValueError("face/hair RC candidate must be an existing GLB")
        references = self._validate_inputs(view_names, images)
        blender = resolve_blender_executable(self.blender_path)
        repository_root = Path(__file__).resolve().parents[2]
        worker = (
            repository_root
            / "hy3dshape"
            / "scripts"
            / "render_face_hair_rc_views_blender.py"
        ).resolve(strict=True)

        audit_root = (
            candidate.parent
            / "quality_audit"
            / "face_hair_rc"
            / _safe_candidate_id(candidate)
        )
        reference_directory = audit_root / "references_full"
        aligned_reference_directory = audit_root / "references_aligned"
        render_directory = audit_root / "renders_full"
        reference_head_directory = audit_root / "references_head"
        render_head_directory = audit_root / "renders_head"
        for directory in (
            reference_directory,
            aligned_reference_directory,
            render_directory,
            reference_head_directory,
            render_head_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        reference_paths: dict[str, Path] = {}
        for name in TEN_VIEW_NAMES:
            path = reference_directory / f"{name}.png"
            references[name].save(path, format="PNG")
            reference_paths[name] = path

        worker_report = audit_root / "render_report.json"
        for name in TEN_VIEW_NAMES:
            (render_directory / f"{name}.png").unlink(missing_ok=True)
        worker_report.unlink(missing_ok=True)
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(worker),
            "--",
            "--mesh",
            str(candidate),
        ]
        for name in TEN_VIEW_NAMES:
            command.extend(
                (
                    f"--{name.replace('_', '-')}",
                    str(reference_paths[name]),
                )
            )
        command.extend(
            (
                "--output-dir",
                str(render_directory),
                "--report",
                str(worker_report),
                "--resolution-scale",
                str(self.resolution_scale),
            )
        )
        run = self.runner or subprocess.run
        with _BLENDER_RC_LOCK:
            run(
                command,
                cwd=str(repository_root),
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=False,
            )
        if not worker_report.is_file():
            raise RuntimeError("Blender RC worker did not write its render report")
        try:
            worker_payload = json.loads(worker_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Blender RC render report is invalid") from error
        if not isinstance(worker_payload, Mapping):
            raise RuntimeError("Blender RC render report must be a JSON object")

        rendered_heads: dict[str, Path] = {}
        reference_heads: dict[str, Path] = {}
        evidence: dict[str, object] = {}
        for name in TEN_VIEW_NAMES:
            full_render_path = render_directory / f"{name}.png"
            rendered = _load_render(full_render_path, name=name)
            reference = references[name]
            if reference.size != rendered.size:
                reference = reference.resize(
                    rendered.size,
                    resample=Image.Resampling.LANCZOS,
                )
            aligned_reference_path = aligned_reference_directory / f"{name}.png"
            reference.save(aligned_reference_path, format="PNG")
            roi = _head_roi(
                reference,
                head_fraction=self.head_fraction,
                margin_fraction=self.head_margin_fraction,
            )
            rendered_head = rendered.crop(roi)
            reference_head = reference.crop(roi)
            rendered_head_path = render_head_directory / f"{name}.png"
            reference_head_path = reference_head_directory / f"{name}.png"
            rendered_head.save(rendered_head_path, format="PNG")
            reference_head.save(reference_head_path, format="PNG")
            rendered_heads[name] = rendered_head_path
            reference_heads[name] = reference_head_path
            evidence[name] = {
                "roi_xyxy": list(roi),
                "full_size": list(rendered.size),
                "reference_full": str(reference_paths[name].resolve()),
                "reference_aligned": str(aligned_reference_path.resolve()),
                "render_full": str(full_render_path.resolve()),
                "reference_head": str(reference_head_path.resolve()),
                "render_head": str(rendered_head_path.resolve()),
            }

        result = score_face_hair_rc(
            [rendered_heads[name] for name in TEN_VIEW_NAMES],
            [reference_heads[name] for name in TEN_VIEW_NAMES],
            view_names=TEN_VIEW_NAMES,
            config=self.config,
        )
        _atomic_json_write(
            audit_root / "evaluation.json",
            {
                "schema_version": 1,
                "metric": "face-hair-reprojection-region-consistency",
                "candidate": str(candidate),
                "worker_report": str(worker_report.resolve()),
                "resolution_scale": self.resolution_scale,
                "head_fraction": self.head_fraction,
                "head_margin_fraction": self.head_margin_fraction,
                "views": evidence,
                "result": result.to_dict(),
            },
        )
        return result


# Short alias for call sites that already identify the metric in the variable
# name (``quality_evaluator = FaceHairRCEvaluator(...)``).
FaceHairRCEvaluator = FaceHairRCCandidateEvaluator


__all__ = [
    "DEFAULT_RC_TIMEOUT_SECONDS",
    "FaceHairRCCandidateEvaluator",
    "FaceHairRCEvaluator",
    "TEN_VIEW_NAMES",
    "resolve_blender_executable",
]
