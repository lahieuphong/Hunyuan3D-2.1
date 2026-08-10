"""Create the colored ``Original`` variant for a generated white mesh.

The normal Web UI process deliberately starts with ``--disable_tex`` so the
large Hunyuan Paint model does not compete with shape generation for VRAM.
This module therefore uses the lightweight visibility-aware Blender baker
already shipped in this repository.  The quality-validated default consumes
the four cardinal reference images and embeds a real texture in
``textured_mesh.glb``.  A native ten-view consensus bake remains available as
an explicit option and as a fallback when the cardinal bake cannot complete.
Six-view requests use a separate orthographic vertex-color projection so the
Top and Bottom references contribute to the published GLB while shape
conditioning remains on the checkpoint's four native camera slots.

If Blender is unavailable or its bake fails, a deterministic vertex-color
fallback projects every supplied supported camera onto the
surface.  The fallback is still a genuinely colored GLB (``COLOR_0``), never a
renamed copy of the white mesh.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.color import ColorVisuals

from hy3dshape.six_view import (
    SIX_VIEW_ANGLES,
    SIX_VIEW_AUXILIARY_KEYS,
    SIX_VIEW_KEYS,
)

from .calibration import ViewFrame, fit_orthographic_from_alpha
from .face_hair_rc import (
    FaceHairRCResult,
    FaceHairRCView,
    rank_face_hair_candidates,
)
from .visibility import (
    alpha_confidence_map,
    bilinear_sample,
    blend_view_colors,
    compute_view_confidence,
    diffuse_surface_colors,
)


CARDINAL_VIEW_KEYS = ("front", "left", "back", "right")
VIEW_ANGLES = {
    "front": (0.0, 0.0),
    "front_right": (45.0, 0.0),
    "right": (90.0, 0.0),
    "back_right": (135.0, 0.0),
    "back": (180.0, 0.0),
    "back_left": (225.0, 0.0),
    "left": (270.0, 0.0),
    "front_left": (315.0, 0.0),
    "high_front": (0.0, 30.0),
    "high_back": (180.0, 30.0),
}
TEN_VIEW_KEYS = tuple(VIEW_ANGLES)
SUPPORTED_VIEW_ANGLES = {
    **VIEW_ANGLES,
    **SIX_VIEW_ANGLES,
}
DEFAULT_TEXTURE_SIZE = 4096
DEFAULT_ATLAS_TILE_SIZE = 1536
DEFAULT_TIMEOUT_SECONDS = 600
_MAX_GLB_JSON_BYTES = 16 * 1024 * 1024
_GLB_JSON_CHUNK = 0x4E4F534A
_BLENDER_BAKE_LOCK = threading.Lock()

CandidateQualityEvaluator = Callable[
    [Path, tuple[str, ...], Mapping[str, Image.Image]],
    Mapping[str, object] | object,
]

def _default_ten_view_quality_evaluator(
    *,
    blender_path: str | os.PathLike[str] | None,
    timeout: int,
) -> CandidateQualityEvaluator:
    """Build the production RC evaluator for direct ten-view API callers.

    The Web UI injects the same evaluator explicitly.  Constructing it lazily
    here closes the lower-level API path where ten references previously
    declared RC as required but could still publish in structural-only mode.
    The local import keeps the generation/evaluator dependency acyclic.
    """

    from .rc_evaluator import FaceHairRCCandidateEvaluator

    return FaceHairRCCandidateEvaluator(
        blender_path=blender_path,
        timeout=timeout,
    )



class OriginalVariantError(RuntimeError):
    """Raised when neither texture baking nor vertex coloring can succeed."""


@dataclass(frozen=True)
class OriginalVariantResult:
    """Files and provenance for one generated colored mesh."""

    output_path: Path
    color_payload: str
    method: str
    views_used: tuple[str, ...]
    source_strategy: str
    seconds: float
    report_path: Path | None = None
    texture_path: Path | None = None
    fallback_reason: str | None = None
    quality_gate_path: Path | None = None
    quality_gate: Mapping[str, object] | None = None

    def to_metadata(self) -> dict[str, object]:
        result: dict[str, object] = {
            "file": self.output_path.name,
            "color_payload": self.color_payload,
            "method": self.method,
            "views_used": list(self.views_used),
            "source_strategy": self.source_strategy,
            "seconds": round(self.seconds, 3),
        }
        if self.report_path is not None:
            result["report"] = self.report_path.name
        if self.texture_path is not None:
            result["texture"] = self.texture_path.name
        if self.fallback_reason:
            result["fallback_reason"] = self.fallback_reason
        if self.quality_gate_path is not None:
            result["quality_gate_report"] = self.quality_gate_path.name
        if self.quality_gate is not None:
            result["quality_gate"] = dict(self.quality_gate)
        return result


@dataclass(frozen=True)
class _EvaluatedOriginalCandidate:
    """A private, structurally valid candidate with a complete RC decision."""

    identifier: str
    candidate_path: Path
    color_payload: str
    method: str
    views_used: tuple[str, ...]
    source_strategy: str
    seconds: float
    evaluation: Mapping[str, object]
    rc_result: FaceHairRCResult | None
    rejection_path: Path
    report_path: Path | None = None
    texture_path: Path | None = None
    fallback_reason: str | None = None


class _OriginalCandidateEvaluated(OriginalVariantError):
    """Carry one private RC-scored candidate to the best-of orchestrator."""

    def __init__(self, candidate: _EvaluatedOriginalCandidate):
        state = (
            "accepted"
            if (
                candidate.rc_result is not None
                and candidate.rc_result.passed_hard_gates
            )
            else "rejected"
        )
        super().__init__(
            "Original candidate was RC-evaluated and kept private "
            f"for best-of selection: {candidate.identifier} ({state})"
        )
        self.candidate = candidate


class _OriginalCandidateRejected(_OriginalCandidateEvaluated):
    """Backward-compatible signal for an RC-rejected private candidate."""


# Private compatibility alias retained for focused regression tests and any
# local audit scripts written against the first RC best-of implementation.
_RejectedOriginalCandidate = _EvaluatedOriginalCandidate


def _json_object(value: Mapping[str, object] | object) -> dict[str, object]:
    """Normalise a quality evaluator result without coupling to its scorer."""

    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise OriginalVariantError(
        "Original candidate quality evaluator must return a mapping or "
        "an object with to_dict()."
    )


def _face_hair_rc_result(
    raw_value: Mapping[str, object] | object,
    payload: Mapping[str, object],
) -> FaceHairRCResult | None:
    """Recover a rankable RC result from an evaluator response."""

    if isinstance(raw_value, FaceHairRCResult):
        return raw_value
    views_value = payload.get("views")
    failures_value = payload.get("hard_gate_failures")
    passed_value = payload.get("passed_hard_gates")
    if (
        not isinstance(views_value, Sequence)
        or isinstance(views_value, (str, bytes))
        or not isinstance(failures_value, Sequence)
        or isinstance(failures_value, (str, bytes))
        or not isinstance(passed_value, bool)
    ):
        return None
    try:
        views = tuple(
            FaceHairRCView(
                name=str(view["name"]),
                score=float(view["score"]),
                semantic_score=float(view["semantic_score"]),
                color_score=float(view["color_score"]),
                detail_score=float(view["detail_score"]),
                foreground_iou=float(view["foreground_iou"]),
                face_leakage=(
                    None
                    if view.get("face_leakage") is None
                    else float(view["face_leakage"])
                ),
                hair_leakage=(
                    None
                    if view.get("hair_leakage") is None
                    else float(view["hair_leakage"])
                ),
                face_pixels=int(view["face_pixels"]),
                hair_pixels=int(view["hair_pixels"]),
            )
            for view in views_value
            if isinstance(view, Mapping)
        )
        if len(views) != len(views_value):
            return None
        result = FaceHairRCResult(
            score=float(payload["score"]),
            raw_score=float(payload["raw_score"]),
            passed_hard_gates=passed_value,
            hard_gate_failures=tuple(str(item) for item in failures_value),
            mean_view_score=float(payload["mean_view_score"]),
            worst_quartile_score=float(payload["worst_quartile_score"]),
            semantic_score=float(payload["semantic_score"]),
            color_score=float(payload["color_score"]),
            detail_score=float(payload["detail_score"]),
            foreground_iou=float(payload["foreground_iou"]),
            worst_quartile_face_leakage=(
                None
                if payload.get("worst_quartile_face_leakage") is None
                else float(payload["worst_quartile_face_leakage"])
            ),
            worst_quartile_hair_leakage=(
                None
                if payload.get("worst_quartile_hair_leakage") is None
                else float(payload["worst_quartile_hair_leakage"])
            ),
            views=views,
        )
    except (KeyError, TypeError, ValueError):
        return None
    numeric = (
        result.score,
        result.raw_score,
        result.mean_view_score,
        result.worst_quartile_score,
        result.semantic_score,
        result.color_score,
        result.detail_score,
        result.foreground_iou,
    )
    return result if all(np.isfinite(value) for value in numeric) else None


def _atomic_json_write(path: Path, document: Mapping[str, object]) -> None:
    pending_path = path.with_name(path.name + ".pending")
    pending_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending_path, path)


def _candidate_artifact_path(
    output_folder: Path,
    filename: str,
    candidate_identifier: str | None,
) -> Path:
    """Return a collision-free audit artifact path for RC best-of runs."""

    path = output_folder / filename
    if candidate_identifier is None:
        return path
    return path.with_name(
        f"{path.stem}.{candidate_identifier}{path.suffix}"
    )


def _candidate_pending_path(
    output_folder: Path,
    candidate_identifier: str | None,
) -> Path:
    if candidate_identifier is None:
        return output_folder / "textured_mesh.pending.glb"
    return output_folder / f"textured_mesh.{candidate_identifier}.pending.glb"


def _promote_original_candidate(
    pending_output: Path,
    output_path: Path,
    *,
    color_payload: str,
    method: str,
    views_used: tuple[str, ...],
    source_strategy: str,
    images: Mapping[str, Image.Image],
    quality_evaluator: CandidateQualityEvaluator | None,
    stage_callback: Callable[[str, str], None] | None,
    candidate_identifier: str | None = None,
    preserve_candidate: bool = False,
    seconds: float = 0.0,
    report_path: Path | None = None,
    texture_path: Path | None = None,
    fallback_reason: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Record the gate decision, then atomically promote one valid candidate.

    The optional evaluator is the integration point for the face/hair RC
    scorer. It receives the still-private pending GLB, every ordered reference
    view and its source image. Returning ``passed_hard_gates=False`` rejects the
    candidate before it can replace the canonical Original.
    """

    if not pending_output.is_file():
        raise OriginalVariantError("Original candidate GLB is missing.")
    detected_payload = glb_color_payload(pending_output)
    if detected_payload != color_payload:
        raise OriginalVariantError(
            "Original candidate failed its color-payload structural gate."
        )

    evaluation_views = tuple(
        name for name in TEN_VIEW_KEYS if name in images
    )
    rc_required = (
        source_strategy == "native-ten-view" or len(evaluation_views) == 10
    )
    rc: dict[str, object] = {
        "metric": "face-hair-reprojection-region-consistency",
        "required": rc_required,
        "reference_views": list(evaluation_views),
    }
    promotion_mode = "structural-only"
    if quality_evaluator is None:
        rc.update(
            {
                "status": "not_evaluated",
                "reason": (
                    "No aligned candidate head renders were supplied to the "
                    "RC evaluator."
                ),
            }
        )
    else:
        if stage_callback:
            stage_callback(
                "scoring_original",
                "Scoring face and hair reprojection consistency before publication",
            )
        selected_images = {
            name: images[name]
            for name in evaluation_views
            if name in images
        }
        try:
            raw_evaluation = quality_evaluator(
                pending_output, evaluation_views, selected_images
            )
        except Exception as error:
            identifier = candidate_identifier or pending_output.name
            raise OriginalVariantError(
                'Face/hair RC evaluation failed for candidate '
                f'{identifier}: {error}'
            ) from error
        evaluation = _json_object(raw_evaluation)
        parsed_rc_result = _face_hair_rc_result(raw_evaluation, evaluation)
        if candidate_identifier is not None and parsed_rc_result is None:
            identifier = candidate_identifier or pending_output.name
            raise OriginalVariantError(
                "Face/hair RC evaluation returned incomplete or non-finite "
                "metrics for best-of candidate "
                f"{identifier}."
            )
        if (
            candidate_identifier is not None
            and evaluation.get('passed_hard_gates') is True
        ):
            decision_rc = dict(rc)
            decision_rc.update(
                {
                    'status': 'evaluated',
                    'result': evaluation,
                }
            )
            decision_path = _candidate_artifact_path(
                output_path.parent,
                'original_quality_gate_candidate.json',
                candidate_identifier,
            )
            _atomic_json_write(
                decision_path,
                {
                    'schema_version': 1,
                    'candidate': pending_output.name,
                    'method': method,
                    'views_used': list(views_used),
                    'source_strategy': source_strategy,
                    'structural': {
                        'color_payload': detected_payload,
                        'valid': True,
                    },
                    'rc': decision_rc,
                    'promotion': {
                        'eligible': True,
                        'deferred': True,
                        'mode': 'rc-best-of',
                    },
                },
            )
            raise _OriginalCandidateEvaluated(
                _EvaluatedOriginalCandidate(
                    identifier=candidate_identifier,
                    candidate_path=pending_output,
                    color_payload=detected_payload,
                    method=method,
                    views_used=views_used,
                    source_strategy=source_strategy,
                    seconds=seconds,
                    evaluation=evaluation,
                    rc_result=parsed_rc_result,
                    rejection_path=decision_path,
                    report_path=report_path,
                    texture_path=texture_path,
                    fallback_reason=fallback_reason,
                )
            )
        rc.update(
            {
                "status": "evaluated",
                "result": evaluation,
            }
        )
        if evaluation.get("passed_hard_gates") is not True:
            rejection_path = _candidate_artifact_path(
                output_path.parent,
                "original_quality_gate_rejected.json",
                candidate_identifier,
            )
            _atomic_json_write(
                rejection_path,
                {
                    "schema_version": 1,
                    "candidate": pending_output.name,
                    "method": method,
                    "views_used": list(views_used),
                    "source_strategy": source_strategy,
                    "structural": {
                        "color_payload": detected_payload,
                        "valid": True,
                    },
                    "rc": rc,
                    "promotion": {
                        "eligible": False,
                        "mode": "rc-hard-gate",
                    },
                },
            )
            if candidate_identifier is not None:
                raise _OriginalCandidateRejected(
                    _RejectedOriginalCandidate(
                        identifier=candidate_identifier,
                        candidate_path=pending_output,
                        color_payload=detected_payload,
                        method=method,
                        views_used=views_used,
                        source_strategy=source_strategy,
                        seconds=seconds,
                        evaluation=evaluation,
                        rc_result=parsed_rc_result,
                        rejection_path=rejection_path,
                        report_path=report_path,
                        texture_path=texture_path,
                        fallback_reason=fallback_reason,
                    )
                )
            raise OriginalVariantError(
                "Original candidate was rejected by the face/hair RC hard gates."
            )
        promotion_mode = "rc-hard-gate"

    quality_gate = {
        "schema_version": 1,
        "candidate": pending_output.name,
        "method": method,
        "views_used": list(views_used),
        "source_strategy": source_strategy,
        "structural": {
            "color_payload": detected_payload,
            "valid": True,
        },
        "rc": rc,
        "promotion": {
            "eligible": True,
            "mode": promotion_mode,
        },
    }
    gate_path = output_path.parent / "original_quality_gate.json"
    pending_gate = gate_path.with_name(gate_path.name + ".pending")
    pending_gate.write_text(
        json.dumps(quality_gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if preserve_candidate:
        canonical_pending = output_path.with_name(
            output_path.name + ".pending"
        )
        shutil.copy2(pending_output, canonical_pending)
        os.replace(canonical_pending, output_path)
    else:
        os.replace(pending_output, output_path)
    os.replace(pending_gate, gate_path)
    return gate_path, quality_gate


def _promote_best_available_candidate(
    candidates: Sequence[_EvaluatedOriginalCandidate],
    output_path: Path,
    *,
    stage_callback: Callable[[str, str], None] | None,
) -> OriginalVariantResult:
    """Publish the strongest RC-scored candidate exactly once."""

    identifiers = [candidate.identifier for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise OriginalVariantError(
            'RC candidate identifiers must be unique before publication.'
        )

    rankable = {
        candidate.identifier: candidate.rc_result
        for candidate in candidates
        if candidate.rc_result is not None
    }
    if not rankable:
        raise OriginalVariantError(
            "No evaluated Original candidate contained complete RC metrics "
            "for deterministic best-of selection."
        )
    ranked_identifiers = rank_face_hair_candidates(rankable)
    by_identifier = {
        candidate.identifier: candidate for candidate in candidates
    }
    selected = by_identifier[ranked_identifiers[0]]
    selected_path = selected.candidate_path.resolve()
    if (
        not selected_path.is_file()
        or selected_path.parent != output_path.parent.resolve(strict=True)
    ):
        raise OriginalVariantError(
            'Selected RC candidate is outside the generation folder.'
        )
    selected_payload = glb_color_payload(selected_path)
    if selected_payload != selected.color_payload:
        raise OriginalVariantError(
            'Selected RC candidate changed its color payload before publication.'
        )
    selected_passed = bool(
        selected.rc_result is not None
        and selected.rc_result.passed_hard_gates
    )
    if selected.rc_result is None:
        raise OriginalVariantError(
            "Selected Original candidate is missing its RC result."
        )

    if stage_callback:
        stage_callback(
            "scoring_original",
            "Selecting the clearest RC-scored Original candidate",
        )

    ranked_entries: list[dict[str, object]] = []
    for rank, identifier in enumerate(ranked_identifiers, start=1):
        candidate = by_identifier[identifier]
        assert candidate.rc_result is not None
        ranked_entries.append(
            {
                "rank": rank,
                "identifier": identifier,
                "candidate": candidate.candidate_path.name,
                "method": candidate.method,
                "color_payload": candidate.color_payload,
                "views_used": list(candidate.views_used),
                "source_strategy": candidate.source_strategy,
                "passed_hard_gates": False,
                "hard_gate_failures": list(
                    candidate.rc_result.hard_gate_failures
                ),
                "score": candidate.rc_result.score,
                "raw_score": candidate.rc_result.raw_score,
                "worst_quartile_score": (
                    candidate.rc_result.worst_quartile_score
                ),
                "mean_view_score": candidate.rc_result.mean_view_score,
                "rejection_report": candidate.rejection_path.name,
                "evaluation": dict(candidate.evaluation),
            }
        )
        ranked_entries[-1]['passed_hard_gates'] = (
            candidate.rc_result.passed_hard_gates
        )
        ranked_entries[-1]['quality_gate_report'] = (
            candidate.rejection_path.name
        )
    unrankable_entries = [
        {
            "rank": None,
            "identifier": candidate.identifier,
            "candidate": candidate.candidate_path.name,
            "method": candidate.method,
            "passed_hard_gates": False,
            "hard_gate_failures": list(
                candidate.evaluation.get("hard_gate_failures", [])
            ),
            "rejection_report": candidate.rejection_path.name,
            "evaluation": dict(candidate.evaluation),
            "ranking_status": "incomplete_rc_metrics",
        }
        for candidate in candidates
        if candidate.rc_result is None
    ]
    quality_gate: dict[str, object] = {
        "schema_version": 2,
        "candidate": selected.candidate_path.name,
        "method": selected.method,
        "views_used": list(selected.views_used),
        "source_strategy": selected.source_strategy,
        "structural": {
            "color_payload": selected.color_payload,
            "valid": True,
        },
        "rc": {
            "metric": "face-hair-reprojection-region-consistency",
            "required": True,
            "status": "evaluated",
            "passed_hard_gates": False,
            "hard_gate_failures": list(
                selected.rc_result.hard_gate_failures
            ),
            "result": dict(selected.evaluation),
        },
        "candidate_ranking": ranked_entries + unrankable_entries,
        "promotion": {
            "eligible": False,
            "published": True,
            "passed_hard_gates": False,
            "mode": "rc-best-available-degraded",
            "selected_candidate": selected.identifier,
            "ranking_algorithm": "rank_face_hair_candidates",
            "ranking_priority": [
                "hard_gate_pass",
                "appearance_viability",
                "robust_head_clarity_80pct_worst_quartile_20pct_detail",
                "worst_quartile_view_score",
                "worst_individual_view_score",
                "boundary_detail_score",
                "worst_single_and_quartile_face_hair_leakage",
                "final_score",
                "raw_score",
                "mean_view_score",
                "case_insensitive_identifier",
            ],
        },
    }
    gate_path = output_path.parent / "original_quality_gate.json"
    pending_gate = gate_path.with_name(gate_path.name + ".pending")
    pending_gate.write_text(
        json.dumps(quality_gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    canonical_pending = output_path.with_name(output_path.name + ".pending")
    shutil.copy2(selected_path, canonical_pending)
    rc_gate = quality_gate['rc']
    promotion = quality_gate['promotion']
    assert isinstance(rc_gate, dict)
    assert isinstance(promotion, dict)
    rc_gate['passed_hard_gates'] = selected_passed
    promotion.update(
        {
            'eligible': selected_passed,
            'passed_hard_gates': selected_passed,
            'degraded': not selected_passed,
            'mode': (
                'rc-best-scored'
                if selected_passed
                else 'rc-best-available-degraded'
            ),
        }
    )
    pending_gate.write_text(
        json.dumps(quality_gate, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    os.replace(canonical_pending, output_path)
    os.replace(pending_gate, gate_path)
    return OriginalVariantResult(
        output_path=output_path,
        color_payload=selected.color_payload,
        method=selected.method,
        views_used=selected.views_used,
        source_strategy=selected.source_strategy,
        seconds=sum(candidate.seconds for candidate in candidates),
        report_path=selected.report_path,
        texture_path=selected.texture_path,
        fallback_reason=selected.fallback_reason,
        quality_gate_path=gate_path,
        quality_gate=quality_gate,
    )


def _glb_json(path: Path) -> dict[str, Any] | None:
    """Read only the JSON chunk of a GLB without loading its geometry."""

    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                return None
            magic, version, total_length = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or total_length < 20:
                return None
            consumed = 12
            while consumed + 8 <= total_length:
                chunk_header = stream.read(8)
                if len(chunk_header) != 8:
                    return None
                chunk_length, chunk_type = struct.unpack("<II", chunk_header)
                consumed += 8
                if chunk_length < 0 or consumed + chunk_length > total_length:
                    return None
                if chunk_type == _GLB_JSON_CHUNK:
                    if chunk_length > _MAX_GLB_JSON_BYTES:
                        return None
                    payload = stream.read(chunk_length)
                    return json.loads(payload.rstrip(b"\x00 \t\r\n").decode("utf-8"))
                stream.seek(chunk_length, os.SEEK_CUR)
                consumed += chunk_length
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return None
    return None


def glb_color_payload(path: str | os.PathLike[str]) -> str | None:
    """Return ``texture`` or ``vertex-color`` for a genuinely colored GLB."""

    document = _glb_json(Path(path))
    if not document:
        return None
    images = document.get("images")
    textures = document.get("textures")
    if isinstance(images, list) and images and isinstance(textures, list) and textures:
        return "texture"
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        return None
    for mesh in meshes:
        if not isinstance(mesh, Mapping):
            continue
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            continue
        for primitive in primitives:
            if not isinstance(primitive, Mapping):
                continue
            attributes = primitive.get("attributes")
            if isinstance(attributes, Mapping) and "COLOR_0" in attributes:
                return "vertex-color"
    return None


def resolve_blender_executable(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve Blender from an explicit path, environment, PATH, or standard install."""

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


def _normalized_images(
    images: Mapping[str, Image.Image],
) -> dict[str, Image.Image]:
    normalized: dict[str, Image.Image] = {}
    for name in SUPPORTED_VIEW_ANGLES:
        image = images.get(name)
        if not isinstance(image, Image.Image):
            continue
        rgba = image.convert("RGBA")
        if rgba.width < 2 or rgba.height < 2:
            continue
        if rgba.getchannel("A").getbbox() is None:
            continue
        normalized[name] = rgba
    if "front" not in normalized:
        raise OriginalVariantError("A valid front image is required for coloring.")
    return normalized


def _save_cardinal_sources(
    images: Mapping[str, Image.Image],
    output_folder: Path,
) -> tuple[dict[str, Path], str]:
    available = all(name in images for name in CARDINAL_VIEW_KEYS)
    source_strategy = "cardinal-four" if available else "replicated-front"
    paths: dict[str, Path] = {}
    for name in CARDINAL_VIEW_KEYS:
        source = images[name] if available else images["front"]
        destination = output_folder / f"texture_source_{name}.png"
        source.save(destination, format="PNG", optimize=True)
        paths[name] = destination
    return paths, source_strategy


def _save_ten_view_sources(
    images: Mapping[str, Image.Image],
    output_folder: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in TEN_VIEW_KEYS:
        destination = output_folder / f"texture_source_{name}.png"
        images[name].save(destination, format="PNG", optimize=True)
        paths[name] = destination
    return paths


def _command_error(error: BaseException) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return f"timeout after {error.timeout}s"
    if isinstance(error, subprocess.CalledProcessError):
        stderr = (error.stderr or "").strip()
        stdout = (error.stdout or "").strip()
        detail = stderr or stdout
        if detail:
            detail = detail[-1200:].replace("\x00", "")
            return f"exit code {error.returncode}: {detail}"
        return f"exit code {error.returncode}"
    return str(error)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )


def _bake_ten_view_with_blender(
    mesh_path: Path,
    images: Mapping[str, Image.Image],
    output_folder: Path,
    *,
    blender_path: str | os.PathLike[str] | None,
    texture_size: int,
    timeout: int,
    stage_callback: Callable[[str, str], None] | None,
    quality_evaluator: CandidateQualityEvaluator | None,
    candidate_identifier: str | None = None,
    preserve_candidate: bool = False,
) -> OriginalVariantResult:
    started = time.perf_counter()
    repository_root = Path(__file__).resolve().parents[2]
    blender_script = (
        repository_root
        / "hy3dshape"
        / "scripts"
        / "bake_ten_view_texture_blender.py"
    )
    blender = resolve_blender_executable(blender_path)
    source_paths = _save_ten_view_sources(images, output_folder)
    pending_output = _candidate_pending_path(
        output_folder,
        candidate_identifier or "ten-view",
    )
    output_path = output_folder / "textured_mesh.glb"
    texture_path = _candidate_artifact_path(
        output_folder,
        "textured_mesh_atlas.png",
        candidate_identifier,
    )
    report_path = _candidate_artifact_path(
        output_folder,
        "texture_bake_report.json",
        candidate_identifier,
    )

    if stage_callback:
        stage_callback(
            "preparing_texture_inputs",
            "Preparing all ten calibrated images for color projection",
        )
        stage_callback(
            "baking_original",
            "Baking strict visibility-aware ten-view color into Original GLB",
        )
    blender_command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(blender_script),
        "--",
        "--mesh",
        str(mesh_path),
    ]
    for name in TEN_VIEW_KEYS:
        blender_command.extend(
            (
                f"--{name.replace('_', '-')}",
                str(source_paths[name]),
            )
        )
    blender_command.extend(
        (
            "--output",
            str(pending_output),
            "--texture-output",
            str(texture_path),
            "--report",
            str(report_path),
            "--texture-size",
            str(texture_size),
            "--bake-margin",
            "8",
            "--device",
            "GPU",
        )
    )
    with _BLENDER_BAKE_LOCK:
        _run_command(blender_command, cwd=repository_root, timeout=timeout)
    payload = glb_color_payload(pending_output)
    if payload != "texture" or not report_path.is_file():
        raise OriginalVariantError(
            "Ten-view Blender bake did not produce a textured GLB and report."
        )
    quality_gate_path, quality_gate = _promote_original_candidate(
        pending_output,
        output_path,
        color_payload=payload,
        method="strict-visibility-ten-view-consensus",
        views_used=TEN_VIEW_KEYS,
        source_strategy="native-ten-view",
        images=images,
        quality_evaluator=quality_evaluator,
        stage_callback=stage_callback,
        candidate_identifier=candidate_identifier,
        preserve_candidate=preserve_candidate,
        seconds=time.perf_counter() - started,
        report_path=report_path,
        texture_path=texture_path,
    )
    return OriginalVariantResult(
        output_path=output_path,
        color_payload=payload,
        method="strict-visibility-ten-view-consensus",
        views_used=TEN_VIEW_KEYS,
        source_strategy="native-ten-view",
        seconds=time.perf_counter() - started,
        report_path=report_path,
        texture_path=texture_path,
        quality_gate_path=quality_gate_path,
        quality_gate=quality_gate,
    )


def _bake_with_blender(
    mesh_path: Path,
    images: Mapping[str, Image.Image],
    output_folder: Path,
    *,
    blender_path: str | os.PathLike[str] | None,
    texture_size: int,
    atlas_tile_size: int,
    timeout: int,
    quality_evaluator: CandidateQualityEvaluator | None,
    stage_callback: Callable[[str, str], None] | None,
    candidate_identifier: str | None = None,
    preserve_candidate: bool = False,
) -> OriginalVariantResult:
    started = time.perf_counter()
    repository_root = Path(__file__).resolve().parents[2]
    prepare_script = repository_root / "hy3dshape" / "scripts" / "prepare_multiview_atlas.py"
    blender_script = repository_root / "hy3dshape" / "scripts" / "bake_visibility_texture_blender.py"
    blender = resolve_blender_executable(blender_path)
    source_paths, source_strategy = _save_cardinal_sources(images, output_folder)
    atlas_path = output_folder / "texture_sources_atlas.png"
    metadata_path = output_folder / "texture_sources_atlas.json"
    pending_output = _candidate_pending_path(
        output_folder,
        candidate_identifier,
    )
    output_path = output_folder / "textured_mesh.glb"
    texture_path = _candidate_artifact_path(
        output_folder,
        "textured_mesh_atlas.png",
        candidate_identifier,
    )
    report_path = _candidate_artifact_path(
        output_folder,
        "texture_bake_report.json",
        candidate_identifier,
    )

    if stage_callback:
        stage_callback(
            "preparing_texture_inputs",
            "Preparing four cardinal images for color projection",
        )
    prepare_command = [
        sys.executable,
        str(prepare_script),
        "--front",
        str(source_paths["front"]),
        "--left",
        str(source_paths["left"]),
        "--back",
        str(source_paths["back"]),
        "--right",
        str(source_paths["right"]),
        "--output",
        str(atlas_path),
        "--metadata",
        str(metadata_path),
        "--tile-size",
        str(atlas_tile_size),
    ]
    _run_command(prepare_command, cwd=repository_root, timeout=timeout)
    if not atlas_path.is_file() or not metadata_path.is_file():
        raise OriginalVariantError("Texture atlas preparation did not produce its outputs.")

    if stage_callback:
        stage_callback(
            "baking_original",
            "Baking visibility-aware color into Original GLB",
        )
    blender_command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(blender_script),
        "--",
        "--mesh",
        str(mesh_path),
        "--atlas",
        str(atlas_path),
        "--metadata",
        str(metadata_path),
        "--output",
        str(pending_output),
        "--texture-output",
        str(texture_path),
        "--report",
        str(report_path),
        "--texture-size",
        str(texture_size),
        "--bake-margin",
        "8",
        "--device",
        "GPU",
        "--skip-hair-guard",
        "--skip-semantic-guard",
        "--skip-arm-palette-repair",
    ]
    with _BLENDER_BAKE_LOCK:
        _run_command(blender_command, cwd=repository_root, timeout=timeout)
    payload = glb_color_payload(pending_output)
    if payload != "texture" or not report_path.is_file():
        raise OriginalVariantError(
            "Blender finished without producing a textured GLB and bake report."
        )
    quality_gate_path, quality_gate = _promote_original_candidate(
        pending_output,
        output_path,
        color_payload=payload,
        method="visibility-aware-cardinal-bake",
        views_used=tuple(CARDINAL_VIEW_KEYS),
        source_strategy=source_strategy,
        images=images,
        candidate_identifier=candidate_identifier,
        preserve_candidate=preserve_candidate,
        seconds=time.perf_counter() - started,
        report_path=report_path,
        texture_path=texture_path,
        quality_evaluator=quality_evaluator,
        stage_callback=stage_callback,
    )
    return OriginalVariantResult(
        output_path=output_path,
        color_payload=payload,
        method="visibility-aware-cardinal-bake",
        views_used=tuple(CARDINAL_VIEW_KEYS),
        source_strategy=source_strategy,
        seconds=time.perf_counter() - started,
        report_path=report_path,
        texture_path=texture_path,
        quality_gate_path=quality_gate_path,
        quality_gate=quality_gate,
    )


def _view_frame(name: str, yaw_degrees: float, elevation_degrees: float) -> ViewFrame:
    yaw = math.radians(yaw_degrees)
    elevation = math.radians(elevation_degrees)
    cosine = math.cos(elevation)
    to_camera = np.asarray(
        (
            -math.sin(yaw) * cosine,
            -math.cos(yaw) * cosine,
            math.sin(elevation),
        ),
        dtype=np.float64,
    )
    right = np.asarray((math.cos(yaw), -math.sin(yaw), 0.0), dtype=np.float64)
    up = np.cross(to_camera, right)
    return ViewFrame(
        name=name,
        right=tuple(right),
        up=tuple(up),
        to_camera=tuple(to_camera),
    )


def _as_blender_coordinates(values: np.ndarray) -> np.ndarray:
    """Map glTF/trimesh XYZ (Y-up) to Blender XYZ (Z-up)."""

    result = np.empty_like(values, dtype=np.float64)
    result[:, 0] = values[:, 0]
    result[:, 1] = -values[:, 2]
    result[:, 2] = values[:, 1]
    return result


def _vertex_color_fallback(
    mesh_path: Path,
    images: Mapping[str, Image.Image],
    output_folder: Path,
    *,
    fallback_reason: str | None,
    stage_callback: Callable[[str, str], None] | None,
    quality_evaluator: CandidateQualityEvaluator | None,
    candidate_identifier: str | None = None,
    preserve_candidate: bool = False,
    method: str = "multi-view-vertex-color-fallback",
    source_strategy: str | None = None,
    stage: str = "coloring_original_fallback",
    stage_message: str = "Using multi-view vertex colors as the Original fallback",
    view_order: Sequence[str] | None = None,
) -> OriginalVariantResult:
    started = time.perf_counter()
    if stage_callback:
        stage_callback(
            stage,
            stage_message,
        )
    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices):
        raise OriginalVariantError("White GLB does not contain a colorable triangle mesh.")
    vertices = _as_blender_coordinates(np.asarray(loaded.vertices, dtype=np.float64))
    normals = _as_blender_coordinates(np.asarray(loaded.vertex_normals, dtype=np.float64))

    frames: dict[str, ViewFrame] = {}
    arrays: dict[str, np.ndarray] = {}
    provisional = {}
    fit_modes: dict[str, str] = {}
    for name, image in images.items():
        angles = SUPPORTED_VIEW_ANGLES.get(name)
        if angles is None:
            continue
        frame = _view_frame(name, *angles)
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[..., 3]
        fit_mode = (
            "contain"
            if name in SIX_VIEW_AUXILIARY_KEYS
            else "height"
        )
        frames[name] = frame
        arrays[name] = rgba
        fit_modes[name] = fit_mode
        provisional[name] = fit_orthographic_from_alpha(
            alpha,
            vertices,
            frame,
            fit_mode=fit_mode,
        )
    if not provisional:
        raise OriginalVariantError("No calibrated image view is available for coloring.")
    horizontal_scales = [
        calibration.pixels_per_unit_v
        for name, calibration in provisional.items()
        if name not in SIX_VIEW_AUXILIARY_KEYS
    ]
    polar_scales = [
        calibration.pixels_per_unit_v
        for name, calibration in provisional.items()
        if name in SIX_VIEW_AUXILIARY_KEYS
    ]
    shared_horizontal_scale = float(np.median(horizontal_scales))
    shared_polar_scale = (
        float(np.median(polar_scales))
        if polar_scales
        else shared_horizontal_scale
    )

    sampled_colors: list[np.ndarray] = []
    sampled_weights: list[np.ndarray] = []
    views_used: list[str] = []
    projection_order = tuple(view_order or SUPPORTED_VIEW_ANGLES)
    for name in projection_order:
        rgba = arrays.get(name)
        if rgba is None:
            continue
        shared_scale = (
            shared_polar_scale
            if name in SIX_VIEW_AUXILIARY_KEYS
            else shared_horizontal_scale
        )
        calibration = fit_orthographic_from_alpha(
            rgba[..., 3],
            vertices,
            frames[name],
            fit_mode=fit_modes[name],
            pixels_per_unit=shared_scale,
        )
        silhouette = alpha_confidence_map(
            rgba[..., 3],
            feather_pixels=3.0,
        )
        confidence = compute_view_confidence(
            vertices,
            normals,
            calibration,
            silhouette,
        )
        rgb = rgba[..., :3].astype(np.float64) / 255.0
        sampled_colors.append(
            bilinear_sample(rgb, confidence.projection.pixels, outside_value=0.0)
        )
        sampled_weights.append(confidence.combined)
        views_used.append(name)

    blended, _, valid = blend_view_colors(
        np.stack(sampled_colors),
        np.stack(sampled_weights),
        weight_exponent=2.0,
    )
    diffused, filled, diffusion_stats = diffuse_surface_colors(
        blended,
        valid,
        np.asarray(loaded.edges_unique, dtype=np.int64),
        normals,
        minimum_normal_dot=0.25,
        max_iterations=48,
    )
    resolved = valid | filled
    if np.any(resolved):
        neutral = np.median(diffused[resolved], axis=0)
    else:
        neutral = np.full(3, 0.7, dtype=np.float64)
    diffused[~resolved] = neutral
    rgba_colors = np.column_stack(
        (
            np.clip(np.rint(diffused * 255.0), 0, 255).astype(np.uint8),
            np.full(len(diffused), 255, dtype=np.uint8),
        )
    )
    colored = loaded.copy()
    colored.visual = ColorVisuals(
        mesh=colored,
        vertex_colors=rgba_colors,
    )
    output_path = output_folder / "textured_mesh.glb"
    pending_output = _candidate_pending_path(
        output_folder,
        candidate_identifier,
    )
    colored.export(pending_output, file_type="glb", include_normals=True)
    payload = glb_color_payload(pending_output)
    if payload != "vertex-color":
        raise OriginalVariantError("Vertex-color fallback did not produce COLOR_0.")
    resolved_source_strategy = source_strategy or (
        "all-supplied-views" if len(views_used) > 1 else "front-only"
    )
    report_path = _candidate_artifact_path(
        output_folder,
        "vertex_color_report.json",
        candidate_identifier,
    )
    report_path.write_text(
        json.dumps(
            {
                "source_mesh": str(mesh_path.resolve()),
                "output_glb": str(
                    (
                        pending_output
                        if candidate_identifier is not None
                        else output_path
                    ).resolve()
                ),
                "views_used": views_used,
                "method": method,
                "source_strategy": resolved_source_strategy,
                "fallback_reason": fallback_reason,
                "diffusion": asdict(diffusion_stats),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    quality_gate_path, quality_gate = _promote_original_candidate(
        pending_output,
        output_path,
        color_payload=payload,
        method=method,
        views_used=tuple(views_used),
        source_strategy=resolved_source_strategy,
        images=images,
        quality_evaluator=quality_evaluator,
        stage_callback=stage_callback,
        candidate_identifier=candidate_identifier,
        preserve_candidate=preserve_candidate,
        seconds=time.perf_counter() - started,
        report_path=report_path,
        fallback_reason=fallback_reason,
    )
    return OriginalVariantResult(
        output_path=output_path,
        color_payload=payload,
        method=method,
        views_used=tuple(views_used),
        source_strategy=resolved_source_strategy,
        seconds=time.perf_counter() - started,
        report_path=report_path,
        fallback_reason=fallback_reason,
        quality_gate_path=quality_gate_path,
        quality_gate=quality_gate,
    )


def _project_six_view_colors(
    mesh_path: Path,
    images: Mapping[str, Image.Image],
    output_folder: Path,
    *,
    stage_callback: Callable[[str, str], None] | None,
) -> OriginalVariantResult:
    """Publish a GLB whose colors are projected from all six real cameras."""

    selected = {
        name: images[name]
        for name in SIX_VIEW_KEYS
        if name in images
    }
    missing = [name for name in SIX_VIEW_KEYS if name not in selected]
    if missing:
        raise OriginalVariantError(
            "Six-view color projection is missing: " + ", ".join(missing)
        )
    result = _vertex_color_fallback(
        mesh_path,
        selected,
        output_folder,
        fallback_reason=None,
        stage_callback=stage_callback,
        quality_evaluator=None,
        method="six-view-orthographic-vertex-projection",
        source_strategy="native-four-shape-six-view-color",
        stage="baking_original",
        stage_message=(
            "Projecting Front, Back, Left, Right, Top and Bottom colors "
            "onto the generated mesh"
        ),
        view_order=SIX_VIEW_KEYS,
    )
    if result.views_used != SIX_VIEW_KEYS:
        raise OriginalVariantError(
            "Six-view projection did not consume every calibrated camera."
        )
    return result


def create_original_variant(
    mesh_path: str | os.PathLike[str],
    images: Mapping[str, Image.Image],
    output_folder: str | os.PathLike[str],
    *,
    blender_path: str | os.PathLike[str] | None = None,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    atlas_tile_size: int = DEFAULT_ATLAS_TILE_SIZE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    stage_callback: Callable[[str, str], None] | None = None,
    prefer_ten_view: bool = False,
    prefer_six_view: bool = False,
    quality_evaluator: CandidateQualityEvaluator | None = None,
) -> OriginalVariantResult:
    """Create ``textured_mesh.glb`` and guarantee it contains real color data."""

    source_mesh = Path(mesh_path).resolve(strict=True)
    folder = Path(output_folder).resolve(strict=True)
    if not source_mesh.is_file() or source_mesh.parent != folder:
        raise OriginalVariantError("White mesh must be a direct file in its generation folder.")
    if source_mesh.suffix.lower() != ".glb":
        raise OriginalVariantError("Original coloring requires a GLB source mesh.")
    if texture_size < 256 or atlas_tile_size < 128 or timeout <= 0:
        raise ValueError("Texture sizes and timeout must be positive production values.")
    normalized = _normalized_images(images)

    bake_errors: list[str] = []
    six_view_available = all(name in normalized for name in SIX_VIEW_KEYS)
    ten_view_available = all(name in normalized for name in TEN_VIEW_KEYS)
    if ten_view_available and quality_evaluator is None:
        quality_evaluator = _default_ten_view_quality_evaluator(
            blender_path=blender_path,
            timeout=timeout,
        )
    rc_best_of = ten_view_available
    evaluated_candidates: list[_EvaluatedOriginalCandidate] = []

    if prefer_six_view and six_view_available:
        try:
            return _project_six_view_colors(
                source_mesh,
                normalized,
                folder,
                stage_callback=stage_callback,
            )
        except (OSError, OriginalVariantError, ValueError) as error:
            bake_errors.append("six-view: " + _command_error(error))

    if prefer_ten_view and ten_view_available:
        try:
            return _bake_ten_view_with_blender(
                source_mesh,
                normalized,
                folder,
                blender_path=blender_path,
                texture_size=texture_size,
                timeout=timeout,
                stage_callback=stage_callback,
                quality_evaluator=quality_evaluator,
                candidate_identifier=(
                    "ten-view" if rc_best_of else None
                ),
                preserve_candidate=rc_best_of,
            )
        except _OriginalCandidateEvaluated as error:
            evaluated_candidates.append(error.candidate)
            bake_errors.append("ten-view: " + _command_error(error))
        except (
            FileNotFoundError,
            OSError,
            OriginalVariantError,
            subprocess.SubprocessError,
        ) as error:
            bake_errors.append("ten-view: " + _command_error(error))

    if all(name in normalized for name in CARDINAL_VIEW_KEYS):
        try:
            return _bake_with_blender(
                source_mesh,
                normalized,
                folder,
                blender_path=blender_path,
                texture_size=texture_size,
                atlas_tile_size=atlas_tile_size,
                timeout=timeout,
                stage_callback=stage_callback,
                quality_evaluator=quality_evaluator,
                candidate_identifier=(
                    "cardinal" if rc_best_of else None
                ),
                preserve_candidate=rc_best_of,
            )
        except _OriginalCandidateEvaluated as error:
            evaluated_candidates.append(error.candidate)
            bake_errors.append("cardinal: " + _command_error(error))
        except (
            FileNotFoundError,
            OSError,
            OriginalVariantError,
            subprocess.SubprocessError,
        ) as error:
            bake_errors.append("cardinal: " + _command_error(error))
    else:
        missing = [
            name
            for name in CARDINAL_VIEW_KEYS
            if name not in normalized
        ]
        bake_errors.append(
            "cardinal: missing " + ", ".join(missing)
        )

    if not prefer_ten_view and ten_view_available:
        try:
            return _bake_ten_view_with_blender(
                source_mesh,
                normalized,
                folder,
                blender_path=blender_path,
                texture_size=texture_size,
                timeout=timeout,
                stage_callback=stage_callback,
                quality_evaluator=quality_evaluator,
                candidate_identifier=(
                    "ten-view" if rc_best_of else None
                ),
                preserve_candidate=rc_best_of,
            )
        except _OriginalCandidateEvaluated as error:
            evaluated_candidates.append(error.candidate)
            bake_errors.append(
                "ten-view fallback: " + _command_error(error)
            )
        except (
            FileNotFoundError,
            OSError,
            OriginalVariantError,
            subprocess.SubprocessError,
        ) as error:
            bake_errors.append(
                "ten-view fallback: " + _command_error(error)
            )

    reason = "; ".join(bake_errors) or "no Blender texture path was available"
    try:
        return _vertex_color_fallback(
            source_mesh,
            normalized,
            folder,
            fallback_reason=reason,
            stage_callback=stage_callback,
            quality_evaluator=quality_evaluator,
            candidate_identifier=("vertex" if rc_best_of else None),
            preserve_candidate=rc_best_of,
        )
    except _OriginalCandidateEvaluated as error:
        evaluated_candidates.append(error.candidate)
        bake_errors.append("vertex: " + _command_error(error))
    except Exception as fallback_error:
        if not evaluated_candidates:
            raise OriginalVariantError(
                "Could not create a colored Original variant. "
                f"Texture bake: {reason}. "
                f"Vertex-color fallback: {fallback_error}"
            ) from fallback_error
        bake_errors.append("vertex: " + _command_error(fallback_error))

    if rc_best_of and evaluated_candidates:
        return _promote_best_available_candidate(
            evaluated_candidates,
            folder / "textured_mesh.glb",
            stage_callback=stage_callback,
        )
    raise OriginalVariantError(
        "Could not create a colored Original variant. "
        f"Texture bake: {'; '.join(bake_errors)}. "
        "No RC-evaluated candidate was available for publication."
    )


__all__ = [
    "CandidateQualityEvaluator",
    "CARDINAL_VIEW_KEYS",
    "TEN_VIEW_KEYS",
    "DEFAULT_ATLAS_TILE_SIZE",
    "DEFAULT_TEXTURE_SIZE",
    "OriginalVariantError",
    "OriginalVariantResult",
    "SIX_VIEW_KEYS",
    "create_original_variant",
    "glb_color_payload",
    "resolve_blender_executable",
]
