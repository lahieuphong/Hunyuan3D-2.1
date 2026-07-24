"""Validated GPU preset catalog and runtime hardware matching.

The JSON catalog is the only place that should contain machine-specific
generation values. This module intentionally has no Gradio or CUDA dependency,
which keeps validation and matching easy to test.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("data") / "gpu_preset_catalog.json"
_VALID_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VALID_VERIFICATION = {"verified", "estimated", "experimental"}
_VALID_BACKENDS = {"cuda", "rocm"}
_VALID_DTYPES = {"float16", "bfloat16", "float32"}
_REQUIRED_PRESET_IDS = frozenset({"safe", "quality"})
_CONTROL_LIMITS = {
    "steps": (1, 100),
    "guidance_scale": (0.000001, 1000),
    "octree_resolution": (16, 512),
    "num_chunks": (1000, 5_000_000),
}


@dataclass(frozen=True)
class GpuPreset:
    id: str
    label: str
    description: str
    tone: str
    verified: bool
    steps: int
    guidance_scale: float
    octree_resolution: int
    num_chunks: int

    @property
    def parameter_tuple(self) -> tuple[int, float, int, int]:
        return (
            self.steps,
            self.guidance_scale,
            self.octree_resolution,
            self.num_chunks,
        )

    def params_snapshot(self) -> dict[str, int | float]:
        return {
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "octree_resolution": self.octree_resolution,
            "num_chunks": self.num_chunks,
        }


@dataclass(frozen=True)
class HardwareProfile:
    id: str
    label: str
    display_name: str
    short_label: str
    vendor: str
    backend: str
    vram_min_gb: float
    vram_max_gb: float | None
    vram_label: str
    aliases: tuple[str, ...]
    examples: tuple[str, ...]
    dtype: str
    verification: str
    verification_label: str
    summary: str
    note: str
    default_preset_id: str
    presets: tuple[GpuPreset, ...]

    def get_preset(self, preset_id: str) -> GpuPreset | None:
        return next((preset for preset in self.presets if preset.id == preset_id), None)

    @property
    def display_vram(self) -> str:
        return self.vram_label


@dataclass(frozen=True)
class GpuPresetCatalog:
    schema_version: int
    default_hardware_id: str | None
    hardware: tuple[HardwareProfile, ...]

    def get_hardware(self, hardware_id: str | None) -> HardwareProfile | None:
        if not isinstance(hardware_id, str):
            return None
        return next((profile for profile in self.hardware if profile.id == hardware_id), None)

    @property
    def preset_count(self) -> int:
        return sum(len(profile.presets) for profile in self.hardware)

    def choices(self) -> list[tuple[str, str]]:
        return [(profile.label, profile.id) for profile in self.hardware]


@dataclass(frozen=True)
class RuntimeHardware:
    requested_device: str
    backend: str
    index: int | None
    name: str
    total_vram_bytes: int | None
    capability: str | None
    bf16_supported: bool | None
    dtype: str
    detected: bool
    error: str | None = None

    @property
    def total_vram_gb(self) -> float | None:
        if self.total_vram_bytes is None:
            return None
        return self.total_vram_bytes / (1024 ** 3)

    @property
    def fingerprint(self) -> str:
        vram = self.total_vram_bytes or 0
        return "|".join(
            (
                self.backend,
                str(self.index if self.index is not None else ""),
                normalize_gpu_name(self.name),
                str(vram),
                self.dtype,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "backend": self.backend,
            "index": self.index,
            "name": self.name,
            "total_vram_bytes": self.total_vram_bytes,
            "capability": self.capability,
            "bf16_supported": self.bf16_supported,
            "dtype": self.dtype,
            "detected": self.detected,
        }


@dataclass(frozen=True)
class HardwareMatch:
    hardware_id: str | None
    method: str
    compatible: bool


def normalize_gpu_name(value: str) -> str:
    """Normalize an exact GPU name without enabling unsafe substring matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", value.upper()).split())


def short_gpu_name(value: str) -> str:
    normalized = " ".join(str(value or "").split())
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if normalized.lower().startswith(prefix.lower()):
            return normalized[len(prefix):]
    return normalized or "GPU Presets"


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_id(value: Any, field: str) -> str:
    value = _require_text(value, field)
    if not _VALID_ID.fullmatch(value):
        raise ValueError(f"{field} must be a stable kebab-case identifier")
    return value


def _require_text_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a string list")
    result = tuple(
        _require_text(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if not allow_empty and not result:
        raise ValueError(f"{field} must contain at least one item")
    return result


def _require_number(
    value: Any,
    field: str,
    *,
    integer: bool,
    minimum: float,
    maximum: float,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    if integer:
        if not numeric.is_integer():
            raise ValueError(f"{field} must be an integer")
        return int(numeric)
    return numeric


def _parse_preset(value: Any, hardware_id: str) -> GpuPreset:
    if not isinstance(value, dict):
        raise ValueError(f"{hardware_id}.presets entries must be objects")
    preset_id = _require_id(value.get("id"), f"{hardware_id}.preset.id")
    params = value.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{hardware_id}.{preset_id}.params must be an object")
    steps = _require_number(
        params.get("steps"),
        f"{hardware_id}.{preset_id}.steps",
        integer=True,
        minimum=_CONTROL_LIMITS["steps"][0],
        maximum=_CONTROL_LIMITS["steps"][1],
    )
    guidance = _require_number(
        params.get("guidance_scale"),
        f"{hardware_id}.{preset_id}.guidance_scale",
        integer=False,
        minimum=_CONTROL_LIMITS["guidance_scale"][0],
        maximum=_CONTROL_LIMITS["guidance_scale"][1],
    )
    octree = _require_number(
        params.get("octree_resolution"),
        f"{hardware_id}.{preset_id}.octree_resolution",
        integer=True,
        minimum=_CONTROL_LIMITS["octree_resolution"][0],
        maximum=_CONTROL_LIMITS["octree_resolution"][1],
    )
    chunks = _require_number(
        params.get("num_chunks"),
        f"{hardware_id}.{preset_id}.num_chunks",
        integer=True,
        minimum=_CONTROL_LIMITS["num_chunks"][0],
        maximum=_CONTROL_LIMITS["num_chunks"][1],
    )
    verified = value.get("verified", False)
    if not isinstance(verified, bool):
        raise ValueError(f"{hardware_id}.{preset_id}.verified must be boolean")
    return GpuPreset(
        id=preset_id,
        label=_require_text(value.get("label"), f"{hardware_id}.{preset_id}.label"),
        description=_require_text(
            value.get("description"),
            f"{hardware_id}.{preset_id}.description",
        ),
        tone=_require_id(value.get("tone", preset_id), f"{hardware_id}.{preset_id}.tone"),
        verified=verified,
        steps=int(steps),
        guidance_scale=float(guidance),
        octree_resolution=int(octree),
        num_chunks=int(chunks),
    )


def _parse_hardware(value: Any) -> HardwareProfile:
    if not isinstance(value, dict):
        raise ValueError("hardware entries must be objects")
    hardware_id = _require_id(value.get("id"), "hardware.id")
    aliases = _require_text_list(
        value.get("aliases", []),
        f"{hardware_id}.aliases",
        allow_empty=True,
    )
    examples = _require_text_list(
        value.get("examples", []),
        f"{hardware_id}.examples",
        allow_empty=False,
    )
    presets = value.get("presets")
    if not isinstance(presets, list) or not presets:
        raise ValueError(f"{hardware_id}.presets must be a non-empty list")

    normalized_aliases = [normalize_gpu_name(alias) for alias in aliases]
    if any(not alias for alias in normalized_aliases):
        raise ValueError(f"{hardware_id}.aliases must contain a GPU name")
    if len(normalized_aliases) != len(set(normalized_aliases)):
        raise ValueError(f"{hardware_id}.aliases contains duplicate exact GPU names")

    parsed_presets = tuple(_parse_preset(item, hardware_id) for item in presets)
    preset_ids = [preset.id for preset in parsed_presets]
    if len(preset_ids) != len(set(preset_ids)):
        raise ValueError(f"{hardware_id} contains duplicate preset IDs")
    preset_id_set = set(preset_ids)
    if preset_id_set != _REQUIRED_PRESET_IDS:
        required = ", ".join(sorted(_REQUIRED_PRESET_IDS))
        raise ValueError(f"{hardware_id}.presets must contain exactly: {required}")
    preset_by_id = {preset.id: preset for preset in parsed_presets}
    for required_id in _REQUIRED_PRESET_IDS:
        if preset_by_id[required_id].tone != required_id:
            raise ValueError(
                f"{hardware_id}.{required_id}.tone must be {required_id!r}"
            )
    tuples = [preset.parameter_tuple for preset in parsed_presets]
    if len(tuples) != len(set(tuples)):
        raise ValueError(f"{hardware_id} contains ambiguous duplicate preset values")

    default_preset_id = _require_id(
        value.get("default_preset_id"),
        f"{hardware_id}.default_preset_id",
    )
    if default_preset_id not in preset_ids:
        raise ValueError(f"{hardware_id}.default_preset_id does not exist")

    vram_min = _require_number(
        value.get("vram_min_gb"),
        f"{hardware_id}.vram_min_gb",
        integer=False,
        minimum=0,
        maximum=1024,
    )
    raw_vram_max = value.get("vram_max_gb")
    vram_max = (
        None
        if raw_vram_max is None
        else float(
            _require_number(
                raw_vram_max,
                f"{hardware_id}.vram_max_gb",
                integer=False,
                minimum=float(vram_min),
                maximum=1024,
            )
        )
    )
    verification = _require_id(
        value.get("verification"),
        f"{hardware_id}.verification",
    )
    if verification not in _VALID_VERIFICATION:
        raise ValueError(
            f"{hardware_id}.verification must be one of {sorted(_VALID_VERIFICATION)}"
        )
    preset_verification = {preset.verified for preset in parsed_presets}
    if verification == "verified" and preset_verification != {True}:
        raise ValueError(
            f"{hardware_id} is verified, so every preset must be verified"
        )
    if verification != "verified" and True in preset_verification:
        raise ValueError(
            f"{hardware_id} is {verification}, so its presets cannot be verified"
        )

    backend = _require_id(value.get("backend"), f"{hardware_id}.backend")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{hardware_id}.backend must be one of {sorted(_VALID_BACKENDS)}"
        )
    dtype = _require_text(value.get("dtype"), f"{hardware_id}.dtype").lower()
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"{hardware_id}.dtype must be one of {sorted(_VALID_DTYPES)}"
        )

    return HardwareProfile(
        id=hardware_id,
        label=_require_text(value.get("label"), f"{hardware_id}.label"),
        display_name=_require_text(
            value.get("display_name"),
            f"{hardware_id}.display_name",
        ),
        short_label=_require_text(
            value.get("short_label"),
            f"{hardware_id}.short_label",
        ),
        vendor=_require_text(value.get("vendor"), f"{hardware_id}.vendor"),
        backend=backend,
        vram_min_gb=float(vram_min),
        vram_max_gb=vram_max,
        vram_label=_require_text(value.get("vram_label"), f"{hardware_id}.vram_label"),
        aliases=aliases,
        examples=examples,
        dtype=dtype,
        verification=verification,
        verification_label=_require_text(
            value.get("verification_label"),
            f"{hardware_id}.verification_label",
        ),
        summary=_require_text(value.get("summary"), f"{hardware_id}.summary"),
        note=_require_text(value.get("note"), f"{hardware_id}.note"),
        default_preset_id=default_preset_id,
        presets=parsed_presets,
    )


def parse_catalog(payload: Any) -> GpuPresetCatalog:
    """Validate and normalize a raw catalog payload."""
    if not isinstance(payload, dict):
        raise ValueError("GPU preset catalog must be an object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("GPU preset catalog schema_version must be 1")
    raw_hardware = payload.get("hardware")
    if not isinstance(raw_hardware, list) or not raw_hardware:
        raise ValueError("GPU preset catalog must contain hardware profiles")
    hardware = tuple(_parse_hardware(value) for value in raw_hardware)
    hardware_ids = [profile.id for profile in hardware]
    if len(hardware_ids) != len(set(hardware_ids)):
        raise ValueError("GPU preset catalog contains duplicate hardware IDs")
    profile_labels = [profile.label.casefold() for profile in hardware]
    if len(profile_labels) != len(set(profile_labels)):
        raise ValueError("GPU preset catalog contains duplicate hardware labels")

    aliases: dict[str, str] = {}
    for profile in hardware:
        for alias in profile.aliases:
            normalized = normalize_gpu_name(alias)
            owner = aliases.get(normalized)
            if owner and owner != profile.id:
                raise ValueError(
                    f"GPU alias {alias!r} is shared by {owner} and {profile.id}"
                )
            aliases[normalized] = profile.id

    for backend in _VALID_BACKENDS:
        generic_profiles = sorted(
            (
                profile
                for profile in hardware
                if profile.backend == backend and not profile.aliases
            ),
            key=lambda profile: profile.vram_min_gb,
        )
        for previous, current in zip(generic_profiles, generic_profiles[1:]):
            if (
                previous.vram_max_gb is None
                or current.vram_min_gb < previous.vram_max_gb
            ):
                raise ValueError(
                    "GPU preset catalog contains overlapping generic VRAM ranges "
                    f"for {previous.id} and {current.id}"
                )

    default_id = payload.get("default_hardware_id")
    if default_id is not None:
        default_id = _require_id(default_id, "default_hardware_id")
        if default_id not in hardware_ids:
            raise ValueError("default_hardware_id does not exist")
    return GpuPresetCatalog(
        schema_version=1,
        default_hardware_id=default_id,
        hardware=hardware,
    )


@lru_cache(maxsize=1)
def load_gpu_preset_catalog() -> GpuPresetCatalog:
    try:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return parse_catalog(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Invalid GPU preset catalog: {error}") from error


def _vram_in_profile(profile: HardwareProfile, total_vram_gb: float) -> bool:
    return (
        total_vram_gb >= profile.vram_min_gb
        and (
            profile.vram_max_gb is None
            or total_vram_gb < profile.vram_max_gb
        )
    )


def _vram_distance(profile: HardwareProfile, total_vram_gb: float) -> float:
    if total_vram_gb < profile.vram_min_gb:
        return profile.vram_min_gb - total_vram_gb
    if profile.vram_max_gb is not None and total_vram_gb >= profile.vram_max_gb:
        return total_vram_gb - profile.vram_max_gb
    return 0.0


def match_runtime_hardware(
    runtime: RuntimeHardware,
    catalog: GpuPresetCatalog | None = None,
) -> HardwareMatch:
    catalog = catalog or load_gpu_preset_catalog()
    total_vram_gb = runtime.total_vram_gb
    if (
        not runtime.detected
        or runtime.backend not in _VALID_BACKENDS
        or total_vram_gb is None
        or not math.isfinite(total_vram_gb)
        or total_vram_gb <= 0
    ):
        return HardwareMatch(None, "unavailable", False)

    normalized_name = normalize_gpu_name(runtime.name)
    for profile in catalog.hardware:
        if (
            profile.backend == runtime.backend
            and normalized_name in {
                normalize_gpu_name(alias) for alias in profile.aliases
            }
            and _vram_in_profile(profile, total_vram_gb)
        ):
            return HardwareMatch(profile.id, "exact", True)

    for profile in catalog.hardware:
        if (
            profile.backend == runtime.backend
            and not profile.aliases
            and _vram_in_profile(profile, total_vram_gb)
        ):
            return HardwareMatch(profile.id, "vram", True)

    generic_profiles = [
        profile
        for profile in catalog.hardware
        if profile.backend == runtime.backend and not profile.aliases
    ]
    if not generic_profiles:
        return HardwareMatch(None, "unavailable", False)
    nearest = min(
        generic_profiles,
        key=lambda profile: _vram_distance(profile, total_vram_gb),
    )
    return HardwareMatch(nearest.id, "nearest", False)


def detect_runtime_hardware(
    torch_module: Any,
    requested_device: str,
    dtype: str,
) -> RuntimeHardware:
    """Read a stable hardware snapshot without shelling out to nvidia-smi."""
    requested = str(requested_device or "unknown")
    try:
        device = torch_module.device(requested)
    except Exception as error:
        return RuntimeHardware(
            requested_device=requested,
            backend="unknown",
            index=None,
            name="Unknown device",
            total_vram_bytes=None,
            capability=None,
            bf16_supported=None,
            dtype=str(dtype),
            detected=False,
            error=str(error),
        )

    if device.type != "cuda":
        return RuntimeHardware(
            requested_device=requested,
            backend=str(device.type),
            index=getattr(device, "index", None),
            name=str(device.type).upper(),
            total_vram_bytes=None,
            capability=None,
            bf16_supported=None,
            dtype=str(dtype),
            detected=True,
        )

    try:
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        index = device.index
        if index is None:
            index = int(torch_module.cuda.current_device())
        properties = torch_module.cuda.get_device_properties(index)
        name = str(torch_module.cuda.get_device_name(index) or properties.name)
        major, minor = torch_module.cuda.get_device_capability(index)
        backend = (
            "rocm"
            if getattr(getattr(torch_module, "version", None), "hip", None)
            else "cuda"
        )
        total_vram_bytes = int(properties.total_memory)
        if total_vram_bytes <= 0:
            raise RuntimeError("GPU reported an invalid total memory value")
        bf16_supported = (
            bool(torch_module.cuda.is_bf16_supported())
            if hasattr(torch_module.cuda, "is_bf16_supported")
            else None
        )
        return RuntimeHardware(
            requested_device=requested,
            backend=backend,
            index=index,
            name=name,
            total_vram_bytes=total_vram_bytes,
            capability=f"{major}.{minor}",
            bf16_supported=bf16_supported,
            dtype=str(dtype),
            detected=True,
        )
    except Exception as error:
        return RuntimeHardware(
            requested_device=requested,
            backend="cuda",
            index=getattr(device, "index", None),
            name="CUDA device unavailable",
            total_vram_bytes=None,
            capability=None,
            bf16_supported=None,
            dtype=str(dtype),
            detected=False,
            error=str(error),
        )


def normalize_control_tuple(
    steps: Any,
    guidance_scale: Any,
    octree_resolution: Any,
    num_chunks: Any,
) -> tuple[int, float, int, int] | None:
    raw_values = (steps, guidance_scale, octree_resolution, num_chunks)
    if any(isinstance(value, bool) for value in raw_values):
        return None
    try:
        numeric_steps = float(steps)
        numeric_guidance = float(guidance_scale)
        numeric_octree = float(octree_resolution)
        numeric_chunks = float(num_chunks)
    except (OverflowError, TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value)
        for value in (
            numeric_steps,
            numeric_guidance,
            numeric_octree,
            numeric_chunks,
        )
    ):
        return None
    if not all(
        value.is_integer()
        for value in (numeric_steps, numeric_octree, numeric_chunks)
    ):
        return None
    ranged_values = {
        "steps": numeric_steps,
        "guidance_scale": numeric_guidance,
        "octree_resolution": numeric_octree,
        "num_chunks": numeric_chunks,
    }
    if any(
        not _CONTROL_LIMITS[field][0] <= value <= _CONTROL_LIMITS[field][1]
        for field, value in ranged_values.items()
    ):
        return None
    return (
        int(numeric_steps),
        numeric_guidance,
        int(numeric_octree),
        int(numeric_chunks),
    )


def resolve_preset_id(
    hardware_id: str | None,
    steps: Any,
    guidance_scale: Any,
    octree_resolution: Any,
    num_chunks: Any,
    catalog: GpuPresetCatalog | None = None,
) -> str | None:
    catalog = catalog or load_gpu_preset_catalog()
    hardware = catalog.get_hardware(hardware_id)
    values = normalize_control_tuple(
        steps,
        guidance_scale,
        octree_resolution,
        num_chunks,
    )
    if not hardware or not values:
        return None
    for preset in hardware.presets:
        if (
            values[0] == preset.steps
            and math.isclose(
                values[1],
                preset.guidance_scale,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and values[2] == preset.octree_resolution
            and values[3] == preset.num_chunks
        ):
            return preset.id
    return None
