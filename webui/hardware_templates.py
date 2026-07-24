"""HTML renderers for the hardware preset catalog."""

from __future__ import annotations

from html import escape
from typing import Any

from .gpu_presets import (
    GpuPreset,
    GpuPresetCatalog,
    HardwareMatch,
    HardwareProfile,
    RuntimeHardware,
    normalize_control_tuple,
    short_gpu_name,
)


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _format_number(value: Any) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _text(value)


def render_catalog_intro(
    runtime: RuntimeHardware,
    match: HardwareMatch,
    catalog: GpuPresetCatalog,
) -> str:
    matched_profile = catalog.get_hardware(match.hardware_id)
    runtime_name = short_gpu_name(runtime.name)
    vram = runtime.total_vram_gb
    vram_label = f"{vram:.0f} GB VRAM" if vram is not None else "VRAM chưa xác định"
    backend_label = runtime.backend.upper() if runtime.backend else "LOCAL"
    if matched_profile and match.compatible:
        detected_copy = f"Tự đề xuất: {matched_profile.label}"
    elif matched_profile:
        detected_copy = (
            f"Cấu hình gần nhất: {matched_profile.label}. "
            "Hãy kiểm tra VRAM và chọn thủ công trước khi áp dụng."
        )
    else:
        detected_copy = "Không tìm thấy cấu hình khớp; hãy chọn thủ công."
    match_class = "is-compatible" if match.compatible else "is-warning"
    match_icon = "check" if match.compatible else "warning"
    return f"""
    <div class="rtx3090-api-intro hardware-catalog-intro">
        <div class="rtx3090-context-tabs">
            <span class="active"><i class="ui-icon-slot" data-ui-icon="memory" aria-hidden="true"></i>{_text(runtime_name)}</span>
            <span>{_text(vram_label)}</span>
            <span>{_text(backend_label)} · {_text(runtime.dtype.upper())}</span>
            <span>1 ảnh</span>
            <span>4 ảnh</span>
        </div>
        <p>
            Chọn đúng nhóm GPU/VRAM của máy rồi áp dụng một preset.
            Các giá trị được cập nhật trực tiếp vào <strong>Advanced Options</strong>.
        </p>
        <div class="hardware-runtime-strip {match_class}" data-runtime-fingerprint="{_text(runtime.fingerprint)}">
            <span class="hardware-runtime-icon ui-icon-slot" data-ui-icon="{match_icon}" aria-hidden="true"></span>
            <span>
                <strong>GPU phát hiện: {_text(runtime.name)}</strong>
                <small>{_text(detected_copy)}</small>
            </span>
        </div>
    </div>
    """


def render_profile_summary(
    profile: HardwareProfile,
    *,
    recommended_hardware_id: str | None,
    legacy: bool = False,
) -> str:
    selected_matches_runtime = profile.id == recommended_hardware_id
    verification_class = f"is-{profile.verification}"
    match_class = "is-runtime-match" if selected_matches_runtime else "is-manual"
    icon = {
        "verified": "check",
        "estimated": "info",
        "experimental": "warning",
    }[profile.verification]
    examples = " · ".join(profile.examples)
    if legacy:
        verification = (
            "Bản ghi cũ không lưu GPU; profile đang hiển thị chỉ dùng để đối chiếu "
            "các thông số đã lưu."
        )
    else:
        verification = profile.verification_label
    return f"""
    <div
        class="rtx3090-machine-strip hardware-profile-summary {verification_class} {match_class}"
        data-hardware-id="{_text(profile.id)}"
    >
        <div class="rtx3090-machine-badge">{_text(profile.vram_label)}</div>
        <div class="rtx3090-machine-copy">
            <strong>{_text(profile.display_name)} · {_text(profile.backend.upper())} · {_text(profile.dtype.upper())}</strong>
            <span>{_text(profile.summary)}</span>
            <small>{_text(examples)}</small>
        </div>
        <span class="hardware-profile-verification">
            <i class="ui-icon-slot" data-ui-icon="{icon}" aria-hidden="true"></i>
            {_text(verification)}
        </span>
    </div>
    """


def render_preset_cards(
    profile: HardwareProfile,
    selected_preset_id: str | None,
) -> str:
    cards: list[str] = []
    for preset in profile.presets:
        is_selected = preset.id == selected_preset_id
        selected_class = " is-selected" if is_selected else ""
        aria_pressed = "true" if is_selected else "false"
        cards.append(
            f"""
            <article
                class="rtx3090-profile-card {preset.tone}{selected_class}"
                data-profile="{_text(preset.id)}"
                data-hardware-id="{_text(profile.id)}"
                data-mutates-generation-settings="true"
                role="button"
                tabindex="0"
                aria-pressed="{aria_pressed}"
                aria-controls="advanced-settings-form"
                aria-label="Áp dụng {_text(preset.label)} cho {_text(profile.label)}"
            >
                <div class="rtx3090-profile-heading">
                    <h3>{_text(preset.label)}</h3>
                    <span class="rtx3090-profile-selector" aria-hidden="true"></span>
                </div>
                <p>{_text(preset.description)}</p>
                <div class="rtx3090-profile-values">
                    <span><b>{preset.steps}</b><small>Steps</small></span>
                    <span><b>{_format_number(preset.guidance_scale)}</b><small>Guidance</small></span>
                    <span><b>{preset.octree_resolution}</b><small>Octree</small></span>
                    <span><b>{preset.num_chunks}</b><small>Chunks</small></span>
                </div>
            </article>
            """
        )
    return (
        '<div class="rtx3090-profile-grid hardware-preset-grid" '
        f'data-hardware-id="{_text(profile.id)}">'
        + "".join(cards)
        + "</div>"
    )


def render_profile_note(profile: HardwareProfile) -> str:
    icon = "warning" if profile.verification != "verified" else "info"
    return f"""
    <div class="rtx3090-modal-note hardware-profile-note is-{_text(profile.verification)}">
        <span class="rtx3090-note-icon ui-icon-slot" data-ui-icon="{icon}" aria-hidden="true"></span>
        <p>
            <strong>{_text(profile.verification_label)}:</strong>
            {_text(profile.note)}
        </p>
    </div>
    """


def render_preset_status(
    profile: HardwareProfile,
    preset: GpuPreset | None,
    values: tuple[Any, Any, Any, Any],
    *,
    saved: bool = False,
    legacy: bool = False,
) -> str:
    if preset:
        profile_class = preset.tone
        preset_id = preset.id
        title = preset.label
        icon = "check"
        displayed_values = preset.parameter_tuple
    else:
        profile_class = "custom"
        preset_id = "custom"
        title = "Cấu hình tùy chỉnh"
        icon = "settings"
        displayed_values = values

    if legacy:
        current_label = "Bản ghi cũ"
        title = f"{title} · GPU chưa được lưu"
    elif saved:
        current_label = "Đã lưu"
    else:
        current_label = "Đang dùng"

    normalized = normalize_control_tuple(*displayed_values)
    if normalized:
        display_steps, display_guidance, display_octree, display_chunks = normalized
    else:
        display_steps, display_guidance, display_octree, display_chunks = displayed_values
    return f"""
    <div
        class="rtx-preset-status {profile_class}"
        data-hardware-id="{_text(profile.id)}"
        data-profile="{_text(preset_id)}"
    >
        <div class="rtx-preset-status-heading">
            <div class="rtx-preset-status-title">
                <span class="rtx-preset-status-check ui-icon-slot" data-ui-icon="{icon}" aria-hidden="true"></span>
                <span>{_text(profile.short_label)} · 1 ảnh &amp; 4 ảnh · {_text(title)}</span>
            </div>
            <span class="rtx-preset-current">{_text(current_label)}</span>
        </div>
        <div class="rtx-preset-values">
            <span><b>{_format_number(display_steps)}</b><small>Steps</small></span>
            <span><b>{_format_number(display_guidance)}</b><small>Guidance</small></span>
            <span><b>{_format_number(display_octree)}</b><small>Octree</small></span>
            <span><b>{_format_number(display_chunks)}</b><small>Chunks</small></span>
        </div>
    </div>
    """
