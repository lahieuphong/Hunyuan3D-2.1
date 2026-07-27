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
    vram_label = (
        f"{vram:.2f} GiB VRAM"
        if vram is not None
        else "VRAM chưa xác định"
    )
    backend_label = runtime.backend.upper() if runtime.backend else "LOCAL"
    capability_label = (
        f"CC {runtime.capability}"
        if runtime.capability
        else "CC chưa xác định"
    )
    if matched_profile and match.compatible:
        if matched_profile.verification == "verified":
            detected_copy = f"Tự đề xuất: {matched_profile.label}"
        elif matched_profile.verification == "runtime-verified":
            detected_copy = (
                f"Đã nhận diện: {matched_profile.label}. "
                "Preset đang ở trạng thái ứng viên chờ benchmark."
            )
        else:
            detected_copy = (
                f"Đã nhận diện: {matched_profile.label}. "
                "Profile chưa được bật để áp dụng."
            )
    elif matched_profile:
        detected_copy = (
            f"Cấu hình gần nhất: {matched_profile.label}. "
            "Hãy kiểm tra VRAM và chọn thủ công trước khi áp dụng."
        )
    else:
        detected_copy = "GPU hiện tại chưa có cấu hình đã kiểm chứng trong catalog."
    runtime_verified = (
        matched_profile is not None
        and matched_profile.verification == "runtime-verified"
        and match.compatible
    )
    match_class = (
        "is-runtime-verified"
        if runtime_verified
        else "is-compatible"
        if match.compatible
        else "is-warning"
    )
    match_icon = (
        "info"
        if runtime_verified
        else "check"
        if match.compatible
        else "warning"
    )
    verified_count = sum(
        profile.verification == "verified" for profile in catalog.hardware
    )
    runtime_verified_count = sum(
        profile.verification == "runtime-verified" for profile in catalog.hardware
    )
    return f"""
    <div class="rtx3090-api-intro hardware-catalog-intro">
        <div class="rtx3090-context-tabs">
            <span class="active"><i class="ui-icon-slot" data-ui-icon="memory" aria-hidden="true"></i>{_text(runtime_name)}</span>
            <span>{_text(vram_label)}</span>
            <span>{_text(backend_label)} · {_text(runtime.dtype.upper())}</span>
            <span>{_text(capability_label)}</span>
            <span>1 ảnh</span>
            <span>4 ảnh</span>
        </div>
        <p>
            Catalog có <strong>{len(catalog.hardware)} cấu hình</strong>:
            {verified_count} đã kiểm chứng end-to-end và
            {runtime_verified_count} đã xác nhận runtime.
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
        "runtime-verified": "info",
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
            <strong>{_text(profile.display_name)} · {_text(profile.backend.upper())} · {_text(profile.dtype.upper())} · CC {_text(profile.compute_capability)}</strong>
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
        action_label = "Áp dụng" if preset.verified else "Áp dụng thử"
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
                aria-label="{_text(action_label)} {_text(preset.label)} cho {_text(profile.label)}"
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
    icon = (
        "info"
        if profile.verification in {"verified", "runtime-verified"}
        else "warning"
    )
    return f"""
    <div class="rtx3090-modal-note hardware-profile-note is-{_text(profile.verification)}">
        <span class="rtx3090-note-icon ui-icon-slot" data-ui-icon="{icon}" aria-hidden="true"></span>
        <p>
            <strong>{_text(profile.verification_label)}:</strong>
            {_text(profile.note)}
        </p>
    </div>
    """


def render_legacy_profile_notice(
    hardware_label: str,
    hardware_id: str | None = None,
) -> str:
    identity = hardware_label or hardware_id or "GPU profile cũ"
    return f"""
    <div class="rtx3090-modal-note hardware-profile-note is-experimental">
        <span class="rtx3090-note-icon ui-icon-slot" data-ui-icon="info" aria-hidden="true"></span>
        <p>
            <strong>{_text(identity)}</strong><br>
            Profile đã lưu không còn trong catalog hiện tại. Các thông số generation
            bên dưới vẫn được giữ nguyên để đối chiếu lịch sử.
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
    legacy_hardware_label: str | None = None,
    legacy_hardware_id: str | None = None,
) -> str:
    if preset:
        profile_class = preset.tone
        preset_id = preset.id
        title = preset.label
        icon = "check" if preset.verified else "info"
        displayed_values = preset.parameter_tuple
    else:
        profile_class = "custom"
        preset_id = "custom"
        title = "Cấu hình tùy chỉnh"
        icon = "settings"
        displayed_values = values

    display_hardware_label = profile.short_label
    display_hardware_id = profile.id
    if legacy and legacy_hardware_label:
        display_hardware_label = legacy_hardware_label
        display_hardware_id = legacy_hardware_id or legacy_hardware_label
        current_label = "Profile cũ"
        title = f"{title} · Không còn trong catalog"
    elif legacy:
        current_label = "Bản ghi cũ"
        title = f"{title} · GPU chưa được lưu"
    elif saved:
        current_label = "Đã lưu"
    else:
        current_label = (
            "Đang dùng thử"
            if preset is not None and not preset.verified
            else "Đang dùng"
        )

    normalized = normalize_control_tuple(*displayed_values)
    if normalized:
        display_steps, display_guidance, display_octree, display_chunks = normalized
    else:
        display_steps, display_guidance, display_octree, display_chunks = displayed_values
    return f"""
    <div
        class="rtx-preset-status {profile_class} is-{_text(profile.verification)}"
        data-hardware-id="{_text(display_hardware_id)}"
        data-profile="{_text(preset_id)}"
    >
        <div class="rtx-preset-status-heading">
            <div class="rtx-preset-status-title">
                <span class="rtx-preset-status-check ui-icon-slot" data-ui-icon="{icon}" aria-hidden="true"></span>
                <span>{_text(display_hardware_label)} · 1 ảnh &amp; 4 ảnh · {_text(title)}</span>
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
