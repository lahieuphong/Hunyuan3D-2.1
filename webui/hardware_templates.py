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
    vram_label = f"{vram:.2f} GiB VRAM" if vram is not None else "VRAM chưa xác định"
    backend_label = runtime.backend.upper() if runtime.backend else "LOCAL"
    capability_label = (
        f"CC {runtime.capability}" if runtime.capability else "CC chưa xác định"
    )
    if matched_profile and match.compatible:
        if matched_profile.verification == "verified":
            status_label = "Đã kiểm tra end-to-end"
        elif matched_profile.verification == "runtime-verified":
            status_label = "Ứng viên chờ benchmark"
        else:
            status_label = "Chưa cho phép áp dụng"
    elif matched_profile:
        status_label = "Cấu hình gần nhất · Chỉ đối chiếu"
    else:
        status_label = "Chưa có profile phù hợp"
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
        "info" if runtime_verified else "check" if match.compatible else "warning"
    )
    runtime_meta = (
        f"{vram_label} · {backend_label} {runtime.dtype.upper()} · {capability_label}"
    )
    accessibility_label = (
        f"GPU phát hiện: {runtime.name}. {runtime_meta}. {status_label}."
    )
    return f"""
    <div class="rtx3090-api-intro hardware-catalog-intro">
        <div
            class="hardware-runtime-summary hardware-runtime-strip {match_class}"
            data-runtime-fingerprint="{_text(runtime.fingerprint)}"
            aria-label="{_text(accessibility_label)}"
        >
            <span class="hardware-runtime-icon ui-icon-slot" data-ui-icon="{match_icon}" aria-hidden="true"></span>
            <span class="hardware-runtime-copy">
                <strong>{_text(runtime_name)}</strong>
                <small>{_text(runtime_meta)}</small>
            </span>
            <span class="hardware-runtime-status">
                <i class="ui-icon-slot" data-ui-icon="{match_icon}" aria-hidden="true"></i>
                {_text(status_label)}
            </span>
        </div>
    </div>
    """


def render_hardware_profile_list(
    catalog: GpuPresetCatalog,
    runtime_hardware_id: str | None,
    *,
    displayed_hardware_id: str | None = None,
    saved_hardware_id: str | None = None,
) -> str:
    """Render every catalog profile while keeping runtime authority explicit."""
    runtime_profile = catalog.get_hardware(runtime_hardware_id)
    runtime_label = (
        runtime_profile.short_label if runtime_profile is not None else "GPU hiện tại"
    )
    ordered_profiles = sorted(
        enumerate(catalog.hardware),
        key=lambda item: (
            item[1].id != runtime_hardware_id,
            item[0],
        ),
    )
    cards: list[str] = []
    for _, profile in ordered_profiles:
        is_runtime_current = profile.id == runtime_hardware_id
        is_details_displayed = profile.id == displayed_hardware_id
        is_history_saved = profile.id == saved_hardware_id
        availability_class = (
            "is-runtime-current" if is_runtime_current else "is-disabled"
        )
        details_class = " is-details-displayed" if is_details_displayed else ""
        history_class = " is-history-saved" if is_history_saved else ""
        state_icon = "check" if is_runtime_current else "ban"
        state_label = "Đang dùng" if is_runtime_current else "Không khả dụng"
        if is_runtime_current:
            if is_history_saved:
                availability_copy = (
                    "GPU của bản ghi khớp máy hiện tại. Bản ghi lịch sử đang ở "
                    "chế độ chỉ đọc; preset không thể áp dụng từ màn hình này."
                )
            else:
                availability_copy = (
                    "Cấu hình khớp GPU hiện tại và có thể áp dụng preset."
                )
            state_attribute = 'aria-current="true"'
        else:
            if runtime_profile is None:
                availability_copy = (
                    "Chỉ để đối chiếu · Không tìm thấy profile khớp GPU runtime."
                )
            else:
                availability_copy = (
                    f"Chỉ để đối chiếu · Máy này đang dùng {runtime_label}."
                )
            if is_history_saved:
                availability_copy += " Bản ghi lịch sử này chỉ đọc."
            state_attribute = 'aria-disabled="true"'
        history_accessibility = " Bản ghi đã lưu, chỉ đọc." if is_history_saved else ""
        history_badge = (
            """
            <span class="hardware-catalog-history-badge">
                <i class="ui-icon-slot" data-ui-icon="history" aria-hidden="true"></i>
                Bản ghi đã lưu · Chỉ đọc
            </span>
            """
            if is_history_saved
            else ""
        )
        disabled_overlay = (
            """
            <span class="hardware-catalog-disabled-overlay" aria-hidden="true">
                <i class="ui-icon-slot" data-ui-icon="ban"></i>
                <span>
                    <strong>Không thể chọn</strong>
                    <small>Không khớp GPU hiện tại</small>
                </span>
            </span>
            """
            if not is_runtime_current
            else ""
        )
        availability_html = (
            f'<p class="hardware-catalog-availability">{_text(availability_copy)}</p>'
            if not is_runtime_current or is_history_saved
            else ""
        )
        aria_label = (
            f"{profile.label}. {state_label}. {availability_copy} "
            f"{profile.verification_label}.{history_accessibility}"
        )
        cards.append(
            f"""
            <li
                class="hardware-catalog-card {availability_class} is-{_text(profile.verification)}{details_class}{history_class}"
                data-hardware-id="{_text(profile.id)}"
                data-runtime-current="{str(is_runtime_current).lower()}"
                data-details-displayed="{str(is_details_displayed).lower()}"
                role="listitem"
                {state_attribute}
                aria-label="{_text(aria_label)}"
            >
                <div class="hardware-catalog-card-topline">
                    <span class="hardware-catalog-vram">{_text(profile.vram_label)}</span>
                    <span class="hardware-catalog-state">
                        <i class="ui-icon-slot" data-ui-icon="{state_icon}" aria-hidden="true"></i>
                        {_text(state_label)}
                    </span>
                </div>
                <strong class="hardware-catalog-name">{_text(profile.short_label)}</strong>
                <span class="hardware-catalog-exact-name">{_text(profile.display_name)}</span>
                <div class="hardware-catalog-meta" aria-hidden="true">
                    <span>{_text(profile.backend.upper())}</span>
                    <span>{_text(profile.dtype.upper())}</span>
                    <span>CC {_text(profile.compute_capability)}</span>
                </div>
                <small class="hardware-catalog-evidence">{_text(profile.verification_label)}</small>
                {availability_html}
                {history_badge}
                {disabled_overlay}
            </li>
            """
        )
    return (
        '<ul class="hardware-profile-list" role="list" '
        'aria-label="Danh sách cấu hình GPU trong catalog">' + "".join(cards) + "</ul>"
    )


def render_profile_summary(
    profile: HardwareProfile,
    *,
    recommended_hardware_id: str | None,
    legacy: bool = False,
    legacy_hardware_label: str | None = None,
    legacy_hardware_id: str | None = None,
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
    legacy_identity = legacy_hardware_label or legacy_hardware_id
    if legacy and legacy_identity:
        verification = (
            f"GPU đã lưu: {legacy_identity}. Profile này không còn trong catalog; "
            f"{profile.short_label} đang hiển thị là GPU runtime hiện tại để đối "
            "chiếu, không phải GPU của bản ghi."
        )
    elif legacy:
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


def render_hardware_profile_block(
    catalog: GpuPresetCatalog,
    profile: HardwareProfile,
    *,
    runtime_hardware_id: str | None,
    saved: bool = False,
    legacy: bool = False,
    legacy_hardware_label: str | None = None,
    legacy_hardware_id: str | None = None,
) -> str:
    """Render the runtime catalog list and the profile currently being detailed."""
    saved_hardware_id = profile.id if saved and not legacy else None
    profile_list = render_hardware_profile_list(
        catalog,
        runtime_hardware_id,
        displayed_hardware_id=profile.id,
        saved_hardware_id=saved_hardware_id,
    )
    if not saved and not legacy:
        return profile_list

    legacy_identity = legacy_hardware_label or legacy_hardware_id
    if saved and not legacy:
        detail_label = "Chi tiết cấu hình của bản ghi đã lưu"
    elif legacy and legacy_identity:
        detail_label = (
            f"GPU đã lưu: {legacy_identity} · Không còn trong catalog; "
            f"đang đối chiếu với {profile.short_label}"
        )
    elif legacy:
        detail_label = "Profile runtime dùng để đối chiếu bản ghi chưa lưu GPU"
    else:
        detail_label = "Chi tiết cấu hình đang dùng"
    return (
        profile_list
        + '<div class="hardware-profile-detail-heading">'
        + '<i class="ui-icon-slot" data-ui-icon="info" aria-hidden="true"></i>'
        + f"<span>{_text(detail_label)}</span></div>"
        + render_profile_summary(
            profile,
            recommended_hardware_id=runtime_hardware_id,
            legacy=legacy,
            legacy_hardware_label=legacy_hardware_label,
            legacy_hardware_id=legacy_hardware_id,
        )
    )


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
        f'data-hardware-id="{_text(profile.id)}">' + "".join(cards) + "</div>"
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
        display_steps, display_guidance, display_octree, display_chunks = (
            displayed_values
        )
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
