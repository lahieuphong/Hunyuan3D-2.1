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
    vram_label = f"{vram:.2f} GiB VRAM" if vram is not None else "VRAM unavailable"
    backend_label = runtime.backend.upper() if runtime.backend else "LOCAL"
    capability_label = (
        f"CC {runtime.capability}" if runtime.capability else "CC unavailable"
    )
    if matched_profile and match.compatible:
        if matched_profile.verification == "verified":
            status_label = "End-to-end verified"
        elif matched_profile.verification == "runtime-verified":
            status_label = "Pending generation benchmark"
        else:
            status_label = "Not available for use"
    elif matched_profile:
        status_label = "Closest profile · Reference only"
    else:
        status_label = "No compatible profile"
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
        f"Detected GPU: {runtime.name}. {runtime_meta}. {status_label}."
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
        runtime_profile.short_label if runtime_profile is not None else "current GPU"
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
        state_label = "Active" if is_runtime_current else "Unavailable"
        if is_runtime_current:
            if is_history_saved:
                availability_copy = (
                    "The saved record's GPU matches the current machine. This "
                    "historical record is read-only; presets cannot be applied "
                    "from this screen."
                )
            else:
                availability_copy = (
                    "This profile matches the current GPU and its presets are available."
                )
            state_attribute = 'aria-current="true"'
        else:
            if runtime_profile is None:
                availability_copy = (
                    "Reference only · No profile matches the runtime GPU."
                )
            else:
                availability_copy = (
                    f"Reference only · This machine is using {runtime_label}."
                )
            if is_history_saved:
                availability_copy += " This historical record is read-only."
            state_attribute = 'aria-disabled="true"'
        history_accessibility = " Saved record, read-only." if is_history_saved else ""
        history_badge = (
            """
            <span class="hardware-catalog-history-badge">
                <i class="ui-icon-slot" data-ui-icon="history" aria-hidden="true"></i>
                Saved record · Read-only
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
                    <strong>Cannot select</strong>
                    <small>Does not match the current GPU</small>
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
        'aria-label="GPU profiles in the catalog">' + "".join(cards) + "</ul>"
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
            f"Saved GPU: {legacy_identity}. This profile is no longer in the catalog; "
            f"{profile.short_label} is shown as the current runtime GPU for "
            "comparison, not as the GPU from the saved record."
        )
    elif legacy:
        verification = (
            "The legacy record did not store a GPU; the displayed profile is "
            "provided only to compare the saved settings."
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
        detail_label = "Saved record profile details"
    elif legacy and legacy_identity:
        detail_label = (
            f"Saved GPU: {legacy_identity} · No longer in the catalog; "
            f"compared with {profile.short_label}"
        )
    elif legacy:
        detail_label = "Runtime profile for a record with no saved GPU"
    else:
        detail_label = "Active profile details"
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
        action_label = "Apply" if preset.verified else "Try"
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
    identity = hardware_label or hardware_id or "Legacy GPU profile"
    return f"""
    <div class="rtx3090-modal-note hardware-profile-note is-experimental">
        <span class="rtx3090-note-icon ui-icon-slot" data-ui-icon="info" aria-hidden="true"></span>
        <p>
            <strong>{_text(identity)}</strong><br>
            This saved profile is no longer in the current catalog. The generation
            settings below are preserved for historical reference.
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
        title = "Custom configuration"
        icon = "settings"
        displayed_values = values

    display_hardware_label = profile.short_label
    display_hardware_id = profile.id
    if legacy and legacy_hardware_label:
        display_hardware_label = legacy_hardware_label
        display_hardware_id = legacy_hardware_id or legacy_hardware_label
        current_label = "Legacy profile"
        title = f"{title} · No longer in the catalog"
    elif legacy:
        current_label = "Legacy record"
        title = f"{title} · GPU was not saved"
    elif saved:
        current_label = "Saved"
    else:
        current_label = (
            "In use for testing"
            if preset is not None and not preset.verified
            else "In use"
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
                <span>{_text(display_hardware_label)} · 1 image &amp; 4 images · {_text(title)}</span>
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
