"""UI-only ten-view input panel.

The native file inputs rendered here deliberately have no backend binding.
They let the ten-view workflow be designed and reviewed without extending the
existing single-view or four-view generation contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True, slots=True)
class TenViewDefinition:
    """One camera slot in the future ten-view generation workflow."""

    key: str
    label: str
    angle: str
    guidance: str


TEN_VIEW_DEFINITIONS: tuple[TenViewDefinition, ...] = (
    TenViewDefinition("front", "Front", "0°", "Straight front"),
    TenViewDefinition(
        "front_right",
        "Front-right",
        "45°",
        "Camera on character right",
    ),
    TenViewDefinition("right", "Right", "90°", "Character right side"),
    TenViewDefinition(
        "back_right",
        "Back-right",
        "135°",
        "Rear three-quarter right",
    ),
    TenViewDefinition("back", "Back", "180°", "Straight back"),
    TenViewDefinition(
        "back_left",
        "Back-left",
        "225°",
        "Rear three-quarter left",
    ),
    TenViewDefinition("left", "Left", "270°", "Character left side"),
    TenViewDefinition(
        "front_left",
        "Front-left",
        "315°",
        "Front three-quarter left",
    ),
    TenViewDefinition(
        "high_front",
        "High-front",
        "30° down",
        "Elevated front for hair and shoulders",
    ),
    TenViewDefinition(
        "high_back",
        "High-back",
        "30° down",
        "Elevated back for hair and shoulders",
    ),
)


def _render_upload_card(index: int, view: TenViewDefinition) -> str:
    input_id = f"ten-view-input-{view.key}"
    safe_label = escape(view.label)
    safe_angle = escape(view.angle)
    safe_guidance = escape(view.guidance)
    return f"""
        <article
            class="ten-view-card"
            data-ten-view-card
            data-view-key="{escape(view.key)}"
        >
            <header class="ten-view-card-header">
                <span class="ten-view-card-index">{index}</span>
                <span class="ten-view-card-title">
                    <strong>{safe_label}</strong>
                    <small>{safe_guidance}</small>
                </span>
                <span class="ten-view-card-angle">{safe_angle}</span>
            </header>
            <div
                class="ten-view-dropzone"
                data-ten-view-dropzone
                role="button"
                tabindex="0"
                aria-controls="{input_id}"
                aria-label="Choose image for {safe_label}"
            >
                <input
                    id="{input_id}"
                    class="ten-view-file-input"
                    data-ten-view-input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    tabindex="-1"
                >
                <img
                    class="ten-view-preview"
                    data-ten-view-preview
                    alt="{safe_label} preview"
                    hidden
                >
                <span class="ten-view-empty" data-ten-view-empty>
                    <span
                        class="ten-view-empty-icon ui-icon-slot"
                        data-ui-icon="box"
                        aria-hidden="true"
                    ></span>
                    <strong>Upload view</strong>
                    <small>PNG, JPG or WebP</small>
                </span>
                <span class="ten-view-error" data-ten-view-error hidden></span>
            </div>
            <button
                class="ten-view-remove"
                data-ten-view-remove
                type="button"
                aria-label="Remove {safe_label} image"
                title="Remove image"
                hidden
            >×</button>
        </article>
    """


def render_ten_view_panel() -> str:
    """Render the isolated ten-view upload experience."""

    cards = "".join(
        _render_upload_card(index, view)
        for index, view in enumerate(TEN_VIEW_DEFINITIONS, start=1)
    )
    total = len(TEN_VIEW_DEFINITIONS)
    return f"""
        <section
            class="ten-view-panel"
            data-ten-view-panel
            data-ui-only="true"
            aria-labelledby="ten-view-title"
        >
            <div class="input-mode-guide ten-view-guide">
                <span
                    class="input-mode-number ui-brand-mark"
                    aria-hidden="true"
                >
                    <img
                        class="app-context-logo"
                        src="/favicon.ico"
                        alt=""
                        draggable="false"
                    >
                </span>
                <div class="input-mode-copy">
                    <strong id="ten-view-title">Maximum coverage with 10 views</strong>
                    <span>
                        Upload one synchronized A-pose from every required angle.
                    </span>
                </div>
                <span class="ten-view-ui-badge">UI preview</span>
            </div>

            <div class="ten-view-progress" aria-live="polite">
                <span>
                    <strong data-ten-view-count>0 / {total}</strong>
                    <small>views selected locally</small>
                </span>
                <span
                    class="ten-view-progress-track"
                    role="progressbar"
                    aria-label="Ten-view upload completeness"
                    aria-valuemin="0"
                    aria-valuemax="{total}"
                    aria-valuenow="0"
                    data-ten-view-progress
                >
                    <i data-ten-view-progress-fill></i>
                </span>
            </div>

            <div class="ten-view-upload-grid">
                {cards}
            </div>

            <div class="ten-view-summary">
                <span
                    class="ten-view-summary-icon ui-icon-slot"
                    data-ui-icon="info"
                    aria-hidden="true"
                ></span>
                <span>
                    Files stay in this browser preview. The current 1-view and
                    4-view generation pipelines are unchanged.
                </span>
            </div>

            <button
                class="ten-view-generate-placeholder"
                type="button"
                disabled
                aria-disabled="true"
            >
                <span
                    class="ten-view-generate-icon ui-icon-slot"
                    data-ui-icon="wand"
                    aria-hidden="true"
                ></span>
                <span>
                    <strong>Generate 3D</strong>
                    <small>10-view pipeline will be connected next</small>
                </span>
            </button>
        </section>
    """


__all__ = [
    "TEN_VIEW_DEFINITIONS",
    "TenViewDefinition",
    "render_ten_view_panel",
]
