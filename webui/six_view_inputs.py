"""Presentation helpers for the native Gradio six-view input workflow.

This module owns only the six-view camera labels and HTML fragments.  Model
conditioning and texture behavior remain backend concerns, which keeps the
Gradio layout free from duplicated camera-order literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from hy3dshape.six_view import (
    SIX_VIEW_CONDITIONING_STRATEGY,
    SIX_VIEW_KEYS,
)


@dataclass(frozen=True, slots=True)
class SixViewDefinition:
    """A camera slot displayed by the six-view upload grid."""

    key: str
    label: str
    angle: str
    guidance: str


SIX_VIEW_DEFINITIONS: tuple[SixViewDefinition, ...] = (
    SixViewDefinition("front", "Front", "0°", "Straight front"),
    SixViewDefinition("back", "Back", "180°", "Straight back"),
    SixViewDefinition("left", "Left", "270°", "Object left side"),
    SixViewDefinition("right", "Right", "90°", "Object right side"),
    SixViewDefinition("top", "Top", "90° down", "Directly above"),
    SixViewDefinition("bottom", "Bottom", "90° up", "Directly below"),
)

if tuple(definition.key for definition in SIX_VIEW_DEFINITIONS) != SIX_VIEW_KEYS:
    raise RuntimeError("Six-view UI order must match the camera contract")


def render_six_view_guide() -> str:
    """Render concise capture guidance above the six-view grid."""

    return f"""
        <section class="input-mode-guide six-view-guide">
            <span class="input-mode-number ui-brand-mark" aria-hidden="true">
                <img class="app-context-logo" src="/favicon.ico" alt="" draggable="false">
            </span>
            <div class="input-mode-copy">
                <strong>Complete coverage with six synchronized views</strong>
                <span>
                    Keep the same object, scale, lighting and background in
                    Front, Back, Left, Right, Top and Bottom views.
                </span>
            </div>
            <span class="six-view-contract-badge">Native 4-view geometry · 6-view color projection</span>
        </section>
    """


def render_six_view_progress(*images: object) -> str:
    """Render upload completeness from the six native Gradio values."""

    selected = sum(image is not None for image in images[: len(SIX_VIEW_KEYS)])
    total = len(SIX_VIEW_KEYS)
    percentage = round(selected / total * 100)
    state = "is-complete" if selected == total else ""
    status = (
        "Ready to generate"
        if selected == total
        else f"Add {total - selected} remaining view(s)"
    )
    return f"""
        <div class="six-view-progress {state}" aria-live="polite">
            <span>
                <strong>{selected} / {total}</strong>
                <small>{escape(status)}</small>
            </span>
            <span
                class="six-view-progress-track"
                role="progressbar"
                aria-label="Six-view upload completeness"
                aria-valuemin="0"
                aria-valuemax="{total}"
                aria-valuenow="{selected}"
            >
                <i style="width: {percentage}%"></i>
            </span>
        </div>
    """


def render_six_view_slot_header(
    index: int,
    definition: SixViewDefinition,
) -> str:
    """Render the semantic label for one upload component."""

    return f"""
        <header class="six-view-slot-header">
            <span class="six-view-slot-index">{index}</span>
            <span>
                <strong>{escape(definition.label)}</strong>
                <small>{escape(definition.guidance)}</small>
            </span>
            <em>{escape(definition.angle)}</em>
        </header>
    """


def render_six_view_summary() -> str:
    """Disclose the checkpoint's four-camera conditioning contract."""

    return f"""
        <div
            class="six-view-summary"
            data-conditioning-strategy="{escape(SIX_VIEW_CONDITIONING_STRATEGY)}"
        >
            <span class="ui-icon-slot" data-ui-icon="info" aria-hidden="true"></span>
            <span>
                All six images are validated and saved. Geometry uses the
                checkpoint's native Front, Left, Back and Right cameras. All
                six views, including Top and Bottom, are used for orthographic
                color projection.
            </span>
        </div>
    """


__all__ = [
    "SIX_VIEW_DEFINITIONS",
    "SIX_VIEW_KEYS",
    "SixViewDefinition",
    "render_six_view_guide",
    "render_six_view_progress",
    "render_six_view_slot_header",
    "render_six_view_summary",
]
