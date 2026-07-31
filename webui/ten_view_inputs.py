"""Presentation helpers for the native Gradio ten-view input workflow."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from hy3dshape.ten_view import (
    TEN_VIEW_CONDITIONING_STRATEGY,
    TEN_VIEW_KEYS,
)


@dataclass(frozen=True, slots=True)
class TenViewDefinition:
    key: str
    label: str
    angle: str
    guidance: str


TEN_VIEW_DEFINITIONS: tuple[TenViewDefinition, ...] = (
    TenViewDefinition("front", "Front", "0°", "Straight front"),
    TenViewDefinition("front_right", "Front-right", "45°", "Front ¾ right"),
    TenViewDefinition("right", "Right", "90°", "Character right side"),
    TenViewDefinition("back_right", "Back-right", "135°", "Rear ¾ right"),
    TenViewDefinition("back", "Back", "180°", "Straight back"),
    TenViewDefinition("back_left", "Back-left", "225°", "Rear ¾ left"),
    TenViewDefinition("left", "Left", "270°", "Character left side"),
    TenViewDefinition("front_left", "Front-left", "315°", "Front ¾ left"),
    TenViewDefinition(
        "high_front",
        "High-front",
        "30° down",
        "Hair and shoulders from above",
    ),
    TenViewDefinition(
        "high_back",
        "High-back",
        "30° down",
        "Back hair and shoulders from above",
    ),
)

if tuple(view.key for view in TEN_VIEW_DEFINITIONS) != TEN_VIEW_KEYS:
    raise RuntimeError("Ten-view UI order must match the model adapter contract")


def render_ten_view_guide() -> str:
    return """
        <section class="input-mode-guide ten-view-guide">
            <span class="input-mode-number ui-brand-mark" aria-hidden="true">
                <img class="app-context-logo" src="/favicon.ico" alt="" draggable="false">
            </span>
            <div class="input-mode-copy">
                <strong>Maximum coverage with 10 synchronized views</strong>
                <span>
                    Keep the same character, neutral A-pose, scale, lighting and
                    transparent background in every camera.
                </span>
            </div>
            <span class="ten-view-experimental-badge">Native 4-view shape · 10-view texture/RC</span>
        </section>
    """


def render_ten_view_progress(*images: object) -> str:
    selected = sum(image is not None for image in images[: len(TEN_VIEW_KEYS)])
    total = len(TEN_VIEW_KEYS)
    percentage = round(selected / total * 100)
    state = "is-complete" if selected == total else ""
    status = (
        "Ready to generate"
        if selected == total
        else f"Add {total - selected} remaining view(s)"
    )
    return f"""
        <div class="ten-view-progress {state}" aria-live="polite">
            <span>
                <strong>{selected} / {total}</strong>
                <small>{escape(status)}</small>
            </span>
            <span
                class="ten-view-progress-track"
                role="progressbar"
                aria-label="Ten-view upload completeness"
                aria-valuemin="0"
                aria-valuemax="{total}"
                aria-valuenow="{selected}"
            >
                <i style="width: {percentage}%"></i>
            </span>
        </div>
    """


def render_ten_view_slot_header(
    index: int,
    definition: TenViewDefinition,
) -> str:
    return f"""
        <header class="ten-view-slot-header">
            <span class="ten-view-slot-index">{index}</span>
            <span>
                <strong>{escape(definition.label)}</strong>
                <small>{escape(definition.guidance)}</small>
            </span>
            <em>{escape(definition.angle)}</em>
        </header>
    """


def render_ten_view_summary() -> str:
    return f"""
        <div
            class="ten-view-summary"
            data-conditioning-strategy="{TEN_VIEW_CONDITIONING_STRATEGY}"
        >
            <span class="ui-icon-slot" data-ui-icon="info" aria-hidden="true"></span>
            <span>
                All 10 images are validated and saved. Shape generation uses
                only the checkpoint's native Front, Left, Back and Right
                cameras. Diagonal and high-angle views are reserved for
                texture projection and RC quality scoring, avoiding unsupported
                10-to-4 feature fusion during shape generation.
            </span>
        </div>
    """


__all__ = [
    "TEN_VIEW_DEFINITIONS",
    "TenViewDefinition",
    "render_ten_view_guide",
    "render_ten_view_progress",
    "render_ten_view_slot_header",
    "render_ten_view_summary",
]
