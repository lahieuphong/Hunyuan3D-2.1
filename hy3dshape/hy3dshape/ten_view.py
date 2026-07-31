"""Shared camera contract for the ten-view workflow.

The released multi-view shape checkpoint was trained with exactly four camera
embeddings: front, left, back and right.  The production ten-view workflow
therefore validates and persists all ten sources, but conditions *shape* with
only those four native cardinal cameras.  Diagonal and elevated cameras remain
available to texture baking and reprojection-consistency (RC) evaluation.

The feature-fusion matrix below is retained for explicit experiments and unit
tests.  It is not the default UI shape-conditioning strategy.
"""

from __future__ import annotations

from typing import Final


CANONICAL_VIEW_KEYS: Final[tuple[str, ...]] = (
    "front",
    "left",
    "back",
    "right",
)

TEN_VIEW_KEYS: Final[tuple[str, ...]] = (
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
    "high_front",
    "high_back",
)

TEN_VIEW_AUXILIARY_KEYS: Final[tuple[str, ...]] = tuple(
    key for key in TEN_VIEW_KEYS if key not in CANONICAL_VIEW_KEYS
)

# Experimental feature-fusion contract. Rows follow TEN_VIEW_KEYS; columns
# follow CANONICAL_VIEW_KEYS. Diagonal cameras contribute equally to their two
# neighbouring cardinal cameras. Elevated cameras reinforce the corresponding
# front/back slot because the checkpoint has no elevation-specific embedding.
TEN_VIEW_BLEND_WEIGHTS: Final[tuple[tuple[float, ...], ...]] = (
    (1.0, 0.0, 0.0, 0.0),  # front
    (0.5, 0.0, 0.0, 0.5),  # front_right
    (0.0, 0.0, 0.0, 1.0),  # right
    (0.0, 0.0, 0.5, 0.5),  # back_right
    (0.0, 0.0, 1.0, 0.0),  # back
    (0.0, 0.5, 0.5, 0.0),  # back_left
    (0.0, 1.0, 0.0, 0.0),  # left
    (0.5, 0.5, 0.0, 0.0),  # front_left
    (1.0, 0.0, 0.0, 0.0),  # high_front
    (0.0, 0.0, 1.0, 0.0),  # high_back
)

TEN_VIEW_CONDITIONING_STRATEGY: Final[str] = (
    "native-cardinal-shape-with-ten-view-texture-rc"
)
TEN_VIEW_FEATURE_FUSION_STRATEGY: Final[str] = (
    "experimental-feature-fusion-10-to-4"
)


def normalized_ten_view_blend_weights() -> tuple[tuple[float, ...], ...]:
    """Return weights normalized across source views for every output slot."""

    column_totals = tuple(
        sum(row[column] for row in TEN_VIEW_BLEND_WEIGHTS)
        for column in range(len(CANONICAL_VIEW_KEYS))
    )
    if any(total <= 0 for total in column_totals):
        raise RuntimeError("Every canonical camera slot needs a positive weight")
    return tuple(
        tuple(value / column_totals[column] for column, value in enumerate(row))
        for row in TEN_VIEW_BLEND_WEIGHTS
    )


__all__ = [
    "CANONICAL_VIEW_KEYS",
    "TEN_VIEW_AUXILIARY_KEYS",
    "TEN_VIEW_BLEND_WEIGHTS",
    "TEN_VIEW_CONDITIONING_STRATEGY",
    "TEN_VIEW_FEATURE_FUSION_STRATEGY",
    "TEN_VIEW_KEYS",
    "normalized_ten_view_blend_weights",
]
