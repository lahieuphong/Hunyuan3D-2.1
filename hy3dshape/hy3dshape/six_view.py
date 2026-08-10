"""Camera contract for the six-view WebUI workflow.

The released Hunyuan3D-2mv shape checkpoint has four native camera slots:
front, left, back and right.  A six-view request therefore keeps those four
views as the shape-conditioning payload and reserves the two polar cameras for
real orthographic color projection onto the generated mesh.

This distinction is intentional: top and bottom improve surface color
coverage without pretending that the checkpoint was trained with six camera
embeddings.
"""

from __future__ import annotations

from typing import Final

from .ten_view import CANONICAL_VIEW_KEYS


# UI/persistence order follows the user-facing capture sequence.  Shape
# conditioning is reordered separately with CANONICAL_VIEW_KEYS.
SIX_VIEW_KEYS: Final[tuple[str, ...]] = (
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
)

SIX_VIEW_AUXILIARY_KEYS: Final[tuple[str, ...]] = (
    "top",
    "bottom",
)

# Angles use the Blender Z-up convention shared by texture_bake.generation:
# yaw=0 faces the front camera and positive elevation moves above the object.
SIX_VIEW_ANGLES: Final[dict[str, tuple[float, float]]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (270.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}

SIX_VIEW_CONDITIONING_STRATEGY: Final[str] = (
    "native-cardinal-shape-with-six-view-color-projection"
)

if set(CANONICAL_VIEW_KEYS) & set(SIX_VIEW_AUXILIARY_KEYS):
    raise RuntimeError("Polar six-view cameras must not be native shape slots")
if tuple(SIX_VIEW_ANGLES) != SIX_VIEW_KEYS:
    raise RuntimeError("Six-view angle order must match the persistence contract")


__all__ = [
    "SIX_VIEW_ANGLES",
    "SIX_VIEW_AUXILIARY_KEYS",
    "SIX_VIEW_CONDITIONING_STRATEGY",
    "SIX_VIEW_KEYS",
]
