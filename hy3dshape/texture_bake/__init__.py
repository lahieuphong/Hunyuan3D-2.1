"""Visibility-aware four-view texture-baking building blocks.

This package is intentionally independent from the legacy one-view-per-polygon
projector.  It exposes calibration and confidence primitives that can be
integrated into a future UV baking pass without mutating any GLB by itself.
"""

from .blender_visibility import (
    BlenderRaycastVisibility,
    blender_raycast_visibility,
)
from .calibration import (
    AlphaBounds,
    CANONICAL_VIEW_FRAMES,
    OrthographicCalibration,
    PlaneBounds,
    Projection,
    ViewFrame,
    alpha_bounds,
    alpha_to_unit,
    fit_four_view_calibrations,
    fit_orthographic_from_alpha,
    frame_for_view,
    projected_plane_bounds,
)
from .semantics import (
    AnimePaletteThresholds,
    SemanticClass,
    SemanticGuardResult,
    apply_semantic_color_guards,
    classify_anime_colors,
    default_compatibility_matrix,
    rgb_to_hsv,
    rgb_to_unit,
)
from .visibility import (
    DepthBuffer,
    SurfaceColorDiffusionStats,
    ViewConfidence,
    ViewConfidenceConfig,
    alpha_confidence_map,
    angular_confidence,
    bilinear_sample,
    blend_view_colors,
    compute_view_confidence,
    depth_visibility_confidence,
    diffuse_surface_colors,
    nearest_sample,
    normalize_view_weights,
    rasterize_depth_buffer,
    smoothstep01,
)

__all__ = [
    "AlphaBounds",
    "AnimePaletteThresholds",
    "BlenderRaycastVisibility",
    "CANONICAL_VIEW_FRAMES",
    "DepthBuffer",
    "OrthographicCalibration",
    "PlaneBounds",
    "Projection",
    "SemanticClass",
    "SemanticGuardResult",
    "SurfaceColorDiffusionStats",
    "ViewConfidence",
    "ViewConfidenceConfig",
    "ViewFrame",
    "alpha_bounds",
    "alpha_confidence_map",
    "alpha_to_unit",
    "angular_confidence",
    "apply_semantic_color_guards",
    "bilinear_sample",
    "blend_view_colors",
    "blender_raycast_visibility",
    "classify_anime_colors",
    "compute_view_confidence",
    "default_compatibility_matrix",
    "depth_visibility_confidence",
    "diffuse_surface_colors",
    "fit_four_view_calibrations",
    "fit_orthographic_from_alpha",
    "frame_for_view",
    "nearest_sample",
    "normalize_view_weights",
    "projected_plane_bounds",
    "rasterize_depth_buffer",
    "rgb_to_hsv",
    "rgb_to_unit",
    "smoothstep01",
]
