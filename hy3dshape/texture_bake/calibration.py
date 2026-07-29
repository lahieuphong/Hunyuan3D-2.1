"""Orthographic camera calibration for four-view texture baking.

The current reference images are already centered alpha cut-outs.  This module
fits a small, serializable orthographic transform between mesh/world
coordinates and those image pixels.  It deliberately has no Blender
dependency, which makes the calibration math usable both in Blender and in
ordinary unit tests.

Coordinate convention
---------------------
``ViewFrame.to_camera`` points from the surface towards the camera.  Projected
depth is ``dot(point, to_camera)``, so a larger value is closer to the camera.
Image-space ``x`` grows to the right and ``y`` grows downwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


ViewName = Literal["front", "left", "back", "right"]
FitMode = Literal["height", "width", "contain", "cover", "anisotropic"]
ScaleStatistic = Literal["median", "mean"]


def _unit_tuple(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite values")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{label} must not be the zero vector")
    vector /= length
    return tuple(float(component) for component in vector)


def _points_array(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError("points must have shape (N, 3) and contain at least one point")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite values")
    return result


@dataclass(frozen=True)
class ViewFrame:
    """An orthonormal frame for one orthographic reference camera."""

    name: str
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    to_camera: tuple[float, float, float]

    def __post_init__(self) -> None:
        right = _unit_tuple(self.right, label="right")
        up = _unit_tuple(self.up, label="up")
        to_camera = _unit_tuple(self.to_camera, label="to_camera")
        if abs(float(np.dot(right, up))) > 1e-6:
            raise ValueError("right and up must be orthogonal")
        handedness = float(np.dot(np.cross(right, up), to_camera))
        if handedness < 1.0 - 1e-6:
            raise ValueError("right x up must point towards to_camera")
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "to_camera", to_camera)

    @property
    def right_array(self) -> np.ndarray:
        return np.asarray(self.right, dtype=np.float64)

    @property
    def up_array(self) -> np.ndarray:
        return np.asarray(self.up, dtype=np.float64)

    @property
    def to_camera_array(self) -> np.ndarray:
        return np.asarray(self.to_camera, dtype=np.float64)

    def plane_coordinates(
        self, points: np.ndarray | Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return horizontal, vertical and near-is-larger depth coordinates."""

        point_array = _points_array(points)
        horizontal = point_array @ self.right_array
        vertical = point_array @ self.up_array
        depth = point_array @ self.to_camera_array
        return horizontal, vertical, depth


# These frames match the Blender orientation used by the existing projection
# script: front camera at -Y, back at +Y, left at +X, and right at -X.
CANONICAL_VIEW_FRAMES: dict[ViewName, ViewFrame] = {
    "front": ViewFrame("front", (1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "left": ViewFrame("left", (0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "back": ViewFrame("back", (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "right": ViewFrame("right", (0, -1, 0), (0, 0, 1), (-1, 0, 0)),
}


def frame_for_view(view: ViewName | str) -> ViewFrame:
    try:
        return CANONICAL_VIEW_FRAMES[view]  # type: ignore[index]
    except KeyError as error:
        names = ", ".join(CANONICAL_VIEW_FRAMES)
        raise ValueError(f"unknown view {view!r}; expected one of {names}") from error


def alpha_to_unit(alpha: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Normalize a 2-D mask or an RGBA alpha channel to float32 in ``[0, 1]``."""

    array = np.asarray(alpha)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[..., 3]
        elif array.shape[2] == 1:
            array = array[..., 0]
        else:
            raise ValueError("a 3-D alpha input must have one or four channels")
    if array.ndim != 2 or not array.size:
        raise ValueError("alpha must be a non-empty 2-D mask or RGBA image")

    if np.issubdtype(array.dtype, np.bool_):
        result = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        result = array.astype(np.float32) / maximum
    else:
        result = array.astype(np.float32)
        finite = np.isfinite(result)
        if not np.all(finite):
            raise ValueError("alpha must contain only finite values")
        minimum = float(result.min())
        maximum = float(result.max())
        if minimum < -1e-6:
            raise ValueError("alpha values must not be negative")
        if maximum > 1.0 + 1e-6:
            if maximum <= 255.0 + 1e-6:
                result /= 255.0
            else:
                raise ValueError(
                    "floating alpha values must use a 0..1 or 0..255 range"
                )
    return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class AlphaBounds:
    """Inclusive pixel-center bounds of the non-transparent silhouette."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5)

    @property
    def extent_x(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def extent_y(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def pixel_width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def pixel_height(self) -> int:
        return self.y_max - self.y_min + 1


def alpha_bounds(alpha: np.ndarray, *, threshold: float = 0.5) -> AlphaBounds:
    """Find inclusive silhouette bounds after applying an alpha threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    unit_alpha = alpha_to_unit(alpha)
    rows, columns = np.nonzero(unit_alpha >= threshold)
    if not len(rows):
        raise ValueError("alpha silhouette is empty")
    return AlphaBounds(
        x_min=int(columns.min()),
        y_min=int(rows.min()),
        x_max=int(columns.max()),
        y_max=int(rows.max()),
    )


@dataclass(frozen=True)
class PlaneBounds:
    u_min: float
    v_min: float
    u_max: float
    v_max: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.u_min + self.u_max) * 0.5, (self.v_min + self.v_max) * 0.5)

    @property
    def extent_u(self) -> float:
        return self.u_max - self.u_min

    @property
    def extent_v(self) -> float:
        return self.v_max - self.v_min


def projected_plane_bounds(
    points: np.ndarray | Sequence[Sequence[float]], frame: ViewFrame
) -> PlaneBounds:
    horizontal, vertical, _ = frame.plane_coordinates(points)
    return PlaneBounds(
        u_min=float(horizontal.min()),
        v_min=float(vertical.min()),
        u_max=float(horizontal.max()),
        v_max=float(vertical.max()),
    )


@dataclass(frozen=True)
class Projection:
    """Projected samples returned by :meth:`OrthographicCalibration.project`."""

    pixels: np.ndarray
    depth: np.ndarray
    inside_image: np.ndarray


@dataclass(frozen=True)
class OrthographicCalibration:
    """Serializable mesh/world-to-image orthographic transform."""

    view: str
    frame: ViewFrame
    image_width: int
    image_height: int
    world_center_u: float
    world_center_v: float
    pixel_center_x: float
    pixel_center_y: float
    pixels_per_unit_u: float
    pixels_per_unit_v: float
    silhouette_bounds: AlphaBounds | None = None

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        scales = (self.pixels_per_unit_u, self.pixels_per_unit_v)
        if not all(np.isfinite(scales)) or min(scales) <= 0:
            raise ValueError("pixels-per-unit scales must be finite and positive")

    def project(self, points: np.ndarray | Sequence[Sequence[float]]) -> Projection:
        horizontal, vertical, depth = self.frame.plane_coordinates(points)
        pixel_x = (
            self.pixel_center_x
            + (horizontal - self.world_center_u) * self.pixels_per_unit_u
        )
        pixel_y = (
            self.pixel_center_y
            - (vertical - self.world_center_v) * self.pixels_per_unit_v
        )
        pixels = np.column_stack((pixel_x, pixel_y))
        inside = (
            (pixel_x >= 0.0)
            & (pixel_x <= self.image_width - 1)
            & (pixel_y >= 0.0)
            & (pixel_y <= self.image_height - 1)
        )
        return Projection(pixels=pixels, depth=depth, inside_image=inside)

    def unproject_plane(self, pixels: np.ndarray) -> np.ndarray:
        """Convert image pixels back to horizontal/vertical plane coordinates."""

        pixel_array = np.asarray(pixels, dtype=np.float64)
        if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
            raise ValueError("pixels must have shape (N, 2)")
        horizontal = (
            self.world_center_u
            + (pixel_array[:, 0] - self.pixel_center_x) / self.pixels_per_unit_u
        )
        vertical = (
            self.world_center_v
            - (pixel_array[:, 1] - self.pixel_center_y) / self.pixels_per_unit_v
        )
        return np.column_stack((horizontal, vertical))

    def normalized_uv(self, pixels: np.ndarray, *, flip_v: bool = True) -> np.ndarray:
        """Convert pixel centers to normalized texture coordinates."""

        pixel_array = np.asarray(pixels, dtype=np.float64)
        if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
            raise ValueError("pixels must have shape (N, 2)")
        width_denominator = max(1, self.image_width - 1)
        height_denominator = max(1, self.image_height - 1)
        u = pixel_array[:, 0] / width_denominator
        image_v = pixel_array[:, 1] / height_denominator
        v = 1.0 - image_v if flip_v else image_v
        return np.column_stack((u, v))

    def to_dict(self) -> dict[str, object]:
        bounds = self.silhouette_bounds
        return {
            "view": self.view,
            "frame": {
                "right": list(self.frame.right),
                "up": list(self.frame.up),
                "to_camera": list(self.frame.to_camera),
            },
            "image_size": [self.image_width, self.image_height],
            "world_center": [self.world_center_u, self.world_center_v],
            "pixel_center": [self.pixel_center_x, self.pixel_center_y],
            "pixels_per_unit": [self.pixels_per_unit_u, self.pixels_per_unit_v],
            "silhouette_bounds": (
                None
                if bounds is None
                else [bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max]
            ),
        }


def _scale_for_mode(
    plane: PlaneBounds, silhouette: AlphaBounds, fit_mode: FitMode
) -> tuple[float, float]:
    if plane.extent_u <= 1e-12 or plane.extent_v <= 1e-12:
        raise ValueError(
            "projected mesh must have non-zero horizontal and vertical extents"
        )
    if silhouette.extent_x <= 0 or silhouette.extent_y <= 0:
        raise ValueError("alpha silhouette must span at least two pixels per axis")

    horizontal_scale = silhouette.extent_x / plane.extent_u
    vertical_scale = silhouette.extent_y / plane.extent_v
    if fit_mode == "height":
        return vertical_scale, vertical_scale
    if fit_mode == "width":
        return horizontal_scale, horizontal_scale
    if fit_mode == "contain":
        scale = min(horizontal_scale, vertical_scale)
        return scale, scale
    if fit_mode == "cover":
        scale = max(horizontal_scale, vertical_scale)
        return scale, scale
    if fit_mode == "anisotropic":
        return horizontal_scale, vertical_scale
    raise ValueError(f"unsupported fit mode {fit_mode!r}")


def fit_orthographic_from_alpha(
    alpha: np.ndarray,
    points: np.ndarray | Sequence[Sequence[float]],
    view: ViewName | str | ViewFrame,
    *,
    threshold: float = 0.5,
    fit_mode: FitMode = "height",
    pixels_per_unit: float | tuple[float, float] | None = None,
) -> OrthographicCalibration:
    """Fit scale and translation from mesh extents to an alpha silhouette.

    ``height`` is the recommended mode for the current full-body references:
    all four views share the same vertical world extent and it avoids stretching
    the narrow side silhouettes to their horizontal bounding boxes.
    """

    frame = view if isinstance(view, ViewFrame) else frame_for_view(view)
    unit_alpha = alpha_to_unit(alpha)
    silhouette = alpha_bounds(unit_alpha, threshold=threshold)
    plane = projected_plane_bounds(points, frame)

    if pixels_per_unit is None:
        scale_u, scale_v = _scale_for_mode(plane, silhouette, fit_mode)
    elif np.isscalar(pixels_per_unit):
        scale_u = scale_v = float(pixels_per_unit)
    else:
        if len(pixels_per_unit) != 2:
            raise ValueError("pixels_per_unit must be a scalar or a pair")
        scale_u, scale_v = (float(value) for value in pixels_per_unit)
        if min(scale_u, scale_v) <= 0 or not np.isfinite((scale_u, scale_v)).all():
            raise ValueError("pixels_per_unit values must be finite and positive")

    world_center_u, world_center_v = plane.center
    pixel_center_x, pixel_center_y = silhouette.center
    return OrthographicCalibration(
        view=frame.name,
        frame=frame,
        image_width=int(unit_alpha.shape[1]),
        image_height=int(unit_alpha.shape[0]),
        world_center_u=world_center_u,
        world_center_v=world_center_v,
        pixel_center_x=pixel_center_x,
        pixel_center_y=pixel_center_y,
        pixels_per_unit_u=scale_u,
        pixels_per_unit_v=scale_v,
        silhouette_bounds=silhouette,
    )


def fit_four_view_calibrations(
    alpha_by_view: Mapping[str, np.ndarray],
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
    fit_mode: FitMode = "height",
    shared_scale: bool = True,
    scale_statistic: ScaleStatistic = "median",
    frames: Mapping[str, ViewFrame] | None = None,
) -> dict[ViewName, OrthographicCalibration]:
    """Fit the canonical front/left/back/right cameras.

    With ``shared_scale=True`` one robust scale (or one pair for anisotropic
    fitting) is shared across all four cameras.  Translation remains per-view
    because each alpha silhouette can be centered a little differently.
    """

    view_order: tuple[ViewName, ...] = ("front", "left", "back", "right")
    missing = [name for name in view_order if name not in alpha_by_view]
    if missing:
        raise ValueError(f"missing alpha silhouettes for: {', '.join(missing)}")
    selected_frames = frames or CANONICAL_VIEW_FRAMES
    missing_frames = [name for name in view_order if name not in selected_frames]
    if missing_frames:
        raise ValueError(f"missing camera frames for: {', '.join(missing_frames)}")

    provisional = {
        name: fit_orthographic_from_alpha(
            alpha_by_view[name],
            points,
            selected_frames[name],
            threshold=threshold,
            fit_mode=fit_mode,
        )
        for name in view_order
    }
    if not shared_scale:
        return provisional

    if scale_statistic == "median":
        aggregate = np.median
    elif scale_statistic == "mean":
        aggregate = np.mean
    else:
        raise ValueError(f"unsupported scale statistic {scale_statistic!r}")

    shared_u = float(
        aggregate(
            [calibration.pixels_per_unit_u for calibration in provisional.values()]
        )
    )
    shared_v = float(
        aggregate(
            [calibration.pixels_per_unit_v for calibration in provisional.values()]
        )
    )
    shared = (shared_u, shared_v)
    return {
        name: fit_orthographic_from_alpha(
            alpha_by_view[name],
            points,
            selected_frames[name],
            threshold=threshold,
            fit_mode=fit_mode,
            pixels_per_unit=shared,
        )
        for name in view_order
    }
