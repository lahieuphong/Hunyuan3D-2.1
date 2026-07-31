"""Input packing and validation shared by Gradio generation callbacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from PIL import Image

from hy3dshape.ten_view import (
    CANONICAL_VIEW_KEYS,
    TEN_VIEW_AUXILIARY_KEYS,
    TEN_VIEW_CONDITIONING_STRATEGY,
    TEN_VIEW_KEYS,
)


MAX_TEN_VIEW_PIXELS = 40_000_000
TEN_VIEW_MODE_ALIASES = frozenset({"ten", "10-view", "ten-view"})
FOUR_VIEW_MODE_ALIASES = frozenset({"four", "4-view", "multi-view"})
SINGLE_VIEW_MODE_ALIASES = frozenset({"single", "1-view", "single-view"})


class GenerationInputError(ValueError):
    """Raised when a generation mode does not have a valid image set."""


@dataclass(frozen=True, slots=True)
class GenerationInputBundle:
    """Validated images plus truthful model-conditioning metadata."""

    mode: str
    provided_images: dict[str, Image.Image]
    conditioning_images: dict[str, Image.Image]
    primary_image: Image.Image
    metadata: dict[str, object]


def normalize_input_mode(value: object) -> str:
    """Normalize all persisted and URL-compatible mode aliases."""

    normalized = str(value or "single").strip().lower()
    if normalized in TEN_VIEW_MODE_ALIASES:
        return "ten"
    if normalized in FOUR_VIEW_MODE_ALIASES:
        return "four"
    if normalized in SINGLE_VIEW_MODE_ALIASES:
        return "single"
    raise GenerationInputError("Chế độ ảnh không hợp lệ. Hãy tải lại trang Web UI.")


def ordered_ten_view_images(
    values: Sequence[Image.Image | None],
) -> dict[str, Image.Image | None]:
    """Map Gradio's positional ten image values onto semantic camera keys."""

    if len(values) != len(TEN_VIEW_KEYS):
        raise GenerationInputError(
            f"Ten-view callback expected {len(TEN_VIEW_KEYS)} images, "
            f"received {len(values)}."
        )
    return dict(zip(TEN_VIEW_KEYS, values, strict=True))


def _validate_ten_view_images(
    images: Mapping[str, Image.Image | None],
) -> dict[str, Image.Image]:
    missing = [key for key in TEN_VIEW_KEYS if key not in images or images[key] is None]
    if missing:
        raise GenerationInputError(
            "Tab 10 ẢNH cần đủ cả 10 góc. Còn thiếu: "
            + ", ".join(key.replace("_", " ").title() for key in missing)
        )

    unknown = sorted(set(images) - set(TEN_VIEW_KEYS))
    if unknown:
        raise GenerationInputError(
            "Ten-view input contains unsupported camera keys: " + ", ".join(unknown)
        )

    validated: dict[str, Image.Image] = {}
    for key in TEN_VIEW_KEYS:
        image = images[key]
        if not isinstance(image, Image.Image):
            raise GenerationInputError(
                f"Ảnh {key.replace('_', ' ')} không phải định dạng ảnh hợp lệ."
            )
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_TEN_VIEW_PIXELS:
            raise GenerationInputError(
                f"Ảnh {key.replace('_', ' ')} có kích thước không hợp lệ "
                f"({width}×{height})."
            )
        if "A" in image.getbands() and image.getchannel("A").getbbox() is None:
            raise GenerationInputError(
                f"Ảnh {key.replace('_', ' ')} hoàn toàn trong suốt."
            )
        validated[key] = image
    return validated


def build_generation_input_bundle(
    input_mode: object,
    single_image: Image.Image | None,
    four_view_images: Mapping[str, Image.Image | None],
    ten_view_images: Mapping[str, Image.Image | None] | None = None,
) -> GenerationInputBundle:
    """Validate a UI mode without changing the existing 1/4-view behavior."""

    mode = normalize_input_mode(input_mode)
    if mode == "single":
        if single_image is None:
            raise GenerationInputError("Tab 1 ẢNH cần một ảnh chính diện của vật thể.")
        provided = {"front": single_image}
        return GenerationInputBundle(
            mode=mode,
            provided_images=provided,
            conditioning_images=dict(provided),
            primary_image=single_image,
            metadata={
                "views_provided": ["front"],
                "views_used": ["front"],
                "conditioned_view_count": 1,
                "conditioning_strategy": "native-single-view",
            },
        )

    if mode == "four":
        ordered_four = {key: four_view_images.get(key) for key in CANONICAL_VIEW_KEYS}
        missing = [key for key, image in ordered_four.items() if image is None]
        if missing:
            raise GenerationInputError(
                "Tab 4 ẢNH cần đủ Front, Back, Left và Right. Còn thiếu: "
                + ", ".join(key.title() for key in missing)
            )
        provided = {
            key: image
            for key, image in ordered_four.items()
            if isinstance(image, Image.Image)
        }
        # Preserve the legacy callback's permissive non-PIL handling. Gradio's
        # native image component supplies PIL images during normal UI use.
        if len(provided) != len(ordered_four):
            provided = dict(ordered_four)  # type: ignore[assignment]
        return GenerationInputBundle(
            mode=mode,
            provided_images=provided,
            conditioning_images=dict(provided),
            primary_image=provided["front"],
            metadata={
                "views_provided": list(CANONICAL_VIEW_KEYS),
                "views_used": list(CANONICAL_VIEW_KEYS),
                "conditioned_view_count": 4,
                "conditioning_strategy": "native-cardinal-4",
            },
        )

    validated_ten = _validate_ten_view_images(ten_view_images or {})
    cardinal_conditioning = {
        key: validated_ten[key] for key in CANONICAL_VIEW_KEYS
    }
    return GenerationInputBundle(
        mode=mode,
        provided_images=validated_ten,
        conditioning_images=cardinal_conditioning,
        primary_image=validated_ten["front"],
        metadata={
            "views_provided": list(TEN_VIEW_KEYS),
            "views_used": list(CANONICAL_VIEW_KEYS),
            "shape_views_used": list(CANONICAL_VIEW_KEYS),
            "conditioned_view_count": len(CANONICAL_VIEW_KEYS),
            "conditioning_strategy": TEN_VIEW_CONDITIONING_STRATEGY,
            "model_native_view_limit": len(CANONICAL_VIEW_KEYS),
            "texture_rc_views": list(TEN_VIEW_KEYS),
            "auxiliary_views_reserved_for_texture_rc": list(
                TEN_VIEW_AUXILIARY_KEYS
            ),
            "experimental_conditioning": False,
        },
    )


def image_has_opaque_alpha(image: Image.Image) -> bool:
    """Return whether an image has no useful transparent-background signal."""

    if "A" not in image.getbands():
        return True
    return image.getchannel("A").getextrema() == (255, 255)


__all__ = [
    "GenerationInputBundle",
    "GenerationInputError",
    "MAX_TEN_VIEW_PIXELS",
    "build_generation_input_bundle",
    "image_has_opaque_alpha",
    "normalize_input_mode",
    "ordered_ten_view_images",
]
