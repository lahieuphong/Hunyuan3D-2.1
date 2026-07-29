"""Expose repository-local extensions alongside the core shape package.

The upstream implementation lives in ``hy3dshape/hy3dshape`` while repository
tools such as ``texture_bake`` live beside it. Extending this package path
keeps both locations importable when the repository root is on ``sys.path``.
Core public exports stay lazy so Blender-only tools can import the lightweight
texture helpers without loading the inference stack.
"""

from importlib import import_module
from pathlib import Path
from typing import Final


_CORE_PACKAGE_PATH = Path(__file__).resolve().parent / "hy3dshape"
if _CORE_PACKAGE_PATH.is_dir():
    __path__.append(str(_CORE_PACKAGE_PATH))

_CORE_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "Hunyuan3DDiTPipeline": ("pipelines", "Hunyuan3DDiTPipeline"),
    "Hunyuan3DDiTFlowMatchingPipeline": (
        "pipelines",
        "Hunyuan3DDiTFlowMatchingPipeline",
    ),
    "ImageProcessorV2": ("preprocessors", "ImageProcessorV2"),
    "IMAGE_PROCESSORS": ("preprocessors", "IMAGE_PROCESSORS"),
    "DEFAULT_IMAGEPROCESSOR": ("preprocessors", "DEFAULT_IMAGEPROCESSOR"),
}
_POSTPROCESSOR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FaceReducer",
        "FloaterRemover",
        "DegenerateFaceRemover",
        "MeshSimplifier",
    }
)


def __getattr__(name: str):
    """Resolve the core package API without eagerly importing model modules."""

    target = _CORE_EXPORTS.get(name)
    module_name = target[0] if target is not None else "postprocessors"
    attribute_name = target[1] if target is not None else name
    if target is None and name not in _POSTPROCESSOR_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as error:
        if name in _POSTPROCESSOR_NAMES and error.name == "pymeshlab":
            raise ModuleNotFoundError(
                "Mesh postprocessors require the optional 'pymeshlab' package. "
                "It is not required for Shape DiT LoRA training."
            ) from error
        raise

    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_CORE_EXPORTS, *_POSTPROCESSOR_NAMES})


__all__ = list(_CORE_EXPORTS)
