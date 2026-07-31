# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from pathlib import Path

from .pipelines import Hunyuan3DDiTPipeline, Hunyuan3DDiTFlowMatchingPipeline
from .preprocessors import ImageProcessorV2, IMAGE_PROCESSORS, DEFAULT_IMAGEPROCESSOR


# ``gradio_app.py`` adds the outer hy3dshape directory to sys.path for
# upstream compatibility. Also expose repository extensions that live beside
# the core package, such as ``hy3dshape.texture_bake``.
_EXTENSION_PACKAGE_PATH = Path(__file__).resolve().parent.parent
if str(_EXTENSION_PACKAGE_PATH) not in __path__:
    __path__.append(str(_EXTENSION_PACKAGE_PATH))


_POSTPROCESSOR_NAMES = {
    "FaceReducer",
    "FloaterRemover",
    "DegenerateFaceRemover",
    "MeshSimplifier",
}

__all__ = [
    "DEFAULT_IMAGEPROCESSOR",
    "Hunyuan3DDiTFlowMatchingPipeline",
    "Hunyuan3DDiTPipeline",
    "IMAGE_PROCESSORS",
    "ImageProcessorV2",
]


def __getattr__(name):
    """Load optional PyMeshLab postprocessors only when they are requested."""
    if name not in _POSTPROCESSOR_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from . import postprocessors
    except ModuleNotFoundError as error:
        if error.name == "pymeshlab":
            raise ModuleNotFoundError(
                "Mesh postprocessors require the optional 'pymeshlab' package. "
                "It is not required for Shape DiT LoRA training."
            ) from error
        raise

    value = getattr(postprocessors, name)
    globals()[name] = value
    return value
