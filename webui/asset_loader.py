"""Compose ordered UI fragments for ``gradio.Blocks``."""

from pathlib import Path

from .asset_manifest import SCRIPT_FRAGMENTS, STYLE_FRAGMENTS
from .i18n import ui_translation_catalog_json

_ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"


def _read_fragment(relative_path: str) -> str:
    """Read one fragment independently of the process working directory."""
    path = _ASSET_DIRECTORY / relative_path
    with path.open("r", encoding="utf-8-sig", newline=None) as fragment:
        content = fragment.read()
    if relative_path == "scripts/05-i18n.js":
        marker = "/*__UI_TRANSLATION_CATALOG__*/"
        if content.count(marker) != 1:
            raise ValueError("The UI translation catalog marker is missing or duplicated")
        content = content.replace(marker, ui_translation_catalog_json())
    return content


def _join_fragments(relative_paths: tuple[str, ...]) -> str:
    """Join fragments verbatim and remove only the bundle's terminal newline."""
    content = "".join(_read_fragment(path) for path in relative_paths)
    return content.removesuffix("\n")


def load_ui_assets() -> tuple[str, str]:
    """Return the ordered CSS and JavaScript bundles consumed by Gradio."""
    return _join_fragments(STYLE_FRAGMENTS), _join_fragments(SCRIPT_FRAGMENTS)
