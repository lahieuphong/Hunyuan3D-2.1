"""Two-locale translation catalog shared by WebUI tooling and tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


SUPPORTED_UI_LOCALES = ("en", "zh-CN")
_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "ui_translations.json"


@lru_cache(maxsize=1)
def ui_translation_catalog() -> dict[str, dict[str, str]]:
    """Load and validate the application-owned translation catalog."""
    raw: Any = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("The UI translation catalog must be a non-empty object")

    catalog: dict[str, dict[str, str]] = {}
    for key, translations in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Every UI translation key must be a non-empty string")
        if not isinstance(translations, dict):
            raise ValueError(f"Translation entry {key!r} must be an object")
        missing = set(SUPPORTED_UI_LOCALES).difference(translations)
        extra = set(translations).difference(SUPPORTED_UI_LOCALES)
        if missing or extra:
            raise ValueError(
                f"Translation entry {key!r} has invalid locales; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        values = {locale: translations[locale] for locale in SUPPORTED_UI_LOCALES}
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError(f"Translation entry {key!r} contains an empty value")
        catalog[key] = values
    return catalog


def normalize_ui_locale(locale: str | None) -> str:
    """Reduce browser locale values to the two locales supported by this app."""
    return "zh-CN" if str(locale or "").lower().startswith("zh") else "en"


def translate_ui(key: str, locale: str = "en", **params: object) -> str:
    """Translate one catalog key and interpolate named parameters."""
    normalized_locale = normalize_ui_locale(locale)
    try:
        template = ui_translation_catalog()[key][normalized_locale]
    except KeyError as exc:
        raise KeyError(f"Unknown UI translation key: {key}") from exc
    return template.format_map({name: str(value) for name, value in params.items()})


def ui_translation_catalog_json() -> str:
    """Serialize the validated catalog for injection into the composed JS bundle."""
    serialized = json.dumps(
        ui_translation_catalog(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


__all__ = [
    "SUPPORTED_UI_LOCALES",
    "normalize_ui_locale",
    "translate_ui",
    "ui_translation_catalog",
    "ui_translation_catalog_json",
]
