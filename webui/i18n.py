"""Two-locale translation catalog shared by WebUI tooling and tests."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .language_config import DEFAULT_UI_LOCALE, UI_LANGUAGE_FLAGS


SUPPORTED_UI_LOCALES = ("en", "zh-CN")
_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "ui_translations.json"

if set(UI_LANGUAGE_FLAGS) != set(SUPPORTED_UI_LOCALES):
    raise ValueError(
        "UI_LANGUAGE_FLAGS must define exactly: "
        + ", ".join(SUPPORTED_UI_LOCALES)
    )
if DEFAULT_UI_LOCALE not in SUPPORTED_UI_LOCALES:
    raise ValueError(f"Unsupported default UI locale: {DEFAULT_UI_LOCALE}")
if not UI_LANGUAGE_FLAGS[DEFAULT_UI_LOCALE]:
    raise ValueError(
        "Simplified Chinese must remain enabled because it is the default UI language"
    )

ENABLED_UI_LOCALES = tuple(
    locale for locale in SUPPORTED_UI_LOCALES if UI_LANGUAGE_FLAGS[locale]
)
if not ENABLED_UI_LOCALES:
    raise ValueError("At least one WebUI language must be enabled")


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
    """Resolve a browser locale to one of the currently enabled locales."""
    requested = str(locale or "").lower().replace("_", "-")
    if requested.startswith("zh"):
        candidate = "zh-CN"
    elif requested.startswith("en"):
        candidate = "en"
    else:
        candidate = DEFAULT_UI_LOCALE
    return candidate if candidate in ENABLED_UI_LOCALES else DEFAULT_UI_LOCALE


def translate_ui(
    key: str,
    locale: str = DEFAULT_UI_LOCALE,
    **params: object,
) -> str:
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


def ui_language_config_json() -> str:
    """Serialize language flags and the Chinese default for the JS bundle."""
    return json.dumps(
        {
            "defaultLocale": DEFAULT_UI_LOCALE,
            "flags": {
                locale: UI_LANGUAGE_FLAGS[locale]
                for locale in SUPPORTED_UI_LOCALES
            },
        },
        separators=(",", ":"),
    )


__all__ = [
    "DEFAULT_UI_LOCALE",
    "ENABLED_UI_LOCALES",
    "SUPPORTED_UI_LOCALES",
    "normalize_ui_locale",
    "translate_ui",
    "ui_language_config_json",
    "ui_translation_catalog",
    "ui_translation_catalog_json",
]
