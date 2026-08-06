"""Human-editable WebUI language availability settings."""

from typing import Final


# Set either flag to False to remove that language from the Settings selector.
ENABLE_ENGLISH: Final[bool] = False
ENABLE_SIMPLIFIED_CHINESE: Final[bool] = True

# Simplified Chinese is the preferred language for every new browser session.
DEFAULT_UI_LOCALE: Final[str] = "zh-CN"

UI_LANGUAGE_FLAGS: Final[dict[str, bool]] = {
    "en": ENABLE_ENGLISH,
    "zh-CN": ENABLE_SIMPLIFIED_CHINESE,
}


__all__ = [
    "DEFAULT_UI_LOCALE",
    "ENABLE_ENGLISH",
    "ENABLE_SIMPLIFIED_CHINESE",
    "UI_LANGUAGE_FLAGS",
]
