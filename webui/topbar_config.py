"""Human-editable visibility flags for the application topbar."""

from typing import Final


# Set a flag to False to remove that button from the topbar.
SHOW_API_DOCS_BUTTON: Final[bool] = True
SHOW_HISTORY_BUTTON: Final[bool] = True
SHOW_SETTINGS_BUTTON: Final[bool] = True
SHOW_GPU_BUTTON: Final[bool] = True

TOPBAR_BUTTON_FLAGS: Final[dict[str, bool]] = {
    "api_docs": SHOW_API_DOCS_BUTTON,
    "history": SHOW_HISTORY_BUTTON,
    "settings": SHOW_SETTINGS_BUTTON,
    "gpu": SHOW_GPU_BUTTON,
}


__all__ = [
    "SHOW_API_DOCS_BUTTON",
    "SHOW_GPU_BUTTON",
    "SHOW_HISTORY_BUTTON",
    "SHOW_SETTINGS_BUTTON",
    "TOPBAR_BUTTON_FLAGS",
]
