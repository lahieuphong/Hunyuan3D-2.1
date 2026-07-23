"""Ordered UI fragment manifests.

The tuple order is a runtime contract: CSS relies on the cascade and the
JavaScript fragments share one lexical scope after composition.
"""

from typing import Final


STYLE_FRAGMENTS: Final[tuple[str, ...]] = (
    "styles/00-foundation.css",
    "styles/05-input-tabs-footer.css",
    "styles/10-native-dialogs.css",
    "styles/11-rtx-modal-shell.css",
    "styles/12-rtx-modal-content.css",
    "styles/13-rtx-modal-responsive.css",
    "styles/20-shell-navigation.css",
    "styles/30-input-forms.css",
    "styles/40-workspace-output.css",
    "styles/50-dashboard-theme.css",
    "styles/60-results-console.css",
    "styles/70-left-rail-settings.css",
    "styles/90-responsive.css",
)

SCRIPT_FRAGMENTS: Final[tuple[str, ...]] = (
    "scripts/00-context.js",
    "scripts/10-icons.js",
    "scripts/20-url-state.js",
    "scripts/30-generation-console.js",
    "scripts/40-tab-routing.js",
    "scripts/50-preset-modal.js",
    "scripts/60-shell-wiring.js",
    "scripts/90-bootstrap.js",
)
