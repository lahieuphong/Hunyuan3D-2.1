from __future__ import annotations

from pathlib import Path
import unittest

from webui import load_ui_assets
from webui.asset_manifest import SCRIPT_FRAGMENTS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RestrictedLanguageChoiceTests(unittest.TestCase):
    def test_language_controller_is_loaded_before_bootstrap(self):
        fragment = "scripts/28-language-choices.js"
        self.assertIn(fragment, SCRIPT_FRAGMENTS)
        self.assertLess(
            SCRIPT_FRAGMENTS.index(fragment),
            SCRIPT_FRAGMENTS.index("scripts/90-bootstrap.js"),
        )

    def test_only_english_and_simplified_chinese_are_exposed(self):
        javascript = (
            REPOSITORY_ROOT / "webui/assets/scripts/28-language-choices.js"
        ).read_text(encoding="utf-8")

        self.assertIn('{code: "en", label: "English"}', javascript)
        self.assertIn('{code: "zh-CN", label: "简体中文"}', javascript)
        self.assertNotIn("zh-TW", javascript)
        self.assertIn('data-testid="dropdown-option"', javascript)
        self.assertIn('new MouseEvent("mousedown"', javascript)
        self.assertIn('input.setAttribute("aria-hidden", "true")', javascript)
        self.assertIn('input.readOnly = true', javascript)

    def test_bootstrap_installs_the_restricted_control(self):
        css, javascript = load_ui_assets()

        self.assertIn("installRestrictedLanguageChoices();", javascript)
        self.assertIn(".ui-language-select-control", css)
        self.assertIn(".ui-language-native-control", css)


if __name__ == "__main__":
    unittest.main()
