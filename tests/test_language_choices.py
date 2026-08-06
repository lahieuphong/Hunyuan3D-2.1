from __future__ import annotations

import json
from pathlib import Path
import unittest

from webui import load_ui_assets
from webui.asset_manifest import SCRIPT_FRAGMENTS
from webui.i18n import (
    DEFAULT_UI_LOCALE,
    ENABLED_UI_LOCALES,
    ui_language_config_json,
)
from webui.language_config import (
    ENABLE_ENGLISH,
    ENABLE_SIMPLIFIED_CHINESE,
    UI_LANGUAGE_FLAGS,
)


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

    def test_language_flags_are_injected_and_chinese_is_default(self):
        self.assertIsInstance(ENABLE_ENGLISH, bool)
        self.assertTrue(ENABLE_SIMPLIFIED_CHINESE)
        self.assertEqual(DEFAULT_UI_LOCALE, "zh-CN")
        self.assertEqual(
            ENABLED_UI_LOCALES,
            tuple(
                locale
                for locale in ("en", "zh-CN")
                if UI_LANGUAGE_FLAGS[locale]
            ),
        )
        self.assertEqual(
            json.loads(ui_language_config_json()),
            {
                "defaultLocale": "zh-CN",
                "flags": {
                    "en": ENABLE_ENGLISH,
                    "zh-CN": ENABLE_SIMPLIFIED_CHINESE,
                },
            },
        )

        _, javascript = load_ui_assets()
        self.assertNotIn("/*__UI_LANGUAGE_CONFIG__*/", javascript)
        self.assertIn(
            'document.documentElement.lang = storedUiLocale ?? uiDefaultLocale;',
            javascript,
        )
        self.assertIn(
            "document.documentElement.lang !== locale",
            javascript,
        )

    def test_disabled_languages_are_filtered_from_the_visible_selector(self):
        choice_javascript = (
            REPOSITORY_ROOT / "webui/assets/scripts/28-language-choices.js"
        ).read_text(encoding="utf-8")
        locale_javascript = (
            REPOSITORY_ROOT / "webui/assets/scripts/05-i18n.js"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "(language) => uiSupportedLocales.includes(language.code)",
            choice_javascript,
        )
        self.assertIn(
            "return uiSupportedLocales.includes(candidate)",
            locale_javascript,
        )
        self.assertIn(
            "The default WebUI language must remain enabled",
            locale_javascript,
        )

    def test_bootstrap_installs_the_restricted_control(self):
        css, javascript = load_ui_assets()

        self.assertIn("installRestrictedLanguageChoices();", javascript)
        self.assertIn(".ui-language-select-control", css)
        self.assertIn(".ui-language-native-control", css)


if __name__ == "__main__":
    unittest.main()
