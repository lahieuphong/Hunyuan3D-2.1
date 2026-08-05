from __future__ import annotations

import re
from pathlib import Path
import unittest

from webui import load_ui_assets
from webui.asset_manifest import SCRIPT_FRAGMENTS, STYLE_FRAGMENTS
from webui.i18n import (
    SUPPORTED_UI_LOCALES,
    normalize_ui_locale,
    translate_ui,
    ui_translation_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z0-9_]+)\}")
DIRECT_UI_T_PATTERN = re.compile(r'uiT\(\s*["\']([^"\']+)["\']')
VIETNAMESE_SPECIFIC_PATTERN = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]"
)


class WebUiI18nTests(unittest.TestCase):
    def test_catalog_is_complete_and_placeholder_safe(self):
        catalog = ui_translation_catalog()

        self.assertGreaterEqual(len(catalog), 400)
        self.assertEqual(SUPPORTED_UI_LOCALES, ("en", "zh-CN"))
        for key, translations in catalog.items():
            with self.subTest(key=key):
                self.assertEqual(set(translations), set(SUPPORTED_UI_LOCALES))
                self.assertTrue(translations["en"].strip())
                self.assertTrue(translations["zh-CN"].strip())
                self.assertEqual(
                    set(PLACEHOLDER_PATTERN.findall(translations["en"])),
                    set(PLACEHOLDER_PATTERN.findall(translations["zh-CN"])),
                )

    def test_locale_normalization_and_translation(self):
        self.assertEqual(normalize_ui_locale("en-US"), "en")
        self.assertEqual(normalize_ui_locale("zh-Hans"), "zh-CN")
        self.assertEqual(normalize_ui_locale("zh_CN"), "zh-CN")
        self.assertEqual(normalize_ui_locale("vi"), "en")
        self.assertEqual(
            translate_ui("action.generate_3d", "zh-CN"),
            "生成 3D",
        )
        self.assertEqual(
            translate_ui("history.generation_title", "zh-CN", uid="abc"),
            "生成记录 abc",
        )

    def test_every_direct_javascript_translation_key_exists(self):
        catalog = ui_translation_catalog()
        missing: dict[str, list[str]] = {}
        scripts_directory = REPOSITORY_ROOT / "webui/assets/scripts"

        for path in scripts_directory.glob("*.js"):
            source = path.read_text(encoding="utf-8")
            for key in DIRECT_UI_T_PATTERN.findall(source):
                if key not in catalog:
                    missing.setdefault(key, []).append(path.name)

        self.assertEqual(missing, {})

    def test_catalog_is_injected_before_feature_scripts(self):
        self.assertLess(
            SCRIPT_FRAGMENTS.index("scripts/05-i18n.js"),
            SCRIPT_FRAGMENTS.index("scripts/10-icons.js"),
        )
        self.assertLess(
            STYLE_FRAGMENTS.index("styles/88-i18n.css"),
            STYLE_FRAGMENTS.index("styles/90-responsive.css"),
        )
        _, javascript = load_ui_assets()
        self.assertNotIn("/*__UI_TRANSLATION_CATALOG__*/", javascript)
        self.assertIn('"action.generate_3d"', javascript)
        self.assertIn("installUiLocalization();", javascript)

    def test_active_webui_sources_do_not_contain_vietnamese_copy(self):
        paths = [
            REPOSITORY_ROOT / "gradio_app.py",
            *(REPOSITORY_ROOT / "webui").rglob("*.py"),
            *(REPOSITORY_ROOT / "webui").rglob("*.js"),
            *(REPOSITORY_ROOT / "webui").rglob("*.html"),
            *(REPOSITORY_ROOT / "webui").rglob("*.json"),
            *(REPOSITORY_ROOT / "assets").glob("modelviewer*"),
        ]
        offenders: list[str] = []
        for path in paths:
            source = path.read_text(encoding="utf-8-sig")
            if VIETNAMESE_SPECIFIC_PATTERN.search(source):
                offenders.append(str(path.relative_to(REPOSITORY_ROOT)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
