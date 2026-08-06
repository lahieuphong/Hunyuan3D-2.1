from __future__ import annotations

import ast
import re
from pathlib import Path
import unittest
from unittest.mock import patch

from webui import load_ui_assets
import webui.i18n as i18n_module
from webui.asset_manifest import SCRIPT_FRAGMENTS, STYLE_FRAGMENTS
from webui.i18n import (
    DEFAULT_UI_LOCALE,
    ENABLED_UI_LOCALES,
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

    def test_application_toast_messages_are_catalogued(self):
        catalog_sources = {
            PLACEHOLDER_PATTERN.sub(
                "{}",
                re.sub(r"\s+", " ", translations["en"]).strip(),
            )
            for translations in ui_translation_catalog().values()
        }

        def static_message(argument: ast.AST) -> str | None:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                return argument.value
            if not isinstance(argument, ast.JoinedStr):
                return None
            parts: list[str] = []
            for part in argument.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                elif isinstance(part, ast.FormattedValue):
                    parts.append("{}")
                else:
                    return None
            return "".join(parts)

        missing: list[str] = []
        constructors = {"Error", "GenerationInputError", "GenerationUidConflictError"}
        for relative_path in ("gradio_app.py", "webui/generation_inputs.py"):
            path = REPOSITORY_ROOT / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                constructor = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                if constructor not in constructors:
                    continue
                message = static_message(node.args[0])
                if message is None:
                    continue
                normalized = PLACEHOLDER_PATTERN.sub(
                    "{}",
                    re.sub(r"\s+", " ", message).strip(),
                )
                if normalized not in catalog_sources:
                    missing.append(f"{relative_path}:{node.lineno}: {message}")

        self.assertEqual(missing, [])

    def test_gradio_toast_chrome_is_bilingual(self):
        catalog = ui_translation_catalog()
        required_sources = {
            "toast.title_error": "Error",
            "toast.title_warning": "Warning",
            "toast.title_info": "Info",
            "toast.title_success": "Success",
            "toast.close": "Close",
            "toast.recording_error": "Recording Error",
            "toast.recording_start_failed": "Failed to start recording: {error}",
            "toast.permission_denied_by_user": "Permission denied by user",
            "toast.processing_video": "Processing video",
            "toast.processing_wait": "This may take a few seconds...",
            "toast.processing_error": "Processing Error",
            "toast.processing_fallback": (
                "Failed to process recording. Saving original version."
            ),
            "toast.queue_connection_can_break": (
                "On mobile, the connection can break if this tab is unfocused "
                "or the device sleeps, losing your position in queue."
            ),
            "toast.queue_long_requests": (
                "There is a long queue of requests pending. "
                "Duplicate this Space to skip."
            ),
            "toast.queue_lost_connection": (
                "Lost connection due to leaving page. Rejoining queue..."
            ),
            "toast.queue_waiting_for_inputs": (
                "Waiting for file(s) to finish uploading, please retry."
            ),
        }
        for key, english in required_sources.items():
            with self.subTest(key=key):
                self.assertEqual(catalog[key]["en"], english)
                self.assertNotEqual(catalog[key]["zh-CN"], english)

        localized_error = translate_ui(
            "toast.permission_denied_by_user",
            "zh-CN",
        )
        self.assertEqual(
            translate_ui(
                "toast.recording_start_failed",
                "zh-CN",
                error=localized_error,
            ),
            "无法开始屏幕录制：用户拒绝了权限请求",
        )
        self.assertEqual(
            [
                translate_ui(key, "zh-CN")
                for key in (
                    "toast.title_error",
                    "toast.title_warning",
                    "toast.title_info",
                    "toast.title_success",
                )
            ],
            ["错误", "警告", "信息", "成功"],
        )

    def test_locale_normalization_and_translation(self):
        self.assertEqual(DEFAULT_UI_LOCALE, "zh-CN")
        self.assertIn(DEFAULT_UI_LOCALE, ENABLED_UI_LOCALES)
        expected_english_locale = (
            "en" if "en" in ENABLED_UI_LOCALES else DEFAULT_UI_LOCALE
        )
        self.assertEqual(normalize_ui_locale("en-US"), expected_english_locale)
        self.assertEqual(normalize_ui_locale("zh-Hans"), "zh-CN")
        self.assertEqual(normalize_ui_locale("zh_CN"), "zh-CN")
        self.assertEqual(normalize_ui_locale("vi"), "zh-CN")
        self.assertEqual(normalize_ui_locale(None), "zh-CN")
        self.assertEqual(
            translate_ui("action.generate_3d"),
            translate_ui("action.generate_3d", "zh-CN"),
        )
        self.assertEqual(
            translate_ui("shell.brand", "en", variant="2mv"),
            (
                "Hunyuan3D-2mv"
                if "en" in ENABLED_UI_LOCALES
                else "混元3D-2mv"
            ),
        )
        self.assertEqual(
            ui_translation_catalog()["shell.brand"]["en"].format(
                variant="2mv"
            ),
            "Hunyuan3D-2mv",
        )
        self.assertEqual(
            translate_ui("shell.brand", "zh-CN", variant="2mv"),
            "混元3D-2mv",
        )
        self.assertEqual(
            translate_ui("action.generate_3d", "zh-CN"),
            "生成 3D",
        )
        self.assertEqual(
            translate_ui("history.generation_title", "zh-CN", uid="abc"),
            "生成记录 abc",
        )

    def test_disabled_english_falls_back_to_simplified_chinese(self):
        with patch.object(i18n_module, "ENABLED_UI_LOCALES", ("zh-CN",)):
            self.assertEqual(normalize_ui_locale("en-US"), "zh-CN")
            self.assertEqual(normalize_ui_locale("en"), "zh-CN")
            self.assertEqual(normalize_ui_locale(None), "zh-CN")

        with patch.dict(
            i18n_module.UI_LANGUAGE_FLAGS,
            {"en": False, "zh-CN": True},
            clear=True,
        ):
            self.assertEqual(
                i18n_module.ui_language_config_json(),
                '{"defaultLocale":"zh-CN","flags":{"en":false,"zh-CN":true}}',
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
        self.assertNotIn("/*__UI_LANGUAGE_CONFIG__*/", javascript)
        self.assertNotIn("/*__UI_TRANSLATION_CATALOG__*/", javascript)
        self.assertIn('"action.generate_3d"', javascript)
        self.assertIn("installUiLocalization();", javascript)

    def test_windows_launcher_messages_are_simplified_chinese_utf8(self):
        source = (REPOSITORY_ROOT / "START_WEBUI.bat").read_text(
            encoding="utf-8"
        )
        simplified_chinese_messages = (
            "混元3D 四视图网页界面",
            "找不到网页界面启动脚本：",
            "找不到 Windows Python 环境：",
            "应存在的目录：.venv-win",
            "正在后台启动混元3D 网页界面...",
            "模型准备就绪后将自动打开浏览器。",
            r"运行日志：hy3dshape\output_folder\webui\logs",
            "混元3D 网页界面启动失败。",
            "退出代码：%WEBUI_EXIT_CODE%",
            "请保持此窗口打开，并发送错误信息以便排查。",
            "按任意键关闭此窗口。",
        )
        removed_english_messages = (
            "Cannot find the Web UI launcher:",
            "Cannot find the Windows Python environment:",
            "Expected folder: .venv-win",
            "Starting Hunyuan3D Web UI in the background...",
            "The browser will open automatically after the model is ready.",
            r"Runtime logs: hy3dshape\output_folder\webui\logs",
            "Hunyuan3D Web UI could not start.",
            "Exit code: %WEBUI_EXIT_CODE%",
            "Keep this window open and send the error text for inspection.",
            "Press any key to close this window.",
        )

        self.assertIn("chcp 65001 >nul", source)
        for message in simplified_chinese_messages:
            with self.subTest(message=message):
                self.assertIn(message, source)
        for message in removed_english_messages:
            with self.subTest(removed=message):
                self.assertNotIn(message, source)
        self.assertEqual(source.count("pause >nul"), 3)
        self.assertNotRegex(source, r"(?m)^\s*pause\s*$")

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
