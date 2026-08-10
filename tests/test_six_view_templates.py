from __future__ import annotations

from pathlib import Path
import unittest

from hy3dshape.six_view import (
    SIX_VIEW_CONDITIONING_STRATEGY,
    SIX_VIEW_KEYS,
)
from webui import load_ui_assets
from webui.asset_manifest import STYLE_FRAGMENTS
from webui.i18n import translate_ui, ui_translation_catalog
from webui.six_view_inputs import (
    SIX_VIEW_DEFINITIONS,
    render_six_view_guide,
    render_six_view_progress,
    render_six_view_summary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SixViewDefinitionTests(unittest.TestCase):
    def test_defines_the_exact_six_camera_slots_in_capture_order(self):
        self.assertEqual(
            tuple((view.key, view.label, view.angle) for view in SIX_VIEW_DEFINITIONS),
            (
                ("front", "Front", "0°"),
                ("back", "Back", "180°"),
                ("left", "Left", "270°"),
                ("right", "Right", "90°"),
                ("top", "Top", "90° down"),
                ("bottom", "Bottom", "90° up"),
            ),
        )
        self.assertEqual(
            tuple(view.key for view in SIX_VIEW_DEFINITIONS),
            SIX_VIEW_KEYS,
        )

    def test_progress_is_server_backed_and_accessible(self):
        empty = render_six_view_progress()
        partial = render_six_view_progress(*([object()] * 4))
        complete = render_six_view_progress(*([object()] * 6))

        self.assertIn("0 / 6", empty)
        self.assertIn("Add 6 remaining view(s)", empty)
        self.assertIn("4 / 6", partial)
        self.assertIn('aria-valuenow="4"', partial)
        self.assertIn("6 / 6", complete)
        self.assertIn("Ready to generate", complete)
        self.assertIn('role="progressbar"', complete)

    def test_copy_truthfully_separates_geometry_and_color_projection(self):
        guide = render_six_view_guide()
        summary = render_six_view_summary()

        self.assertIn("Native 4-view geometry", guide)
        self.assertIn("6-view color projection", guide)
        self.assertIn(
            f'data-conditioning-strategy="{SIX_VIEW_CONDITIONING_STRATEGY}"',
            summary,
        )
        self.assertIn("Geometry uses", summary)
        normalized_summary = " ".join(summary.split())
        self.assertIn("All six views, including Top and Bottom", normalized_summary)
        self.assertIn("orthographic color projection", normalized_summary)


class SixViewFrontendContractTests(unittest.TestCase):
    def test_asset_is_isolated_and_loaded_before_ten_view_overrides(self):
        self.assertIn("styles/74-six-view-generation.css", STYLE_FRAGMENTS)
        self.assertLess(
            STYLE_FRAGMENTS.index("styles/74-six-view-generation.css"),
            STYLE_FRAGMENTS.index("styles/75-ten-view-generation.css"),
        )
        css = (
            REPOSITORY_ROOT / "webui/assets/styles/74-six-view-generation.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".six-view-upload-grid", css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr));", css)
        self.assertIn(".six-view-image", css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)

    def test_route_mode_and_history_lookup_are_data_driven(self):
        _, javascript = load_ui_assets()

        self.assertIn('mode: "six", slug: "six-view"', javascript)
        self.assertIn('tabId: "tab_six_prompt"', javascript)
        self.assertIn('label: "6 Views"', javascript)
        self.assertIn("const promptRouteForMode", javascript)
        self.assertIn("promptRouteForMode(reviewMode)", javascript)
        self.assertIn("candidate.mode === item.input_mode", javascript)
        self.assertNotIn('reviewMode === "ten"', javascript)
        self.assertIn('"6-VIEW": "generation.console.mode.six"', javascript)

    def test_upload_preview_and_generate_copy_support_six(self):
        _, javascript = load_ui_assets()

        for key in SIX_VIEW_KEYS:
            self.assertIn(f'"six-image-{key}"', javascript)
        self.assertIn('return "six"', javascript)
        self.assertIn('six: "action.generate_subtitle_six"', javascript)
        self.assertIn("route.tabId === selectedTabId", javascript)

    def test_gradio_contract_uses_native_components_and_progress_callback(self):
        application = (REPOSITORY_ROOT / "gradio_app.py").read_text(
            encoding="utf-8"
        )

        self.assertEqual(application.count("id='tab_six_prompt'"), 1)
        self.assertEqual(application.count("elem_id='six-view-upload-grid'"), 1)
        self.assertIn(
            "for index, definition in enumerate(SIX_VIEW_DEFINITIONS):",
            application,
        )
        self.assertIn("six_view_components[definition.key] = gr.Image(", application)
        self.assertIn("tab_six.select(", application)
        self.assertIn("six_view_input.input(", application)
        self.assertIn("fn=render_six_view_progress", application)
        self.assertIn("api_name='shape_generation_six'", application)

    def test_english_and_simplified_chinese_catalog_is_complete(self):
        expectations = {
            "shell.workspace.six": ("Six-View Image to 3D Generation", "六视图图像转 3D 生成"),
            "input.six": ("6 Views", "6 视图"),
            "action.generate_six": ("Generate 3D · 6 Images", "生成 3D · 6 张图像"),
            "generation.console.mode.six": ("6-VIEW", "六视图"),
        }
        catalog = ui_translation_catalog()
        for key, (english, chinese) in expectations.items():
            with self.subTest(key=key):
                self.assertEqual(catalog[key]["en"], english)
                self.assertEqual(translate_ui(key, "zh-CN"), chinese)


if __name__ == "__main__":
    unittest.main()
