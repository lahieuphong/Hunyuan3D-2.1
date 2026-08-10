from __future__ import annotations

from pathlib import Path
import unittest

from webui import load_ui_assets
from webui.asset_manifest import SCRIPT_FRAGMENTS, STYLE_FRAGMENTS
from webui.ten_view_inputs import (
    TEN_VIEW_DEFINITIONS,
    render_ten_view_progress,
    render_ten_view_summary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TenViewDefinitionTests(unittest.TestCase):
    def test_defines_the_exact_ten_camera_slots_in_order(self):
        self.assertEqual(
            tuple((view.key, view.label, view.angle) for view in TEN_VIEW_DEFINITIONS),
            (
                ("front", "Front", "0°"),
                ("front_right", "Front-right", "45°"),
                ("right", "Right", "90°"),
                ("back_right", "Back-right", "135°"),
                ("back", "Back", "180°"),
                ("back_left", "Back-left", "225°"),
                ("left", "Left", "270°"),
                ("front_left", "Front-left", "315°"),
                ("high_front", "High-front", "30° down"),
                ("high_back", "High-back", "30° down"),
            ),
        )
        keys = [view.key for view in TEN_VIEW_DEFINITIONS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_progress_is_server_backed_and_reaches_ready_state(self):
        empty = render_ten_view_progress()
        complete = render_ten_view_progress(*([object()] * 10))

        self.assertIn("0 / 10", empty)
        self.assertIn("Add 10 remaining view(s)", empty)
        self.assertIn("10 / 10", complete)
        self.assertIn("Ready to generate", complete)
        self.assertIn('aria-valuenow="10"', complete)

    def test_summary_discloses_the_experimental_adapter(self):
        summary = render_ten_view_summary()
        self.assertIn(
            'data-conditioning-strategy="native-cardinal-shape-with-ten-view-texture-rc"',
            summary,
        )
        self.assertIn("All 10 images are validated and saved", summary)
        self.assertIn("Front, Left, Back and Right", summary)
        self.assertIn("reserved for", summary)
        self.assertIn("RC quality scoring", summary)


class TenViewAssetContractTests(unittest.TestCase):
    def test_assets_load_after_the_existing_upload_and_left_rail_layers(self):
        self.assertGreater(
            STYLE_FRAGMENTS.index("styles/75-ten-view-generation.css"),
            STYLE_FRAGMENTS.index("styles/70-left-rail-settings.css"),
        )
        self.assertNotIn("scripts/18-ten-view-inputs.js", SCRIPT_FRAGMENTS)
        self.assertIn("scripts/18-ten-view-loading.js", SCRIPT_FRAGMENTS)
        self.assertGreater(
            SCRIPT_FRAGMENTS.index("scripts/18-ten-view-loading.js"),
            SCRIPT_FRAGMENTS.index("scripts/15-upload-previews.js"),
        )

        css, javascript = load_ui_assets()
        self.assertIn(".ten-view-upload-grid", css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr));", css)
        self.assertIn("min-height: 160px !important;", css)
        self.assertIn("min-height: 0 !important;", css)
        self.assertIn(".ten-view-experimental-badge", css)
        self.assertIn('data-ten-view-preview-state="loading"', css)
        self.assertNotIn('data-ten-view-loading-state="loading"', css)
        self.assertNotIn("const installTenViewInputs", javascript)
        self.assertIn("const installTenViewHistoryLoading", javascript)
        self.assertIn('image.loading = "eager"', javascript)
        self.assertIn("image.fetchPriority = priority", javascript)
        self.assertIn("await image.decode()", javascript)
        self.assertIn("tenViewHistoryPreviewState.slots", javascript)
        self.assertNotIn("Promise.all(", javascript)
        self.assertIn('slug: "ten-view"', javascript)
        self.assertIn('mode === "ten"', javascript)
        self.assertIn('"10-VIEW"', javascript)

    def test_tab_routing_supports_gradio_overflow_menu(self):
        _, javascript = load_ui_assets()

        self.assertIn('tabId: "tab_ten_prompt"', javascript)
        self.assertIn('label: "10 Views"', javascript)
        self.assertIn("const promptOverflowTabButton", javascript)
        self.assertIn(".overflow-dropdown button", javascript)
        self.assertIn("activatePromptTab(expectedRoute)", javascript)
        self.assertLess(
            javascript.index("activatePromptTab(expectedRoute)"),
            javascript.index('document.body.classList.add("is-history-review")'),
        )

    def test_input_mode_switcher_keeps_all_four_tabs_on_one_row(self):
        css = (
            REPOSITORY_ROOT / "webui/assets/styles/70-left-rail-settings.css"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "#prompt-mode-tabs > .tab-wrapper > .tab-container.visually-hidden",
            css,
        )
        self.assertIn("flex: 1 1 100%;", css)
        self.assertIn("flex: 1 1 0 !important;", css)
        self.assertIn("text-overflow: ellipsis;", css)
        self.assertNotIn(
            "grid-template-columns: repeat(2, minmax(0, 1fr));",
            css,
        )

    def test_generation_contract_uses_native_gradio_images_and_callbacks(self):
        application = (REPOSITORY_ROOT / "gradio_app.py").read_text(
            encoding="utf-8",
        )
        self.assertEqual(application.count("id='tab_ten_prompt'"), 1)
        self.assertEqual(application.count("elem_id='ten-view-upload-grid'"), 1)
        self.assertEqual(application.count("elem_classes='ten-view-upload-grid'"), 1)
        self.assertIn(
            "for index, definition in enumerate(TEN_VIEW_DEFINITIONS):",
            application,
        )
        self.assertNotIn("for row_start in range", application)
        grid_start = application.index("elem_id='ten-view-upload-grid'")
        grid_end = application.index("elem_id='ten-view-summary-host'", grid_start)
        self.assertIn("height=160", application[grid_start:grid_end])
        self.assertIn(") as tab_ten:", application)
        self.assertIn("ten_view_components[definition.key] = gr.Image(", application)
        self.assertIn("type='pil'", application)
        self.assertIn("image_mode='RGBA'", application)
        self.assertIn("tab_ten.select(", application)
        self.assertIn("ten_view_input.input(", application)
        self.assertNotIn("ten_view_input.change(", application)
        self.assertIn("preprocess=False", application)
        self.assertEqual(application.count("*ten_view_inputs"), 4)
        self.assertIn("api_name='shape_generation_ten'", application)
        self.assertIn("api_name='shape_generation'", application)
        gallery_guard = application.index("# A hidden gr.Examples")
        gallery_end = application.index("with gr.Column(", gallery_guard + 1)
        gallery_setup = application[gallery_guard:gallery_end]
        self.assertIn("if not MV_MODE:", gallery_setup)
        self.assertNotIn("visible=not MV_MODE", gallery_setup)
        self.assertNotIn("render_ten_view_panel", application)
        self.assertNotIn("data-ui-only", application)

    def test_obsolete_browser_file_picker_is_not_loaded(self):
        bootstrap = (
            REPOSITORY_ROOT / "webui/assets/scripts/90-bootstrap.js"
        ).read_text(encoding="utf-8")
        self.assertNotIn("installTenViewInputs", bootstrap)


if __name__ == "__main__":
    unittest.main()
