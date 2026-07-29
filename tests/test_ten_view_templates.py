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
            'data-conditioning-strategy="experimental-feature-fusion-10-to-4"',
            summary,
        )
        self.assertIn("All 10 images are encoded", summary)
        self.assertIn("not a natively trained 10-camera checkpoint", summary)


class TenViewAssetContractTests(unittest.TestCase):
    def test_assets_load_after_the_existing_upload_and_left_rail_layers(self):
        self.assertGreater(
            STYLE_FRAGMENTS.index("styles/75-ten-view-generation.css"),
            STYLE_FRAGMENTS.index("styles/70-left-rail-settings.css"),
        )
        self.assertNotIn("scripts/18-ten-view-inputs.js", SCRIPT_FRAGMENTS)

        css, javascript = load_ui_assets()
        self.assertIn(".ten-view-upload-row", css)
        self.assertIn(".ten-view-experimental-badge", css)
        self.assertNotIn("const installTenViewInputs", javascript)
        self.assertIn('slug: "ten-view"', javascript)
        self.assertIn('mode === "ten"', javascript)
        self.assertIn('"10-VIEW"', javascript)

    def test_generation_contract_uses_native_gradio_images_and_callbacks(self):
        application = (REPOSITORY_ROOT / "gradio_app.py").read_text(
            encoding="utf-8",
        )
        self.assertEqual(application.count("id='tab_ten_prompt'"), 1)
        self.assertIn(") as tab_ten:", application)
        self.assertIn("ten_view_components[definition.key] = gr.Image(", application)
        self.assertIn("type='pil'", application)
        self.assertIn("image_mode='RGBA'", application)
        self.assertIn("tab_ten.select(", application)
        self.assertEqual(application.count("*ten_view_inputs"), 4)
        self.assertIn("api_name='shape_generation_ten'", application)
        self.assertIn("api_name='shape_generation'", application)
        self.assertNotIn("render_ten_view_panel", application)
        self.assertNotIn("data-ui-only", application)

    def test_obsolete_browser_file_picker_is_not_loaded(self):
        bootstrap = (
            REPOSITORY_ROOT / "webui/assets/scripts/90-bootstrap.js"
        ).read_text(encoding="utf-8")
        self.assertNotIn("installTenViewInputs", bootstrap)


if __name__ == "__main__":
    unittest.main()
