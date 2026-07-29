from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import unittest

from webui import load_ui_assets, render_ten_view_panel
from webui.asset_manifest import SCRIPT_FRAGMENTS, STYLE_FRAGMENTS
from webui.ten_view_templates import TEN_VIEW_DEFINITIONS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _TenViewMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: list[dict[str, str | None]] = []
        self.file_inputs: list[dict[str, str | None]] = []
        self.ui_only = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("data-ui-only") == "true":
            self.ui_only = True
        if "data-ten-view-card" in attributes:
            self.cards.append(attributes)
        if tag == "input" and "data-ten-view-input" in attributes:
            self.file_inputs.append(attributes)


class TenViewDefinitionTests(unittest.TestCase):
    def test_defines_the_exact_ten_camera_slots_in_order(self):
        self.assertEqual(
            [
                (view.key, view.label, view.angle)
                for view in TEN_VIEW_DEFINITIONS
            ],
            [
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
            ],
        )
        keys = [view.key for view in TEN_VIEW_DEFINITIONS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rendered_inputs_are_unique_and_explicitly_ui_only(self):
        parser = _TenViewMarkupParser()
        parser.feed(render_ten_view_panel())

        self.assertTrue(parser.ui_only)
        self.assertEqual(len(parser.cards), 10)
        self.assertEqual(len(parser.file_inputs), 10)
        self.assertEqual(
            [card["data-view-key"] for card in parser.cards],
            [view.key for view in TEN_VIEW_DEFINITIONS],
        )
        input_ids = [attributes["id"] for attributes in parser.file_inputs]
        self.assertEqual(len(input_ids), len(set(input_ids)))
        self.assertTrue(
            all(identifier.startswith("ten-view-input-") for identifier in input_ids)
        )
        self.assertTrue(
            all(attributes["type"] == "file" for attributes in parser.file_inputs)
        )
        self.assertTrue(
            all("name" not in attributes for attributes in parser.file_inputs)
        )


class TenViewAssetContractTests(unittest.TestCase):
    def test_assets_load_after_the_existing_upload_and_left_rail_layers(self):
        self.assertGreater(
            STYLE_FRAGMENTS.index("styles/75-ten-view-inputs.css"),
            STYLE_FRAGMENTS.index("styles/70-left-rail-settings.css"),
        )
        self.assertGreater(
            SCRIPT_FRAGMENTS.index("scripts/18-ten-view-inputs.js"),
            SCRIPT_FRAGMENTS.index("scripts/15-upload-previews.js"),
        )

        css, javascript = load_ui_assets()
        self.assertIn('data-tab-id="tab_ten_prompt"', css)
        self.assertIn(".tab-container.visually-hidden", css)
        self.assertIn(".generate-actions", css)
        self.assertIn("display: none !important", css)
        self.assertIn("const installTenViewInputs", javascript)
        self.assertIn('slug: "ten-view"', javascript)
        self.assertEqual(javascript.count("installTenViewInputs();"), 2)

    def test_generation_contract_has_no_ten_view_mode_or_callback_inputs(self):
        application = (REPOSITORY_ROOT / "gradio_app.py").read_text(
            encoding="utf-8",
        )
        self.assertEqual(application.count("id='tab_ten_prompt'"), 1)
        self.assertEqual(application.count("render_ten_view_panel()"), 1)
        self.assertNotIn("'ten-ui-only'", application)
        self.assertNotIn("input_mode='ten'", application)
        self.assertNotIn("ten_view_input_", application)


if __name__ == "__main__":
    unittest.main()
