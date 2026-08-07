from __future__ import annotations

import unittest

from webui.templates import render_topbar
from webui.topbar_config import (
    SHOW_API_DOCS_BUTTON,
    SHOW_GPU_BUTTON,
    SHOW_HISTORY_BUTTON,
    SHOW_SETTINGS_BUTTON,
    TOPBAR_BUTTON_FLAGS,
)


HARDWARE_ACTION = """
<button id="app-rtx-profile" class="app-topbar-button" type="button">
    <span>RTX 3090</span>
</button>
"""

BUTTON_IDS = {
    "show_api_docs_button": "app-api-docs",
    "show_history_button": "app-generation-history",
    "show_settings_button": "app-theme-settings",
    "show_gpu_button": "app-rtx-profile",
}


class TopbarConfigTests(unittest.TestCase):
    def test_topbar_flags_are_boolean_and_mapping_stays_aligned(self):
        for flag in (
            SHOW_API_DOCS_BUTTON,
            SHOW_HISTORY_BUTTON,
            SHOW_SETTINGS_BUTTON,
            SHOW_GPU_BUTTON,
        ):
            self.assertIsInstance(flag, bool)
        self.assertEqual(
            TOPBAR_BUTTON_FLAGS,
            {
                "api_docs": SHOW_API_DOCS_BUTTON,
                "history": SHOW_HISTORY_BUTTON,
                "settings": SHOW_SETTINGS_BUTTON,
                "gpu": SHOW_GPU_BUTTON,
            },
        )

    def test_renderer_keeps_all_four_buttons_enabled_by_default(self):
        rendered = render_topbar("Brand", "Workspace", HARDWARE_ACTION)
        for button_id in BUTTON_IDS.values():
            with self.subTest(button_id=button_id):
                self.assertIn(f'id="{button_id}"', rendered)

    def test_each_false_flag_removes_only_its_topbar_button(self):
        for disabled_argument, disabled_id in BUTTON_IDS.items():
            with self.subTest(disabled_argument=disabled_argument):
                arguments = {name: True for name in BUTTON_IDS}
                arguments[disabled_argument] = False
                rendered = render_topbar(
                    "Brand",
                    "Workspace",
                    HARDWARE_ACTION,
                    **arguments,
                )

                self.assertNotIn(f'id="{disabled_id}"', rendered)
                for enabled_argument, enabled_id in BUTTON_IDS.items():
                    if enabled_argument != disabled_argument:
                        self.assertIn(f'id="{enabled_id}"', rendered)

    def test_runtime_config_controls_the_rendered_buttons(self):
        configured_flags = {
            "show_api_docs_button": SHOW_API_DOCS_BUTTON,
            "show_history_button": SHOW_HISTORY_BUTTON,
            "show_settings_button": SHOW_SETTINGS_BUTTON,
            "show_gpu_button": SHOW_GPU_BUTTON,
        }
        rendered = render_topbar(
            "Brand",
            "Workspace",
            HARDWARE_ACTION,
            **configured_flags,
        )

        for argument, button_id in BUTTON_IDS.items():
            assertion = self.assertIn if configured_flags[argument] else self.assertNotIn
            assertion(f'id="{button_id}"', rendered)

    def test_topbar_action_navigation_is_omitted_when_all_flags_are_false(self):
        rendered = render_topbar(
            "Brand",
            "Workspace",
            HARDWARE_ACTION,
            **{key: False for key in BUTTON_IDS},
        )

        self.assertNotIn("app-topbar-actions", rendered)
        for button_id in BUTTON_IDS.values():
            self.assertNotIn(f'id="{button_id}"', rendered)

    def test_gpu_flag_cannot_create_a_button_without_available_hardware(self):
        rendered = render_topbar(
            "Brand",
            "Workspace",
            "",
            **{key: True for key in BUTTON_IDS},
        )

        self.assertNotIn('id="app-rtx-profile"', rendered)

    def test_renderer_exposes_dynamic_title_runtime_flags(self):
        rendered = render_topbar(
            "Brand",
            "Workspace",
            HARDWARE_ACTION,
            dynamic_input_titles=True,
            turbo_mode=True,
        )

        self.assertIn('data-ui-dynamic-input-title="true"', rendered)
        self.assertIn('data-ui-turbo-mode="true"', rendered)


if __name__ == "__main__":
    unittest.main()
