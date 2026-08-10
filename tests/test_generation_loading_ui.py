from __future__ import annotations

from pathlib import Path
import unittest

from webui import load_ui_assets, render_generation_loading
from webui.asset_manifest import SCRIPT_FRAGMENTS, STYLE_FRAGMENTS
from webui.i18n import ui_translation_catalog


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class GenerationLoadingTemplateTests(unittest.TestCase):
    def test_all_output_surfaces_have_accessible_loading_shells(self):
        expected_icons = {
            "preview": "box",
            "export": "download",
            "statistics": "terminal",
        }

        for kind, icon in expected_icons.items():
            with self.subTest(kind=kind):
                markup = render_generation_loading(kind)
                self.assertIn(
                    f'data-generation-loading-kind="{kind}"',
                    markup,
                )
                self.assertIn(f'data-ui-icon="{icon}"', markup)
                self.assertIn('data-state="idle"', markup)
                self.assertIn('role="status"', markup)
                self.assertIn('aria-live="polite"', markup)
                self.assertIn('aria-atomic="true"', markup)
                self.assertIn('aria-hidden="true"', markup)
                self.assertIn('role="progressbar"', markup)
                self.assertIn('aria-valuemin="0"', markup)
                self.assertIn('aria-valuemax="100"', markup)
                self.assertIn('aria-valuenow="0"', markup)

    def test_unknown_loading_surface_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown generation loading kind",
        ):
            render_generation_loading("unknown")


class GenerationLoadingAssetTests(unittest.TestCase):
    def test_loading_assets_follow_console_and_results_layers(self):
        self.assertGreater(
            SCRIPT_FRAGMENTS.index("scripts/32-generation-workspace-loading.js"),
            SCRIPT_FRAGMENTS.index("scripts/30-generation-console.js"),
        )
        self.assertLess(
            SCRIPT_FRAGMENTS.index("scripts/32-generation-workspace-loading.js"),
            SCRIPT_FRAGMENTS.index("scripts/90-bootstrap.js"),
        )
        self.assertGreater(
            STYLE_FRAGMENTS.index("styles/62-generation-loading.css"),
            STYLE_FRAGMENTS.index("styles/60-results-console.css"),
        )
        self.assertLess(
            STYLE_FRAGMENTS.index("styles/62-generation-loading.css"),
            STYLE_FRAGMENTS.index("styles/80-light-theme.css"),
        )

    def test_loading_lifecycle_uses_manifest_progress_and_stable_hosts(self):
        application = (REPOSITORY_ROOT / "gradio_app.py").read_text(
            encoding="utf-8",
        )
        console = (
            REPOSITORY_ROOT
            / "webui/assets/scripts/30-generation-console.js"
        ).read_text(encoding="utf-8")
        loading = (
            REPOSITORY_ROOT
            / "webui/assets/scripts/32-generation-workspace-loading.js"
        ).read_text(encoding="utf-8")
        bootstrap = (
            REPOSITORY_ROOT / "webui/assets/scripts/90-bootstrap.js"
        ).read_text(encoding="utf-8")

        for host_id in (
            "mesh-viewer-loading",
            "mesh-export-loading",
            "mesh-stats-loading",
        ):
            self.assertIn(f"elem_id='{host_id}'", application)
        self.assertIn("startGenerationWorkspaceLoading(uid, resumed)", console)
        self.assertIn(
            "syncGenerationWorkspaceProgress(safeProgress, stage)",
            console,
        )
        self.assertIn(
            "syncGenerationWorkspaceManifest(manifest, lastEventMessage)",
            console,
        )
        self.assertIn("manifest?.generation_uid", loading)
        self.assertIn('status === "completed"', loading)
        self.assertIn('status === "failed"', loading)
        self.assertIn("waitForGenerationWorkspaceResult(uid)", loading)
        self.assertIn(
            'generationWorkspaceLoadingState.status !== "queued"',
            loading,
        )
        self.assertIn(
            'generationWorkspaceLoadingState.status !== "running"',
            loading,
        )
        self.assertIn('pathname === "/generation-viewer/" + uid', loading)
        self.assertIn('pathname.startsWith("/static/" + uid + "/")', loading)
        self.assertIn("generationWorkspaceFrameLoadedForUid", loading)
        self.assertIn("window.setInterval(check, 500)", loading)
        self.assertIn("window.clearInterval(generationWorkspaceReadyTimer)", loading)
        self.assertIn("const setGenerationWorkspaceText", loading)
        self.assertIn(
            "if (element.textContent !== nextValue) element.textContent = nextValue",
            loading,
        )
        self.assertIn('surface.toggleAttribute("inert", visible)', loading)
        self.assertIn('surface.setAttribute("aria-busy"', loading)
        self.assertIn('getElementById("generation-output-card")?.setAttribute', loading)
        self.assertIn("installGenerationWorkspaceLoading();", bootstrap)
        self.assertGreaterEqual(application.count(".success("), 2)

    def test_loading_supports_all_input_modes_and_error_keeps_overlay(self):
        console = (
            REPOSITORY_ROOT
            / "webui/assets/scripts/30-generation-console.js"
        ).read_text(encoding="utf-8")
        loading = (
            REPOSITORY_ROOT
            / "webui/assets/scripts/32-generation-workspace-loading.js"
        ).read_text(encoding="utf-8")

        self.assertIn("tabRoutes.find((route) => route.slug === inputTab)?.mode", console)
        self.assertIn('"10-VIEW"', console)
        self.assertIn('"6-VIEW"', console)
        self.assertIn('"4-VIEW"', console)
        self.assertIn('"1-VIEW"', console)
        self.assertIn('getAttribute("aria-busy") === "true"', console)
        self.assertIn("event?.stopImmediatePropagation()", console)
        self.assertIn('status: "failed"', loading)
        self.assertIn('host.classList.toggle("is-visible", visible)', loading)
        self.assertIn('state.status !== "idle"', loading)
        self.assertIn('window.addEventListener("ui-language-change"', loading)

    def test_loading_css_matches_stage_and_reduced_motion_contract(self):
        css = (
            REPOSITORY_ROOT
            / "webui/assets/styles/62-generation-loading.css"
        ).read_text(encoding="utf-8")

        self.assertIn("#mesh-viewer-loading", css)
        self.assertIn("#mesh-export-loading", css)
        self.assertIn("#mesh-stats-loading", css)
        self.assertIn(").is-visible", css)
        self.assertIn("height: var(--ui-stage-height) !important;", css)
        self.assertIn("#090e19", css)
        self.assertIn("#252d40", css)
        self.assertIn(".generation-tab-loading-indicator", css)
        self.assertIn(".generation-action-spinner", css)
        self.assertIn(".is-generation-metric-pending", css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)
        self.assertIn("animation: none !important;", css)

    def test_loading_copy_is_complete_in_english_and_simplified_chinese(self):
        catalog = ui_translation_catalog()
        expected_english = {
            "generation.loading.preview_title": "Generating 3D preview",
            "generation.loading.preview_body": (
                "Building geometry and the interactive preview from your input views."
            ),
            "generation.loading.export_title": "Preparing mesh for export",
            "generation.loading.export_body": (
                "The downloadable mesh will appear after generation is finalized."
            ),
            "generation.loading.statistics_title": "Calculating mesh statistics",
            "generation.loading.statistics_body": (
                "Polygon, vertex and timing data will appear when the mesh is ready."
            ),
            "generation.loading.progress_label": "Live generation progress",
            "generation.loading.console_hint": (
                "Live progress is available in Generation Console."
            ),
            "generation.loading.output_file": "Preparing output file…",
        }

        for key, english in expected_english.items():
            with self.subTest(key=key):
                self.assertEqual(catalog[key]["en"], english)
                self.assertTrue(catalog[key]["zh-CN"].strip())
                self.assertNotEqual(catalog[key]["zh-CN"], english)

        css, javascript = load_ui_assets()
        self.assertIn("generation-workspace-loading", css)
        self.assertIn("const startGenerationWorkspaceLoading", javascript)


if __name__ == "__main__":
    unittest.main()
