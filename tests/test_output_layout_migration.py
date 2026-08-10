from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from hy3dshape.scripts.migrate_output_layout import (
    LiveWebUIProcessError,
    OutputMigrationCollisionError,
    OutputRootValidationError,
    main,
    migrate_output_layout,
)


class OutputLayoutMigrationTests(unittest.TestCase):
    def _output_root(self, temporary_directory: str) -> Path:
        root = Path(temporary_directory) / "output_folder"
        (root / "webui").mkdir(parents=True)
        return root

    def _legacy_generation(self, root: Path) -> tuple[str, Path]:
        generation_uid = str(uuid.uuid4())
        generation = root / "webui" / generation_uid
        generation.mkdir()
        (generation / "white_mesh.glb").write_bytes(b"legacy-generation")
        (generation / "generation.json").write_text(
            json.dumps(
                {
                    "generation_uid": generation_uid,
                    "storage_folder": f"legacy/{generation_uid}",
                    "events": [
                        {
                            "storage_folder": f"legacy/{generation_uid}",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return generation_uid, generation

    def test_default_is_a_read_only_dry_run_with_complete_plan(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._output_root(temporary_directory)
            generation_uid, generation = self._legacy_generation(root)
            (root / "dit").mkdir()
            (root / "inference").mkdir()
            (root / "quality_tests").mkdir()
            (root / "_runlogs").mkdir()
            (root / "cau_mong_multiview").mkdir()
            (root / "webui" / "env_maps").mkdir()
            (root / "webui" / "webui.stderr.log").write_text(
                "legacy stderr", encoding="utf-8"
            )
            (root / "webui" / "not-a-generation").mkdir()
            (root / "webui" / "loose.txt").write_text("loose", encoding="utf-8")

            plan = migrate_output_layout(
                root,
                process_is_alive=lambda _pid: False,
            )

            destinations = {action.destination for action in plan.actions}
            webui = (root / "webui").resolve()
            self.assertEqual(len(plan.actions), 10)
            self.assertIn(webui / "training", destinations)
            self.assertIn(webui / "inference", destinations)
            self.assertIn(webui / "quality_tests", destinations)
            self.assertIn(
                webui / "projects" / "cau_mong_multiview",
                destinations,
            )
            self.assertIn(webui / "logs" / "training_smoke", destinations)
            self.assertIn(webui / "generations" / generation_uid, destinations)
            self.assertIn(
                webui / "logs" / "legacy" / "webui.stderr.log",
                destinations,
            )
            self.assertIn(webui / "archive" / "legacy" / "env_maps", destinations)
            self.assertIn(
                webui / "archive" / "legacy" / "not-a-generation",
                destinations,
            )
            self.assertIn(
                webui / "archive" / "legacy" / "loose.txt",
                destinations,
            )
            self.assertTrue(generation.is_dir())
            self.assertTrue((root / "dit").is_dir())
            self.assertFalse((webui / "generations").exists())

    def test_apply_moves_every_category_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._output_root(temporary_directory)
            generation_uid, _generation = self._legacy_generation(root)
            payloads = {
                "dit": b"adapter",
                "inference": b"mesh",
                "quality_tests": b"quality",
                "_runlogs": b"training-log",
                "cau_mong_multiview": b"project",
            }
            for directory_name, payload in payloads.items():
                directory = root / directory_name
                directory.mkdir()
                (directory / "payload.bin").write_bytes(payload)

            webui = root / "webui"
            env_maps = webui / "env_maps"
            env_maps.mkdir()
            (env_maps / "studio.bin").write_bytes(b"environment")
            (webui / "webui.stderr.log").write_bytes(b"stderr")
            malformed_uuid = webui / "ec72d3f3-e4ea6-9747-ab12cf5e47e1"
            malformed_uuid.mkdir()
            (malformed_uuid / "metadata.json").write_bytes(b"metadata")
            (webui / "loose.bin").write_bytes(b"loose")

            first_plan = migrate_output_layout(
                root,
                apply=True,
                process_is_alive=lambda _pid: False,
            )

            self.assertEqual(len(first_plan.actions), 10)
            self.assertEqual(
                (webui / "training" / "payload.bin").read_bytes(), b"adapter"
            )
            self.assertEqual(
                (webui / "inference" / "payload.bin").read_bytes(), b"mesh"
            )
            self.assertEqual(
                (webui / "quality_tests" / "payload.bin").read_bytes(),
                b"quality",
            )
            self.assertEqual(
                (webui / "logs" / "training_smoke" / "payload.bin").read_bytes(),
                b"training-log",
            )
            self.assertEqual(
                (
                    webui / "projects" / "cau_mong_multiview" / "payload.bin"
                ).read_bytes(),
                b"project",
            )
            self.assertEqual(
                (webui / "logs" / "legacy" / "webui.stderr.log").read_bytes(),
                b"stderr",
            )
            self.assertEqual(
                (webui / "archive" / "legacy" / "env_maps" / "studio.bin").read_bytes(),
                b"environment",
            )
            self.assertEqual(
                (
                    webui / "archive" / "legacy" / malformed_uuid.name / "metadata.json"
                ).read_bytes(),
                b"metadata",
            )
            self.assertEqual(
                (webui / "archive" / "legacy" / "loose.bin").read_bytes(),
                b"loose",
            )
            generation = webui / "generations" / generation_uid
            self.assertEqual(
                (generation / "white_mesh.glb").read_bytes(),
                b"legacy-generation",
            )
            manifest = json.loads(
                (generation / "generation.json").read_text(encoding="utf-8")
            )
            expected_suffix = f"webui/generations/{generation_uid}"
            self.assertTrue(manifest["storage_folder"].endswith(expected_suffix))
            self.assertTrue(
                manifest["events"][0]["storage_folder"].endswith(expected_suffix)
            )

            second_plan = migrate_output_layout(
                root,
                apply=True,
                process_is_alive=lambda _pid: False,
            )
            self.assertEqual(second_plan.actions, ())
            self.assertEqual(
                (generation / "white_mesh.glb").read_bytes(),
                b"legacy-generation",
            )

    def test_collision_refuses_before_any_source_is_moved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._output_root(temporary_directory)
            _generation_uid, generation = self._legacy_generation(root)
            (root / "dit").mkdir()
            (root / "webui" / "training").mkdir()

            with self.assertRaises(OutputMigrationCollisionError):
                migrate_output_layout(
                    root,
                    apply=True,
                    process_is_alive=lambda _pid: False,
                )

            self.assertTrue((root / "dit").is_dir())
            self.assertTrue(generation.is_dir())
            self.assertFalse((root / "webui" / "generations").exists())

    def test_cli_is_dry_run_by_default_and_apply_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._output_root(temporary_directory)
            source = root / "dit"
            source.mkdir()
            (source / "adapter.bin").write_bytes(b"adapter")

            dry_run_output = StringIO()
            with redirect_stdout(dry_run_output):
                result = main(["--output-root", str(root)])

            self.assertEqual(result, 0)
            self.assertIn("DRY-RUN", dry_run_output.getvalue())
            self.assertTrue(source.is_dir())
            self.assertFalse((root / "webui" / "training").exists())

            apply_output = StringIO()
            with redirect_stdout(apply_output):
                result = main(["--output-root", str(root), "--apply"])

            self.assertEqual(result, 0)
            self.assertIn("APPLIED", apply_output.getvalue())
            self.assertFalse(source.exists())
            self.assertEqual(
                (root / "webui" / "training" / "adapter.bin").read_bytes(),
                b"adapter",
            )

    def test_live_legacy_or_current_pid_refuses_even_dry_run(self):
        for pid_parent in (Path("logs"), Path("runtime")):
            with self.subTest(pid_parent=pid_parent):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = self._output_root(temporary_directory)
                    pid_directory = root / "webui" / pid_parent
                    pid_directory.mkdir()
                    (pid_directory / "webui-8080.pid").write_text(
                        "4242", encoding="ascii"
                    )
                    (root / "dit").mkdir()

                    with self.assertRaises(LiveWebUIProcessError) as context:
                        migrate_output_layout(
                            root,
                            process_is_alive=lambda pid: pid == 4242,
                        )

                    self.assertEqual(context.exception.live_processes[0][0], 4242)
                    self.assertTrue((root / "dit").is_dir())

    def test_missing_or_invalid_webui_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "output_folder"
            root.mkdir()

            with self.assertRaises(OutputRootValidationError):
                migrate_output_layout(root, process_is_alive=lambda _pid: False)

            (root / "webui").write_text("not a directory", encoding="utf-8")
            with self.assertRaises(OutputRootValidationError):
                migrate_output_layout(root, process_is_alive=lambda _pid: False)


if __name__ == "__main__":
    unittest.main()
