from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from webui.storage import (
    StorageMigrationCollisionError,
    WebUIStorageLayout,
    prepare_webui_storage,
)


def _manifest_storage_path(path: Path) -> str:
    try:
        value = os.path.relpath(path, start=Path.cwd())
    except ValueError:
        value = str(path.resolve(strict=False))
    return value.replace(os.sep, "/")


class WebUIStorageTests(unittest.TestCase):
    def test_prepares_only_runtime_directories_and_defines_optional_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"

            layout = prepare_webui_storage(root, migrate_legacy=False)

            self.assertIsInstance(layout, WebUIStorageLayout)
            self.assertEqual(layout.root, root.resolve())
            self.assertEqual(
                {
                    path.relative_to(layout.root).as_posix()
                    for path in layout.directories()
                },
                {
                    "generations",
                    "logs",
                    "runtime",
                    "training",
                    "inference",
                    "quality_tests",
                    "projects",
                    "archive",
                },
            )
            self.assertTrue(layout.root.is_dir())
            for directory in layout.runtime_directories():
                with self.subTest(directory=directory.name):
                    self.assertTrue(directory.is_dir())
            for directory in (
                layout.training,
                layout.inference,
                layout.quality_tests,
                layout.projects,
                layout.archive,
            ):
                with self.subTest(optional_directory=directory.name):
                    self.assertFalse(directory.exists())

    def test_migrates_canonical_uuid_and_updates_all_manifest_storage_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            generation_uid = str(uuid.uuid4())
            legacy_folder = root / generation_uid
            legacy_folder.mkdir()
            (legacy_folder / "white_mesh.glb").write_bytes(b"glTF-test")
            (legacy_folder / "generation.json").write_text(
                json.dumps(
                    {
                        "generation_uid": generation_uid,
                        "storage_folder": f"legacy/{generation_uid}",
                        "events": [
                            {
                                "details": {
                                    "storage_folder": f"legacy/{generation_uid}"
                                }
                            },
                            {"storage_folder": f"legacy/{generation_uid}"},
                        ],
                        "unrelated": {"storage_folder_name": "unchanged"},
                    }
                ),
                encoding="utf-8",
            )

            layout = prepare_webui_storage(root)

            migrated_folder = layout.generations / generation_uid
            self.assertFalse(os.path.lexists(legacy_folder))
            self.assertEqual(
                (migrated_folder / "white_mesh.glb").read_bytes(),
                b"glTF-test",
            )
            manifest = json.loads(
                (migrated_folder / "generation.json").read_text(encoding="utf-8")
            )
            expected_storage = _manifest_storage_path(migrated_folder)
            self.assertEqual(manifest["storage_folder"], expected_storage)
            self.assertEqual(
                manifest["events"][0]["details"]["storage_folder"],
                expected_storage,
            )
            self.assertEqual(
                manifest["events"][1]["storage_folder"],
                expected_storage,
            )
            self.assertEqual(
                manifest["unrelated"]["storage_folder_name"],
                "unchanged",
            )
            self.assertEqual(
                list(migrated_folder.glob(".generation.json.*.tmp")),
                [],
            )

    def test_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            generation_uid = str(uuid.uuid4())
            legacy_folder = root / generation_uid
            legacy_folder.mkdir()
            (legacy_folder / "marker.txt").write_text("keep", encoding="utf-8")

            first_layout = prepare_webui_storage(root)
            second_layout = prepare_webui_storage(root)

            self.assertEqual(first_layout, second_layout)
            self.assertEqual(
                (second_layout.generations / generation_uid / "marker.txt").read_text(
                    encoding="utf-8"
                ),
                "keep",
            )
            self.assertFalse(os.path.lexists(legacy_folder))

    def test_second_prepare_repairs_manifest_after_first_atomic_write_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            generation_uid = str(uuid.uuid4())
            legacy_folder = root / generation_uid
            legacy_folder.mkdir()
            manifest_path = legacy_folder / "generation.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "generation_uid": generation_uid,
                        "storage_folder": f"legacy/{generation_uid}",
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "webui.storage._atomic_write_json",
                side_effect=OSError("simulated manifest write failure"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "simulated manifest write failure",
                ):
                    prepare_webui_storage(root)

            migrated_folder = root / "generations" / generation_uid
            self.assertFalse(os.path.lexists(legacy_folder))
            self.assertTrue(migrated_folder.is_dir())
            self.assertEqual(
                json.loads(
                    (migrated_folder / "generation.json").read_text(
                        encoding="utf-8"
                    )
                )["storage_folder"],
                f"legacy/{generation_uid}",
            )

            layout = prepare_webui_storage(root)

            repaired_manifest = json.loads(
                (layout.generations / generation_uid / "generation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                repaired_manifest["storage_folder"],
                _manifest_storage_path(layout.generations / generation_uid),
            )

    def test_collision_preflight_does_not_move_or_overwrite_any_candidate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            first_uid = str(uuid.uuid4())
            colliding_uid = str(uuid.uuid4())
            first_source = root / first_uid
            colliding_source = root / colliding_uid
            first_source.mkdir()
            colliding_source.mkdir()
            (first_source / "marker.txt").write_text("first", encoding="utf-8")
            (colliding_source / "marker.txt").write_text("source", encoding="utf-8")

            initial_layout = prepare_webui_storage(root, migrate_legacy=False)
            collision = initial_layout.generations / colliding_uid
            collision.mkdir()
            (collision / "marker.txt").write_text("destination", encoding="utf-8")

            with self.assertRaises(StorageMigrationCollisionError) as raised:
                prepare_webui_storage(root)

            self.assertEqual(raised.exception.collisions, (collision,))
            self.assertEqual(
                (first_source / "marker.txt").read_text(encoding="utf-8"),
                "first",
            )
            self.assertFalse(
                os.path.lexists(initial_layout.generations / first_uid)
            )
            self.assertEqual(
                (colliding_source / "marker.txt").read_text(encoding="utf-8"),
                "source",
            )
            self.assertEqual(
                (collision / "marker.txt").read_text(encoding="utf-8"),
                "destination",
            )

    def test_leaves_unknown_and_noncanonical_entries_untouched(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            uppercase_uuid = str(uuid.uuid4()).upper()
            unknown_directory = root / "manual-project"
            uppercase_directory = root / uppercase_uuid
            invalid_uuid_directory = root / "ec72d3f3-e4ea6-9747-ab12cf5e47e1"
            uuid_named_file = root / str(uuid.uuid4())
            unknown_directory.mkdir()
            uppercase_directory.mkdir()
            invalid_uuid_directory.mkdir()
            uuid_named_file.write_text("not a directory", encoding="utf-8")

            layout = prepare_webui_storage(root)

            for untouched in (
                unknown_directory,
                uppercase_directory,
                invalid_uuid_directory,
                uuid_named_file,
            ):
                with self.subTest(path=untouched.name):
                    self.assertTrue(os.path.lexists(untouched))
                    self.assertFalse(
                        os.path.lexists(layout.generations / untouched.name)
                    )

    def test_leaves_symlinked_uuid_directory_untouched(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            outside = Path(temporary_directory) / "outside"
            root.mkdir()
            outside.mkdir()
            generation_uid = str(uuid.uuid4())
            linked_folder = root / generation_uid
            try:
                linked_folder.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlinks are unavailable: {error}")

            layout = prepare_webui_storage(root)

            self.assertTrue(linked_folder.is_symlink())
            self.assertFalse(
                os.path.lexists(layout.generations / generation_uid)
            )

    def test_rejects_symlinked_storage_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            real_root = temporary_root / "real-webui"
            linked_root = temporary_root / "linked-webui"
            real_root.mkdir()
            try:
                linked_root.symlink_to(real_root, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(
                OSError,
                "must not be a link or reparse point",
            ):
                prepare_webui_storage(linked_root)

            self.assertTrue(linked_root.is_symlink())
            self.assertEqual(tuple(real_root.iterdir()), ())

    def test_migrate_legacy_false_preserves_legacy_uuid_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            generation_uid = str(uuid.uuid4())
            legacy_folder = root / generation_uid
            legacy_folder.mkdir()

            layout = prepare_webui_storage(root, migrate_legacy=False)

            self.assertTrue(legacy_folder.is_dir())
            self.assertFalse(
                os.path.lexists(layout.generations / generation_uid)
            )

    def test_malformed_manifest_is_preserved_verbatim(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            generation_uid = str(uuid.uuid4())
            legacy_folder = root / generation_uid
            legacy_folder.mkdir()
            invalid_manifest = b'{"storage_folder": invalid json'
            (legacy_folder / "generation.json").write_bytes(invalid_manifest)

            layout = prepare_webui_storage(root)

            self.assertEqual(
                (layout.generations / generation_uid / "generation.json").read_bytes(),
                invalid_manifest,
            )

    def test_rejects_canonical_layout_path_that_is_a_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "webui"
            root.mkdir()
            (root / "logs").write_text("not a directory", encoding="utf-8")

            with self.assertRaises(NotADirectoryError):
                prepare_webui_storage(root, migrate_legacy=False)


if __name__ == "__main__":
    unittest.main()
