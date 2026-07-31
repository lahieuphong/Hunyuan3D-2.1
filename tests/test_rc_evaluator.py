import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hy3dshape.texture_bake.rc_evaluator import (
    FaceHairRCCandidateEvaluator,
    TEN_VIEW_NAMES,
    resolve_blender_executable,
)


def _reference_image() -> Image.Image:
    image = Image.new("RGBA", (160, 224), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # The upper 38% of the silhouette contains both the cyan hair and face.
    draw.polygon(
        ((48, 40), (58, 8), (70, 34), (82, 5), (91, 36), (112, 18),
         (108, 52), (124, 45), (111, 71), (49, 71), (34, 47)),
        fill=(18, 184, 210, 255),
    )
    draw.ellipse((55, 51, 106, 101), fill=(239, 190, 151, 255))
    draw.line((56, 70, 104, 70), fill=(22, 20, 24, 255), width=3)
    draw.rectangle((70, 94, 91, 119), fill=(239, 190, 151, 255))
    draw.polygon(
        ((39, 107), (121, 107), (136, 212), (24, 212)),
        fill=(235, 99, 22, 255),
    )
    return image


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class RCEvaluatorTests(unittest.TestCase):
    def test_real_adapter_renders_crops_scores_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            candidate = folder / "textured_mesh.ten-view.pending.glb"
            candidate.write_bytes(b"private candidate")
            blender = folder / "blender.exe"
            blender.write_bytes(b"test executable")
            images = {name: _reference_image() for name in TEN_VIEW_NAMES}
            calls: list[list[str]] = []

            def fake_runner(command, **kwargs):
                calls.append(list(command))
                output_directory = Path(_argument(command, "--output-dir"))
                report = Path(_argument(command, "--report"))
                output_directory.mkdir(parents=True, exist_ok=True)
                rendered = {}
                for name in TEN_VIEW_NAMES:
                    source = Path(
                        _argument(command, f"--{name.replace('_', '-')}")
                    )
                    output = output_directory / f"{name}.png"
                    shutil.copyfile(source, output)
                    rendered[name] = {"file": str(output), "size": [160, 224]}
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    json.dumps({"method": "test", "views": rendered}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            evaluator = FaceHairRCCandidateEvaluator(
                blender_path=blender,
                resolution_scale=0.5,
                runner=fake_runner,
            )
            result = evaluator(candidate, TEN_VIEW_NAMES, images)

            self.assertTrue(result.passed_hard_gates)
            self.assertGreater(result.score, 99.0)
            self.assertEqual(len(result.views), 10)
            self.assertEqual(len(calls), 1)
            command = calls[0]
            self.assertIn(
                "render_face_hair_rc_views_blender.py",
                _argument(command, "--python"),
            )
            for name in TEN_VIEW_NAMES:
                self.assertIn(f"--{name.replace('_', '-')}", command)

            audit = (
                folder
                / "quality_audit"
                / "face_hair_rc"
                / "textured_mesh.ten-view.pending"
            )
            self.assertTrue((audit / "evaluation.json").is_file())
            self.assertTrue((audit / "render_report.json").is_file())
            self.assertEqual(
                sorted(path.stem for path in (audit / "renders_head").glob("*.png")),
                sorted(TEN_VIEW_NAMES),
            )
            report = json.loads(
                (audit / "evaluation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["result"]["passed_hard_gates"])
            self.assertEqual(set(report["views"]), set(TEN_VIEW_NAMES))

    def test_all_ten_views_are_required_before_subprocess_runs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            candidate = folder / "candidate.glb"
            candidate.write_bytes(b"candidate")
            blender = folder / "blender.exe"
            blender.write_bytes(b"test executable")
            images = {name: _reference_image() for name in TEN_VIEW_NAMES[:-1]}
            called = False

            def fake_runner(*_args, **_kwargs):
                nonlocal called
                called = True

            evaluator = FaceHairRCCandidateEvaluator(
                blender_path=blender,
                runner=fake_runner,
            )
            with self.assertRaisesRegex(
                ValueError,
                "requires all ten canonical views",
            ):
                evaluator(candidate, TEN_VIEW_NAMES[:-1], images)
            self.assertFalse(called)

    def test_resolver_accepts_an_explicit_file_without_generation_import(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            blender = Path(temporary_directory) / "blender.exe"
            blender.write_bytes(b"test executable")
            self.assertEqual(
                resolve_blender_executable(blender),
                blender.resolve(),
            )

    def test_opaque_reference_is_rejected_instead_of_faking_alignment(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            folder = Path(temporary_directory)
            candidate = folder / "candidate.glb"
            candidate.write_bytes(b"candidate")
            blender = folder / "blender.exe"
            blender.write_bytes(b"test executable")
            opaque = Image.fromarray(
                np.full((64, 64, 4), 255, dtype=np.uint8),
                mode="RGBA",
            )
            evaluator = FaceHairRCCandidateEvaluator(blender_path=blender)
            with self.assertRaisesRegex(ValueError, "transparent background"):
                evaluator(
                    candidate,
                    TEN_VIEW_NAMES,
                    {name: opaque for name in TEN_VIEW_NAMES},
                )


if __name__ == "__main__":
    unittest.main()
