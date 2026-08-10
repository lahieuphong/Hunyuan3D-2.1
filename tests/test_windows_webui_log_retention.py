import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "hy3dshape" / "scripts" / "start_windows_multiview_webui.ps1"


def _extract_powershell_function(source: str, function_name: str) -> str:
    marker = f"function {function_name} {{"
    start = source.index(marker)
    brace_start = source.index("{", start)
    depth = 0
    for index in range(brace_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Unterminated PowerShell function: {function_name}")


class WindowsWebUiLogRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text(encoding="utf-8-sig")

    def test_retention_parameter_and_ready_only_call(self):
        self.assertIn("[ValidateRange(1, 100)]", self.source)
        self.assertIn("[int]$LogRetentionLaunches = 5", self.source)

        call = (
            "Remove-StaleWebUiLaunchLogs `\n"
            "            -CurrentStdoutLog $StdoutLog `\n"
            "            -CurrentStderrLog $StderrLog"
        )
        self.assertEqual(self.source.count(call), 1)
        call_index = self.source.index(call)
        background_start_index = self.source.index("$process = Start-Process")
        ready_branch_index = self.source.rfind(
            'if ($null -ne $health -and $health.status -eq "ready") {',
            background_start_index,
            call_index,
        )
        self.assertGreater(ready_branch_index, background_start_index)
        self.assertGreater(call_index, ready_branch_index)

    def test_retention_scope_and_current_log_guards_are_explicit(self):
        function_source = _extract_powershell_function(
            self.source, "Remove-StaleWebUiLaunchLogs"
        )
        self.assertIn(
            "'^webui_(\\d{8}_\\d{6})\\.(stdout|stderr)\\.log$'",
            function_source,
        )
        self.assertIn("Get-ChildItem -LiteralPath $LogsDir -File", function_source)
        self.assertIn("$currentPaths.Contains($fullPath)", function_source)
        self.assertIn("$keepLaunchIds.Count -ge $LogRetentionLaunches", function_source)

    def test_retention_keeps_current_launch_even_when_it_is_oldest(self):
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell is unavailable")

        function_source = _extract_powershell_function(
            self.source, "Remove-StaleWebUiLaunchLogs"
        )
        launch_ids = [f"20260101_00000{index}" for index in range(1, 7)]
        current_launch = launch_ids[0]

        logs_dir = REPO_ROOT / f".tmp-webui-log-retention-{uuid.uuid4().hex}"
        logs_dir.mkdir()
        try:
            for launch_id in launch_ids:
                (logs_dir / f"webui_{launch_id}.stdout.log").write_text(
                    "stdout", encoding="utf-8"
                )
                (logs_dir / f"webui_{launch_id}.stderr.log").write_text(
                    "stderr", encoding="utf-8"
                )
            unrelated = logs_dir / "restart_20260101_000000.stdout.log"
            unrelated.write_text("keep", encoding="utf-8")

            escaped_logs_dir = str(logs_dir).replace("'", "''")
            current_stdout = str(
                logs_dir / f"webui_{current_launch}.stdout.log"
            ).replace("'", "''")
            current_stderr = str(
                logs_dir / f"webui_{current_launch}.stderr.log"
            ).replace("'", "''")
            harness = f"""
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$LogsDir = '{escaped_logs_dir}'
$LogRetentionLaunches = 3
{function_source}
Remove-StaleWebUiLaunchLogs -CurrentStdoutLog '{current_stdout}' -CurrentStderrLog '{current_stderr}'
"""
            completed = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    harness,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

            remaining_launches = {
                path.name.split(".", 1)[0].removeprefix("webui_")
                for path in logs_dir.glob("webui_*.log")
            }
            self.assertEqual(
                remaining_launches,
                {current_launch, launch_ids[-1], launch_ids[-2]},
            )
            self.assertTrue(
                (logs_dir / f"webui_{current_launch}.stdout.log").is_file()
            )
            self.assertTrue(
                (logs_dir / f"webui_{current_launch}.stderr.log").is_file()
            )
            self.assertTrue(unrelated.is_file())
        finally:
            shutil.rmtree(logs_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
