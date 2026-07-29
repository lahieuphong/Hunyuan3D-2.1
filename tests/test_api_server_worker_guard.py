from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_SERVER_PATH = REPOSITORY_ROOT / "api_server.py"


class _StubModelWorker:
    pass


class _FakeWorker:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.calls: list[tuple[object, dict[str, object]]] = []

    def generate(
        self,
        uid: object,
        params: dict[str, object],
    ) -> tuple[str, object]:
        self.calls.append((uid, params))
        return str(self.output_path), uid


class _InlineThread:
    instances: list["_InlineThread"] = []

    def __init__(self, *, target, args) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True
        self.target(*self.args)


def _load_api_server_for_test():
    model_worker_stub = ModuleType("model_worker")
    model_worker_stub.ModelWorker = _StubModelWorker

    logger_utils_stub = ModuleType("logger_utils")
    logger_utils_stub.build_logger = lambda *_args, **_kwargs: logging.getLogger(
        "api-server-worker-guard-test"
    )

    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(CudaError=RuntimeError)

    module_name = "_api_server_worker_guard_test"
    spec = importlib.util.spec_from_file_location(module_name, API_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load api_server.py for testing")

    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "logger_utils": logger_utils_stub,
            "model_worker": model_worker_stub,
            "torch": torch_stub,
        },
    ):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class ApiServerWorkerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api_server = _load_api_server_for_test()
        cls.client = TestClient(cls.api_server.app)
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.output_path = Path(cls.temp_directory.name) / "generated.glb"
        cls.output_bytes = b"glTF-test-payload"
        cls.output_path.write_bytes(cls.output_bytes)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.temp_directory.cleanup()
        sys.modules.pop(cls.api_server.__name__, None)

    def setUp(self) -> None:
        self.api_server.worker = None
        _InlineThread.instances.clear()

    def test_generation_endpoints_return_503_when_worker_is_missing(self):
        payload = {"image": "AA=="}

        generate_response = self.client.post("/generate", json=payload)
        send_response = self.client.post("/send", json=payload)

        self.assertEqual(generate_response.status_code, 503)
        self.assertEqual(send_response.status_code, 503)
        self.assertEqual(
            generate_response.json(),
            {"detail": "Model worker is not initialized"},
        )
        self.assertEqual(
            send_response.json(),
            {"detail": "Model worker is not initialized"},
        )

    def test_generate_uses_initialized_worker(self):
        fake_worker = _FakeWorker(self.output_path)
        self.api_server.worker = fake_worker

        response = self.client.post(
            "/generate",
            json={"image": "AA==", "seed": 77},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.output_bytes)
        self.assertEqual(len(fake_worker.calls), 1)
        self.assertEqual(fake_worker.calls[0][1]["image"], "AA==")
        self.assertEqual(fake_worker.calls[0][1]["seed"], 77)

    def test_send_binds_initialized_worker_to_background_thread(self):
        fake_worker = _FakeWorker(self.output_path)
        self.api_server.worker = fake_worker

        with patch.object(self.api_server.threading, "Thread", _InlineThread):
            response = self.client.post(
                "/send",
                json={"image": "AA==", "seed": 91},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(_InlineThread.instances), 1)
        self.assertTrue(_InlineThread.instances[0].started)
        self.assertEqual(len(fake_worker.calls), 1)
        self.assertEqual(str(fake_worker.calls[0][0]), response.json()["uid"])
        self.assertEqual(fake_worker.calls[0][1]["seed"], 91)

    def test_health_reports_worker_readiness(self):
        unavailable_response = self.client.get("/health")

        self.api_server.worker = _FakeWorker(self.output_path)
        healthy_response = self.client.get("/health")

        self.assertEqual(unavailable_response.status_code, 503)
        self.assertEqual(unavailable_response.json()["status"], "unavailable")
        self.assertEqual(healthy_response.status_code, 200)
        self.assertEqual(healthy_response.json()["status"], "healthy")

    def test_source_never_dereferences_generate_on_optional_global(self):
        tree = ast.parse(API_SERVER_PATH.read_text(encoding="utf-8"))
        direct_optional_accesses = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "generate"
                and isinstance(node.value, ast.Name)
                and node.value.id == "worker"
            )
        ]

        self.assertEqual(direct_optional_accesses, [])


if __name__ == "__main__":
    unittest.main()
