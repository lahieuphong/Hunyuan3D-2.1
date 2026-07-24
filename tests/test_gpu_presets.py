from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from webui.gpu_presets import (
    CATALOG_PATH,
    HardwareMatch,
    RuntimeHardware,
    detect_runtime_hardware,
    load_gpu_preset_catalog,
    match_runtime_hardware,
    normalize_control_tuple,
    parse_catalog,
    resolve_preset_id,
)
from webui.hardware_templates import (
    render_catalog_intro,
    render_preset_cards,
    render_profile_summary,
)


GIB = 1024**3


def catalog_payload() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def runtime_gpu(
    name: str,
    vram_gb: float,
    *,
    backend: str = "cuda",
    detected: bool = True,
) -> RuntimeHardware:
    return RuntimeHardware(
        requested_device="cuda:0",
        backend=backend,
        index=0,
        name=name,
        total_vram_bytes=round(vram_gb * GIB),
        capability="8.6",
        bf16_supported=True,
        dtype="float16",
        detected=detected,
    )


class CatalogValidationTests(unittest.TestCase):
    def test_repository_catalog_is_valid_and_explicit(self) -> None:
        load_gpu_preset_catalog.cache_clear()
        catalog = load_gpu_preset_catalog()

        self.assertEqual(catalog.schema_version, 1)
        self.assertEqual(len(catalog.hardware), 6)
        self.assertEqual(catalog.preset_count, 12)
        for profile in catalog.hardware:
            self.assertEqual({preset.id for preset in profile.presets}, {"safe", "quality"})
            self.assertIn(profile.default_preset_id, {"safe", "quality"})
            self.assertTrue(profile.vram_label)

        verified = [profile.id for profile in catalog.hardware if profile.verification == "verified"]
        self.assertEqual(verified, ["nvidia-rtx-3090-24gb"])
        self.assertTrue(
            all(
                preset.verified
                for preset in catalog.get_hardware("nvidia-rtx-3090-24gb").presets
            )
        )
        self.assertTrue(
            all(
                not preset.verified
                for profile in catalog.hardware
                if profile.verification != "verified"
                for preset in profile.presets
            )
        )

    def test_schema_version_rejects_boolean(self) -> None:
        payload = catalog_payload()
        payload["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse_catalog(payload)

        payload["schema_version"] = 1.0
        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse_catalog(payload)

    def test_profile_requires_exact_safe_and_quality_contract(self) -> None:
        payload = catalog_payload()
        payload["hardware"][0]["presets"] = payload["hardware"][0]["presets"][1:]
        with self.assertRaisesRegex(ValueError, "exactly: quality, safe"):
            parse_catalog(payload)

        payload = catalog_payload()
        extra = copy.deepcopy(payload["hardware"][0]["presets"][0])
        extra["id"] = "balanced"
        extra["tone"] = "balanced"
        extra["params"]["steps"] = 29
        payload["hardware"][0]["presets"].append(extra)
        with self.assertRaisesRegex(ValueError, "exactly: quality, safe"):
            parse_catalog(payload)

    def test_default_and_tone_must_be_consistent(self) -> None:
        payload = catalog_payload()
        payload["hardware"][0]["default_preset_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "default_preset_id does not exist"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["presets"][0]["tone"] = "quality"
        with self.assertRaisesRegex(ValueError, "safe.tone must be 'safe'"):
            parse_catalog(payload)

    def test_non_finite_and_out_of_range_values_are_rejected(self) -> None:
        payload = catalog_payload()
        payload["hardware"][0]["presets"][0]["params"]["guidance_scale"] = math.nan
        with self.assertRaisesRegex(ValueError, "guidance_scale"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["vram_min_gb"] = math.inf
        with self.assertRaisesRegex(ValueError, "vram_min_gb"):
            parse_catalog(payload)

    def test_verification_label_cannot_overstate_presets(self) -> None:
        payload = catalog_payload()
        payload["hardware"][0]["verification"] = "verified"
        payload["hardware"][0]["verification_label"] = "Verified"
        with self.assertRaisesRegex(ValueError, "every preset must be verified"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["presets"][0]["verified"] = True
        with self.assertRaisesRegex(ValueError, "presets cannot be verified"):
            parse_catalog(payload)

    def test_aliases_are_unique_after_exact_name_normalization(self) -> None:
        payload = catalog_payload()
        profile = payload["hardware"][3]
        profile["aliases"].append("nvidia-geforce rtx 3090")
        with self.assertRaisesRegex(ValueError, "duplicate exact GPU names"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["aliases"] = ["NVIDIA GeForce RTX 3090"]
        with self.assertRaisesRegex(ValueError, "is shared by"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["aliases"] = ["---"]
        with self.assertRaisesRegex(ValueError, "must contain a GPU name"):
            parse_catalog(payload)

    def test_generic_vram_ranges_cannot_overlap(self) -> None:
        payload = catalog_payload()
        payload["hardware"][1]["vram_min_gb"] = 9.8
        with self.assertRaisesRegex(ValueError, "overlapping generic VRAM ranges"):
            parse_catalog(payload)


class RuntimeMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = parse_catalog(catalog_payload())

    def test_rtx_3090_uses_verified_exact_alias(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090", 24),
            self.catalog,
        )
        self.assertEqual(match, HardwareMatch("nvidia-rtx-3090-24gb", "exact", True))

    def test_exact_alias_normalization_does_not_become_substring_matching(self) -> None:
        normalized_match = match_runtime_hardware(
            runtime_gpu("nvidia-geforce rtx 3090", 24),
            self.catalog,
        )
        self.assertEqual(normalized_match.method, "exact")

        ti_match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090 Ti", 24),
            self.catalog,
        )
        self.assertEqual(ti_match.hardware_id, "nvidia-24gb")
        self.assertEqual(ti_match.method, "vram")

    def test_alias_with_wrong_vram_falls_back_to_generic_profile(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090", 16),
            self.catalog,
        )
        self.assertEqual(match.hardware_id, "nvidia-16gb")
        self.assertEqual(match.method, "vram")

    def test_backend_must_match_profile_backend(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090", 24, backend="rocm"),
            self.catalog,
        )
        self.assertEqual(match, HardwareMatch(None, "unavailable", False))

    def test_invalid_runtime_memory_is_unavailable(self) -> None:
        invalid = replace(runtime_gpu("GPU", 8), total_vram_bytes=0)
        self.assertEqual(
            match_runtime_hardware(invalid, self.catalog),
            HardwareMatch(None, "unavailable", False),
        )

    def test_sub_8gb_gpu_is_only_a_nearest_experimental_match(self) -> None:
        for vram_gb in (4, 6):
            with self.subTest(vram_gb=vram_gb):
                match = match_runtime_hardware(
                    runtime_gpu("Unlisted NVIDIA GPU", vram_gb),
                    self.catalog,
                )
                self.assertEqual(match.hardware_id, "nvidia-8gb-experimental")
                self.assertEqual(match.method, "nearest")
                self.assertFalse(match.compatible)

    def test_preset_resolution_requires_finite_in_range_exact_values(self) -> None:
        hardware_id = "nvidia-rtx-3090-24gb"
        self.assertEqual(
            resolve_preset_id(hardware_id, 30, 5.0, 384, 8000, self.catalog),
            "quality",
        )
        self.assertIsNone(
            resolve_preset_id(hardware_id, 30, math.inf, 384, 8000, self.catalog)
        )
        self.assertIsNone(
            resolve_preset_id(hardware_id, 0, 5.0, 384, 8000, self.catalog)
        )
        self.assertIsNone(normalize_control_tuple(True, 5.0, 384, 8000))


class _FakeDevice:
    def __init__(self, requested: str) -> None:
        parts = requested.split(":", 1)
        self.type = parts[0]
        self.index = int(parts[1]) if len(parts) == 2 else None


class _FakeCuda:
    def __init__(self, *, available: bool = True, total_memory: int = 24 * GIB) -> None:
        self.available = available
        self.total_memory = total_memory

    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        return 1

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=self.total_memory, name="Fallback GPU")

    def get_device_name(self, index: int) -> str:
        return "NVIDIA GeForce RTX 3090"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (8, 6)

    def is_bf16_supported(self) -> bool:
        return True


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda

    @staticmethod
    def device(requested: str) -> _FakeDevice:
        return _FakeDevice(requested)


class RuntimeDetectionTests(unittest.TestCase):
    def test_detects_cuda_without_requiring_torch_version_attribute(self) -> None:
        runtime = detect_runtime_hardware(_FakeTorch(_FakeCuda()), "cuda", "float16")
        self.assertTrue(runtime.detected)
        self.assertEqual(runtime.backend, "cuda")
        self.assertEqual(runtime.index, 1)
        self.assertEqual(runtime.name, "NVIDIA GeForce RTX 3090")
        self.assertEqual(runtime.total_vram_bytes, 24 * GIB)
        self.assertEqual(runtime.capability, "8.6")

    def test_non_cuda_device_is_detected_but_has_no_gpu_memory(self) -> None:
        runtime = detect_runtime_hardware(_FakeTorch(_FakeCuda()), "cpu", "float32")
        self.assertTrue(runtime.detected)
        self.assertEqual(runtime.backend, "cpu")
        self.assertIsNone(runtime.total_vram_bytes)

    def test_unavailable_or_invalid_cuda_is_reported_without_raising(self) -> None:
        unavailable = detect_runtime_hardware(
            _FakeTorch(_FakeCuda(available=False)),
            "cuda:0",
            "float16",
        )
        self.assertFalse(unavailable.detected)
        self.assertIn("unavailable", unavailable.error.lower())

        invalid_memory = detect_runtime_hardware(
            _FakeTorch(_FakeCuda(total_memory=0)),
            "cuda:0",
            "float16",
        )
        self.assertFalse(invalid_memory.detected)
        self.assertIn("invalid total memory", invalid_memory.error.lower())


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = parse_catalog(catalog_payload())
        cls.verified = cls.catalog.get_hardware("nvidia-rtx-3090-24gb")
        cls.estimated = cls.catalog.get_hardware("nvidia-16gb")

    def test_catalog_intro_distinguishes_compatible_from_nearest(self) -> None:
        runtime = runtime_gpu("NVIDIA GeForce RTX 3090", 24)
        compatible = render_catalog_intro(
            runtime,
            HardwareMatch(self.verified.id, "exact", True),
            self.catalog,
        )
        self.assertIn("Tự đề xuất", compatible)
        self.assertNotIn("Cấu hình gần nhất", compatible)

        nearest = render_catalog_intro(
            runtime,
            HardwareMatch(self.estimated.id, "nearest", False),
            self.catalog,
        )
        self.assertIn("Cấu hình gần nhất", nearest)
        self.assertIn("chọn thủ công", nearest)

    def test_profile_summary_uses_catalog_vram_label_and_escapes_text(self) -> None:
        profile = replace(
            self.estimated,
            display_name="<script>alert(1)</script>",
            vram_label="VRAM tùy chỉnh",
        )
        rendered = render_profile_summary(
            profile,
            recommended_hardware_id=profile.id,
        )
        self.assertIn("VRAM tùy chỉnh", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("Đề xuất theo VRAM", rendered)

    def test_preset_cards_expose_stable_hardware_and_preset_ids(self) -> None:
        rendered = render_preset_cards(self.verified, "quality")
        self.assertIn('data-hardware-id="nvidia-rtx-3090-24gb"', rendered)
        self.assertIn('data-profile="safe"', rendered)
        self.assertIn('data-profile="quality"', rendered)
        self.assertEqual(rendered.count('aria-pressed="true"'), 1)


if __name__ == "__main__":
    unittest.main()
