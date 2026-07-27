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
    HardwareProfile,
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
BLACKWELL_ID = "nvidia-rtx-pro-6000-blackwell-workstation-96gb"
BLACKWELL_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


def catalog_payload() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def synthetic_generic_profile(
    source: dict,
    *,
    profile_id: str,
    vram_min_gb: float,
    vram_max_gb: float | None,
) -> dict:
    """Build an unverified generic profile used only by matching tests."""
    profile = copy.deepcopy(source)
    profile.update(
        {
            "id": profile_id,
            "label": f"Synthetic {profile_id}",
            "display_name": "Synthetic NVIDIA GPU",
            "short_label": "Synthetic GPU",
            "vram_min_gb": vram_min_gb,
            "vram_max_gb": vram_max_gb,
            "vram_label": "Synthetic VRAM",
            "aliases": [],
            "examples": ["Synthetic test GPU"],
            "verification": "estimated",
            "verification_label": "Estimated test profile",
            "summary": "Synthetic generic profile for unit tests.",
            "note": "This profile must never be added to the production catalog.",
        }
    )
    for preset in profile["presets"]:
        preset["verified"] = False
    return profile


def catalog_with_synthetic_generic_profiles(
    *ranges: tuple[float, float | None],
):
    payload = catalog_payload()
    verified = next(
        profile
        for profile in payload["hardware"]
        if profile["id"] == "nvidia-rtx-3090-24gb"
    )
    payload["default_hardware_id"] = None
    payload["hardware"] = [verified]
    payload["hardware"].extend(
        synthetic_generic_profile(
            verified,
            profile_id=f"synthetic-generic-{index}",
            vram_min_gb=vram_min,
            vram_max_gb=vram_max,
        )
        for index, (vram_min, vram_max) in enumerate(ranges, start=1)
    )
    return parse_catalog(payload)


def runtime_gpu(
    name: str,
    vram_gb: float,
    *,
    backend: str = "cuda",
    detected: bool = True,
    capability: str = "8.6",
) -> RuntimeHardware:
    return RuntimeHardware(
        requested_device="cuda:0",
        backend=backend,
        index=0,
        name=name,
        total_vram_bytes=round(vram_gb * GIB),
        capability=capability,
        bf16_supported=True,
        dtype="float16",
        detected=detected,
    )


class CatalogValidationTests(unittest.TestCase):
    def test_repository_catalog_is_valid_and_explicit(self) -> None:
        load_gpu_preset_catalog.cache_clear()
        catalog = load_gpu_preset_catalog()

        self.assertEqual(catalog.schema_version, 1)
        self.assertEqual(catalog.default_hardware_id, "nvidia-rtx-3090-24gb")
        self.assertEqual(len(catalog.hardware), 2)
        self.assertEqual(catalog.preset_count, 4)
        for profile in catalog.hardware:
            self.assertEqual({preset.id for preset in profile.presets}, {"safe", "quality"})
            self.assertIn(profile.default_preset_id, {"safe", "quality"})
            self.assertTrue(profile.vram_label)

        verified = [profile.id for profile in catalog.hardware if profile.verification == "verified"]
        self.assertEqual(verified, ["nvidia-rtx-3090-24gb"])
        runtime_verified = [
            profile.id
            for profile in catalog.hardware
            if profile.verification == "runtime-verified"
        ]
        self.assertEqual(runtime_verified, [BLACKWELL_ID])
        verified_profile = catalog.get_hardware("nvidia-rtx-3090-24gb")
        blackwell_profile = catalog.get_hardware(BLACKWELL_ID)
        self.assertIsNotNone(verified_profile)
        self.assertIsNotNone(blackwell_profile)
        assert verified_profile is not None
        assert blackwell_profile is not None
        self.assertEqual(verified_profile.compute_capability, "8.6")
        self.assertEqual(blackwell_profile.compute_capability, "12.0")
        self.assertTrue(
            all(preset.verified for preset in verified_profile.presets)
        )
        self.assertTrue(verified_profile.presets_enabled)
        self.assertTrue(blackwell_profile.presets_enabled)
        self.assertTrue(all(not preset.verified for preset in blackwell_profile.presets))
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

    def test_compute_capability_uses_major_minor_format(self) -> None:
        payload = catalog_payload()
        payload["hardware"][0]["compute_capability"] = "sm_86"
        with self.assertRaisesRegex(ValueError, "compute_capability"):
            parse_catalog(payload)

    def test_verification_label_cannot_overstate_presets(self) -> None:
        payload = catalog_payload()
        payload["hardware"][0]["presets"][0]["verified"] = False
        with self.assertRaisesRegex(ValueError, "every preset must be verified"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["verification"] = "estimated"
        with self.assertRaisesRegex(ValueError, "presets cannot be verified"):
            parse_catalog(payload)

    def test_runtime_verified_profile_requires_an_exact_alias(self) -> None:
        payload = catalog_payload()
        blackwell = next(
            profile for profile in payload["hardware"] if profile["id"] == BLACKWELL_ID
        )
        blackwell["aliases"] = []
        with self.assertRaisesRegex(ValueError, "must define an exact GPU alias"):
            parse_catalog(payload)

    def test_aliases_are_unique_after_exact_name_normalization(self) -> None:
        payload = catalog_payload()
        profile = payload["hardware"][0]
        profile["aliases"].append("nvidia-geforce rtx 3090")
        with self.assertRaisesRegex(ValueError, "duplicate exact GPU names"):
            parse_catalog(payload)

        payload = catalog_payload()
        duplicate = copy.deepcopy(payload["hardware"][0])
        duplicate["id"] = "synthetic-shared-alias"
        duplicate["label"] = "Synthetic shared alias"
        payload["hardware"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "is shared by"):
            parse_catalog(payload)

        payload = catalog_payload()
        payload["hardware"][0]["aliases"] = ["---"]
        with self.assertRaisesRegex(ValueError, "must contain a GPU name"):
            parse_catalog(payload)

    def test_generic_vram_ranges_cannot_overlap(self) -> None:
        payload = catalog_payload()
        verified = payload["hardware"][0]
        payload["hardware"].extend(
            (
                synthetic_generic_profile(
                    verified,
                    profile_id="synthetic-generic-one",
                    vram_min_gb=8,
                    vram_max_gb=16,
                ),
                synthetic_generic_profile(
                    verified,
                    profile_id="synthetic-generic-two",
                    vram_min_gb=15,
                    vram_max_gb=24,
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "overlapping generic VRAM ranges"):
            parse_catalog(payload)

    def test_generic_vram_ranges_may_overlap_across_capabilities(self) -> None:
        payload = catalog_payload()
        verified = payload["hardware"][0]
        first = synthetic_generic_profile(
            verified,
            profile_id="synthetic-generic-cc86",
            vram_min_gb=8,
            vram_max_gb=24,
        )
        second = synthetic_generic_profile(
            verified,
            profile_id="synthetic-generic-cc120",
            vram_min_gb=8,
            vram_max_gb=24,
        )
        second["compute_capability"] = "12.0"
        payload["hardware"].extend((first, second))

        catalog = parse_catalog(payload)

        self.assertIsNotNone(catalog.get_hardware("synthetic-generic-cc86"))
        self.assertIsNotNone(catalog.get_hardware("synthetic-generic-cc120"))


class RuntimeMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = parse_catalog(catalog_payload())
        cls.generic_catalog = catalog_with_synthetic_generic_profiles((7.5, 28))

    def test_rtx_3090_uses_verified_exact_alias(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090", 24),
            self.catalog,
        )
        self.assertEqual(match, HardwareMatch("nvidia-rtx-3090-24gb", "exact", True))

    def test_blackwell_workstation_uses_exact_runtime_verified_profile(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu(
                BLACKWELL_NAME,
                97887 / 1024,
                capability="12.0",
            ),
            self.catalog,
        )
        self.assertEqual(match, HardwareMatch(BLACKWELL_ID, "exact", True))

    def test_blackwell_accepts_both_vram_boundaries_inside_half_open_range(self) -> None:
        for vram_gib in (95.0, 96.999):
            with self.subTest(vram_gib=vram_gib):
                match = match_runtime_hardware(
                    runtime_gpu(BLACKWELL_NAME, vram_gib, capability="12.0"),
                    self.catalog,
                )
                self.assertEqual(
                    match,
                    HardwareMatch(BLACKWELL_ID, "exact", True),
                )

    def test_blackwell_match_rejects_other_names_vram_backend_and_dtype(self) -> None:
        measured_vram_gib = 97887 / 1024
        invalid_runtimes = (
            runtime_gpu(
                "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                measured_vram_gib,
                capability="12.0",
            ),
            runtime_gpu(
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                measured_vram_gib,
                capability="12.0",
            ),
            runtime_gpu(BLACKWELL_NAME, 94.99, capability="12.0"),
            runtime_gpu(BLACKWELL_NAME, 97.0, capability="12.0"),
            runtime_gpu(
                BLACKWELL_NAME,
                measured_vram_gib,
                backend="rocm",
                capability="12.0",
            ),
            replace(
                runtime_gpu(BLACKWELL_NAME, measured_vram_gib, capability="12.0"),
                dtype="bfloat16",
            ),
            runtime_gpu(BLACKWELL_NAME, measured_vram_gib, capability="8.6"),
        )
        for runtime in invalid_runtimes:
            with self.subTest(runtime=runtime):
                self.assertEqual(
                    match_runtime_hardware(runtime, self.catalog),
                    HardwareMatch(None, "unavailable", False),
                )

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
        self.assertEqual(ti_match, HardwareMatch(None, "unavailable", False))

    def test_alias_with_wrong_vram_is_unavailable_in_repository_catalog(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090", 16),
            self.catalog,
        )
        self.assertEqual(match, HardwareMatch(None, "unavailable", False))

    def test_backend_must_match_profile_backend(self) -> None:
        match = match_runtime_hardware(
            runtime_gpu("NVIDIA GeForce RTX 3090", 24, backend="rocm"),
            self.catalog,
        )
        self.assertEqual(match, HardwareMatch(None, "unavailable", False))

    def test_verified_profile_requires_matching_dtype(self) -> None:
        runtime = replace(
            runtime_gpu("NVIDIA GeForce RTX 3090", 24),
            dtype="float32",
        )
        self.assertEqual(
            match_runtime_hardware(runtime, self.catalog),
            HardwareMatch(None, "unavailable", False),
        )

    def test_invalid_runtime_memory_is_unavailable(self) -> None:
        invalid = replace(runtime_gpu("GPU", 8), total_vram_bytes=0)
        self.assertEqual(
            match_runtime_hardware(invalid, self.catalog),
            HardwareMatch(None, "unavailable", False),
        )

    def test_unlisted_gpu_is_unavailable_in_repository_catalog(self) -> None:
        for vram_gb in (4, 6):
            with self.subTest(vram_gb=vram_gb):
                match = match_runtime_hardware(
                    runtime_gpu("Unlisted NVIDIA GPU", vram_gb),
                    self.catalog,
                )
                self.assertEqual(match, HardwareMatch(None, "unavailable", False))

    def test_synthetic_generic_profile_covers_vram_and_nearest_matching(self) -> None:
        vram_match = match_runtime_hardware(
            runtime_gpu("Unlisted NVIDIA GPU", 16),
            self.generic_catalog,
        )
        self.assertEqual(
            vram_match,
            HardwareMatch("synthetic-generic-1", "vram", True),
        )

        nearest_match = match_runtime_hardware(
            runtime_gpu("Unlisted NVIDIA GPU", 6),
            self.generic_catalog,
        )
        self.assertEqual(
            nearest_match,
            HardwareMatch("synthetic-generic-1", "nearest", False),
        )

        wrong_capability = match_runtime_hardware(
            runtime_gpu("Unlisted NVIDIA GPU", 16, capability="12.0"),
            self.generic_catalog,
        )
        self.assertEqual(
            wrong_capability,
            HardwareMatch(None, "unavailable", False),
        )

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
        assert unavailable.error is not None
        self.assertIn("unavailable", unavailable.error.lower())

        invalid_memory = detect_runtime_hardware(
            _FakeTorch(_FakeCuda(total_memory=0)),
            "cuda:0",
            "float16",
        )
        self.assertFalse(invalid_memory.detected)
        assert invalid_memory.error is not None
        self.assertIn("invalid total memory", invalid_memory.error.lower())


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = parse_catalog(catalog_payload())
        verified = cls.catalog.get_hardware("nvidia-rtx-3090-24gb")
        blackwell = cls.catalog.get_hardware(BLACKWELL_ID)
        cls.generic_catalog = catalog_with_synthetic_generic_profiles((7.5, 28))
        estimated = cls.generic_catalog.get_hardware("synthetic-generic-1")
        if verified is None or blackwell is None or estimated is None:
            raise AssertionError("Expected test hardware profiles were not created")
        cls.verified: HardwareProfile = verified
        cls.blackwell: HardwareProfile = blackwell
        cls.estimated: HardwareProfile = estimated

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
            self.generic_catalog,
        )
        self.assertIn("Cấu hình gần nhất", nearest)
        self.assertIn("chọn thủ công", nearest)

    def test_blackwell_intro_displays_measured_runtime_without_claiming_benchmark(self) -> None:
        runtime = runtime_gpu(
            BLACKWELL_NAME,
            97887 / 1024,
            capability="12.0",
        )
        rendered = render_catalog_intro(
            runtime,
            HardwareMatch(BLACKWELL_ID, "exact", True),
            self.catalog,
        )

        self.assertIn("95.59 GiB VRAM", rendered)
        self.assertIn("CC 12.0", rendered)
        self.assertIn("2 cấu hình", rendered)
        self.assertIn("ứng viên chờ benchmark", rendered)
        self.assertIn("is-runtime-verified", rendered)
        self.assertNotIn("Catalog hiện chỉ giữ cấu hình RTX 3090", rendered)

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
        self.assertIn(self.estimated.verification_label, rendered)

    def test_preset_cards_expose_stable_hardware_and_preset_ids(self) -> None:
        rendered = render_preset_cards(self.verified, "quality")
        self.assertIn('data-hardware-id="nvidia-rtx-3090-24gb"', rendered)
        self.assertIn('data-profile="safe"', rendered)
        self.assertIn('data-profile="quality"', rendered)
        self.assertEqual(rendered.count('aria-pressed="true"'), 1)


if __name__ == "__main__":
    unittest.main()
