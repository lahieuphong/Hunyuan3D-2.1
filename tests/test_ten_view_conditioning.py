from __future__ import annotations

import unittest

import torch
from PIL import Image

from torchvision_fix import apply_fix

from hy3dshape.models.feature_fusion import fuse_multiview_features
from hy3dshape.preprocessors import MVImageProcessorV2
from hy3dshape.ten_view import (
    CANONICAL_VIEW_KEYS,
    TEN_VIEW_AUXILIARY_KEYS,
    TEN_VIEW_KEYS,
    normalized_ten_view_blend_weights,
)
from webui.generation_inputs import (
    GenerationInputError,
    build_generation_input_bundle,
    ordered_ten_view_images,
)


def _opaque_image(value: int = 128) -> Image.Image:
    return Image.new("RGBA", (32, 48), (value, value, value, 255))


class TenViewInputContractTests(unittest.TestCase):
    def test_requires_all_ten_images_in_semantic_order(self):
        images = [_opaque_image(index) for index in range(10)]
        mapping = ordered_ten_view_images(images)
        bundle = build_generation_input_bundle(
            "ten",
            None,
            {},
            mapping,
        )

        self.assertEqual(tuple(mapping), TEN_VIEW_KEYS)
        self.assertEqual(tuple(bundle.provided_images), TEN_VIEW_KEYS)
        self.assertEqual(tuple(bundle.conditioning_images), CANONICAL_VIEW_KEYS)
        self.assertEqual(
            list(bundle.conditioning_images.values()),
            [mapping[key] for key in CANONICAL_VIEW_KEYS],
        )
        self.assertEqual(bundle.primary_image, images[0])
        self.assertEqual(bundle.metadata["conditioned_view_count"], 4)
        self.assertEqual(
            bundle.metadata["views_used"],
            list(CANONICAL_VIEW_KEYS),
        )
        self.assertEqual(
            bundle.metadata["texture_rc_views"],
            list(TEN_VIEW_KEYS),
        )
        self.assertEqual(
            bundle.metadata["auxiliary_views_reserved_for_texture_rc"],
            list(TEN_VIEW_AUXILIARY_KEYS),
        )
        self.assertFalse(bundle.metadata["experimental_conditioning"])

    def test_diagonal_and_high_views_never_condition_the_shape_model(self):
        mapping = {key: _opaque_image(index) for index, key in enumerate(TEN_VIEW_KEYS)}
        bundle = build_generation_input_bundle("ten", None, {}, mapping)

        self.assertTrue(set(bundle.conditioning_images).isdisjoint(TEN_VIEW_AUXILIARY_KEYS))

    def test_reports_the_exact_missing_camera(self):
        mapping = {key: _opaque_image() for key in TEN_VIEW_KEYS if key != "high_back"}
        with self.assertRaisesRegex(GenerationInputError, "High Back"):
            build_generation_input_bundle("ten", None, {}, mapping)

    def test_single_and_four_modes_keep_their_native_contract(self):
        front = _opaque_image()
        single = build_generation_input_bundle("single", front, {})
        four_images = {
            key: _opaque_image(index) for index, key in enumerate(CANONICAL_VIEW_KEYS)
        }
        four = build_generation_input_bundle(
            "four",
            None,
            four_images,
        )

        self.assertEqual(single.metadata["conditioning_strategy"], "native-single-view")
        self.assertEqual(tuple(four.conditioning_images), CANONICAL_VIEW_KEYS)
        self.assertEqual(four.metadata["conditioning_strategy"], "native-cardinal-4")


class TenViewPreprocessorTests(unittest.TestCase):
    @staticmethod
    def _processor_with_fake_loader() -> MVImageProcessorV2:
        processor = MVImageProcessorV2(size=2)

        def fake_load(value, border_ratio=0.15, to_tensor=True):
            image = torch.full((1, 3, 2, 2), float(value))
            mask = torch.ones((1, 1, 2, 2))
            return image, mask

        processor.load_image = fake_load
        return processor

    def test_ten_views_emit_four_native_indices_and_blend_matrix(self):
        processor = self._processor_with_fake_loader()
        inputs = {key: index for index, key in enumerate(reversed(TEN_VIEW_KEYS))}
        outputs = processor(inputs)

        self.assertEqual(outputs["image"].shape, (1, 10, 3, 2, 2))
        self.assertEqual(outputs["mask"].shape, (1, 10, 1, 2, 2))
        self.assertEqual(outputs["view_idxs"], (0, 1, 2, 3))
        self.assertEqual(outputs["view_blend_weights"].shape, (1, 10, 4))
        self.assertTrue(
            torch.allclose(
                outputs["view_blend_weights"].sum(dim=1),
                torch.ones((1, 4)),
            )
        )
        self.assertEqual(
            outputs["image"][0, :, 0, 0, 0].tolist(),
            [float(inputs[key]) for key in TEN_VIEW_KEYS],
        )

    def test_four_view_sorting_is_unchanged(self):
        processor = self._processor_with_fake_loader()
        outputs = processor(
            {
                "right": 3,
                "front": 0,
                "back": 2,
                "left": 1,
            }
        )

        self.assertEqual(outputs["view_idxs"], (0, 1, 2, 3))
        self.assertNotIn("view_blend_weights", outputs)
        self.assertEqual(
            outputs["image"][0, :, 0, 0, 0].tolist(),
            [0.0, 1.0, 2.0, 3.0],
        )


class FeatureFusionTests(unittest.TestCase):
    def test_normalized_contract_has_unit_weight_per_output_slot(self):
        weights = torch.tensor(normalized_ten_view_blend_weights())
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(4)))

    def test_fusion_preserves_native_four_view_context(self):
        features = torch.arange(
            1 * 10 * 3 * 2,
            dtype=torch.float32,
        ).reshape(1, 10, 3, 2)
        weights = torch.tensor(
            normalized_ten_view_blend_weights(),
            dtype=torch.float32,
        ).unsqueeze(0)
        fused = fuse_multiview_features(features, weights)

        expected = torch.einsum("bvph,bvo->boph", features, weights)
        self.assertEqual(fused.shape, (1, 4, 3, 2))
        self.assertTrue(torch.allclose(fused, expected))

    def test_fusion_rejects_a_mismatched_source_view_count(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            fuse_multiview_features(
                torch.zeros((1, 10, 3, 2)),
                torch.zeros((1, 9, 4)),
            )


class TenViewEncoderTests(unittest.TestCase):
    def test_ten_sources_flatten_to_the_same_context_as_native_four(self):
        apply_fix()
        from hy3dshape.models.conditioner import DinoImageEncoderMV

        encoder = DinoImageEncoderMV(
            config={
                "hidden_size": 8,
                "image_size": 28,
                "patch_size": 14,
                "num_channels": 3,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "mlp_ratio": 2,
                "use_swiglu_ffn": False,
            },
            image_size=28,
            view_num=4,
        )
        view_indices = [(0, 1, 2, 3)]
        native = encoder(
            torch.zeros((1, 4, 3, 28, 28)),
            view_idxs=view_indices,
        )
        fused = encoder(
            torch.zeros((1, 10, 3, 28, 28)),
            view_idxs=view_indices,
            view_blend_weights=torch.tensor(
                normalized_ten_view_blend_weights(),
                dtype=torch.float32,
            ).unsqueeze(0),
        )

        self.assertEqual(native.shape, (1, 20, 8))
        self.assertEqual(fused.shape, native.shape)


if __name__ == "__main__":
    unittest.main()
