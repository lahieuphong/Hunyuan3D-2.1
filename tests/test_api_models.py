from __future__ import annotations

import unittest

from api_models import GenerationRequest


class GenerationRequestSchemaTests(unittest.TestCase):
    def test_image_is_required_and_preserves_the_openapi_example(self):
        schema = GenerationRequest.model_json_schema()
        image_schema = schema["properties"]["image"]

        self.assertIn("image", schema["required"])
        self.assertEqual(
            image_schema["example"],
            "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP8z4AATAxEcQAz0QEHOoQ+uAAAAABJRU5ErkJggg==",
        )
        self.assertNotIn("examples", image_schema)

    def test_generation_defaults_remain_unchanged(self):
        request = GenerationRequest.model_validate({"image": "aGVsbG8="})

        self.assertTrue(request.remove_background)
        self.assertFalse(request.texture)
        self.assertEqual(request.seed, 1234)
        self.assertEqual(request.octree_resolution, 256)
        self.assertEqual(request.num_inference_steps, 5)
        self.assertEqual(request.guidance_scale, 5.0)
        self.assertEqual(request.num_chunks, 8000)
        self.assertEqual(request.face_count, 40000)


if __name__ == "__main__":
    unittest.main()
