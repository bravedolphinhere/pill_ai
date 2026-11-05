"""사용자 불확실성이 정답 확정으로 바뀌지 않는지 작은 입력으로 검사한다."""

from __future__ import annotations

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT
from tamgenie.classifier import Classification, UnavailableClassifier
from tamgenie.detector import Detection
from tamgenie.pipeline import annotate, crop_image, overlap_flags, read_image, route
from tamgenie.question_engine import Product, filter_candidates, load_catalog, next_question
from tamgenie.settings import Settings


class UncertaintyTests(unittest.TestCase):
    """가상 제품은 테스트에서만 쓰고 실제 카탈로그에 추가하지 않는다."""

    def setUp(self) -> None:
        self.catalog = {
            0: Product(0, "TEST-A", "가상 제품 A", {"shape": {"values": ["원형"]}}),
            1: Product(1, "TEST-B", "가상 제품 B", {"shape": {"values": ["타원형"]}}),
            2: Product(2, "TEST-C", "정보 없는 가상 제품", {}),
        }
        self.settings = Settings(Path("missing.pt"), Path("map.csv"), Path("catalog.json"), None)

    def test_unknown_metadata_is_not_excluded(self) -> None:
        remaining = filter_candidates(self.catalog, {0, 1, 2}, {"shape": "원형"})
        self.assertEqual(remaining, {0, 2})

    def test_unknown_answer_keeps_candidates(self) -> None:
        self.assertEqual(
            filter_candidates(self.catalog, {0, 1, 2}, {"shape": "모르겠어요"}), {0, 1, 2}
        )

    def test_question_requires_a_real_difference(self) -> None:
        self.assertIsNotNone(next_question(self.catalog, {0, 1, 2}, set()))
        self.assertIsNone(next_question(self.catalog, {0, 2}, set()))
        self.assertIsNone(next_question(self.catalog, {0, 1}, {"shape"}))

    def test_unsourced_metadata_is_not_used(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "map.csv").write_text(
                "yolo_class_id,product_code,pill_name\n0,TEST-A,가상A\n", encoding="utf-8"
            )
            (root / "meta.json").write_text(
                '{"products":{"TEST-A":{"shape":{"values":["원형"],"verified":true}}}}',
                encoding="utf-8",
            )
            self.assertEqual(load_catalog(root / "map.csv", root / "meta.json")[0].attributes, {})

    def test_real_catalog_has_only_reviewed_attributes(self) -> None:
        catalog = load_catalog(
            PROJECT_ROOT / "docs/results/yolo11n_20261105/class_map.csv",
            PROJECT_ROOT / "configs/tamgenie/verified_metadata.json",
        )
        self.assertEqual(len(catalog), 56)
        self.assertEqual(sum(bool(p.attributes) for p in catalog.values()), 2)
        self.assertTrue(all("imprint" not in p.attributes for p in catalog.values()))

    def test_invalid_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            read_image(b"not an image", self.settings)

    def test_image_is_rgb_without_persistence(self) -> None:
        data = io.BytesIO()
        Image.new("RGBA", (64, 64), (20, 40, 60, 255)).save(data, format="PNG")
        image = read_image(data.getvalue(), self.settings)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (64, 64))

    def test_secondary_is_not_fabricated(self) -> None:
        self.assertFalse(UnavailableClassifier.available)
        self.assertIsNone(UnavailableClassifier().predict(Image.new("RGB", (64, 64))))

    def test_model_disagreement_never_confirms_identity(self) -> None:
        class ContradictingClassifier:
            available = True

            def predict(self, image: Image.Image) -> Classification:
                return Classification(1, 0.99)

        image = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        )
        result = route(
            Detection(1, 0, 0.99, (0, 0, 100, 100)),
            image,
            0.6,
            ContradictingClassifier(),
            flags=["임시 품질 우려"],
        )
        self.assertTrue(result["needs_review"])
        self.assertFalse(result["identity_confirmed"])
        self.assertTrue(any("불일치" in reason for reason in result["reasons"]))

    def test_overlap_routes_both_classes_for_review(self) -> None:
        items = [Detection(1, 0, 0.99, (10, 10, 80, 80)), Detection(2, 1, 0.99, (10, 10, 80, 80))]
        flags = overlap_flags(items)
        self.assertTrue(flags[1] and flags[2])

    def test_annotation_does_not_modify_input(self) -> None:
        image = Image.new("RGB", (120, 120), "white")
        original = image.tobytes()
        item = Detection(1, 0, 0.8, (10, 10, 100, 100))
        annotate(image, [item], self.catalog, self.settings)
        self.assertEqual(original, image.tobytes())
        self.assertGreater(crop_image(image, item).width, 0)


if __name__ == "__main__":
    unittest.main()
