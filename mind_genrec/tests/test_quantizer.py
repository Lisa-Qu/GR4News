"""Unit tests for ResidualQuantizer and SemanticIDMapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mind_genrec.model.residual_quantizer import ResidualQuantizer, ResidualQuantizerConfig
from mind_genrec.model.semantic_id_mapper import SemanticIDMapper


def _make_vectors(n: int = 20, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestResidualQuantizer(unittest.TestCase):
    def _make_quantizer(self, num_codebooks: int = 4, codebook_size: int = 8) -> ResidualQuantizer:
        cfg = ResidualQuantizerConfig(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_iterations=5,
            sample_size=20,
            batch_size=32,
            seed=7,
        )
        return ResidualQuantizer(cfg)

    def test_fit_output_shape(self) -> None:
        q = self._make_quantizer(num_codebooks=4)
        vectors = _make_vectors(n=20, dim=16)
        codes = q.fit(vectors)
        self.assertEqual(codes.shape, (20, 4))

    def test_fit_codes_within_codebook_range(self) -> None:
        codebook_size = 8
        q = self._make_quantizer(codebook_size=codebook_size)
        vectors = _make_vectors(n=20, dim=16)
        codes = q.fit(vectors)
        self.assertTrue(np.all(codes >= 0))
        self.assertTrue(np.all(codes < codebook_size))

    def test_is_fitted_after_fit(self) -> None:
        q = self._make_quantizer()
        self.assertFalse(q.is_fitted)
        q.fit(_make_vectors(n=20))
        self.assertTrue(q.is_fitted)

    def test_encode_shape_matches_fit(self) -> None:
        q = self._make_quantizer(num_codebooks=4)
        train_vectors = _make_vectors(n=20)
        q.fit(train_vectors)
        new_vectors = _make_vectors(n=5, seed=99)
        codes = q.encode(new_vectors)
        self.assertEqual(codes.shape, (5, 4))

    def test_encode_before_fit_raises(self) -> None:
        q = self._make_quantizer()
        with self.assertRaises(RuntimeError):
            q.encode(_make_vectors(n=5))

    def test_decode_shape(self) -> None:
        q = self._make_quantizer(num_codebooks=4)
        vectors = _make_vectors(n=20, dim=16)
        codes = q.fit(vectors)
        reconstructed = q.decode(codes)
        self.assertEqual(reconstructed.shape, (20, 16))

    def test_decode_before_fit_raises(self) -> None:
        q = self._make_quantizer()
        with self.assertRaises(RuntimeError):
            q.decode(np.zeros((5, 4), dtype=np.int32))

    def test_fit_single_vector(self) -> None:
        q = self._make_quantizer()
        codes = q.fit(_make_vectors(n=1))
        self.assertEqual(codes.shape, (1, 4))

    def test_codebooks_count_after_fit(self) -> None:
        num_codebooks = 3
        q = self._make_quantizer(num_codebooks=num_codebooks)
        q.fit(_make_vectors(n=20))
        self.assertEqual(len(q.codebooks), num_codebooks)


class TestSemanticIDMapper(unittest.TestCase):
    def _make_mapper(self) -> tuple[SemanticIDMapper, list[str], np.ndarray]:
        item_ids = ["N1", "N2", "N3", "N4", "N5"]
        codes = np.array([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [0, 1, 2, 3],  # intentional collision with N1
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ], dtype=np.int32)
        mapper = SemanticIDMapper.from_codes(item_ids, codes)
        return mapper, item_ids, codes

    def test_code_for_item_returns_tuple(self) -> None:
        mapper, _, _ = self._make_mapper()
        code = mapper.code_for_item("N1")
        self.assertIsInstance(code, tuple)
        self.assertEqual(len(code), 4)

    def test_code_for_item_correct_values(self) -> None:
        mapper, _, _ = self._make_mapper()
        self.assertEqual(mapper.code_for_item("N1"), (0, 1, 2, 3))
        self.assertEqual(mapper.code_for_item("N2"), (1, 0, 3, 2))

    def test_items_for_code_returns_collisions(self) -> None:
        mapper, _, _ = self._make_mapper()
        # N1 and N3 share the same code
        items = mapper.items_for_code((0, 1, 2, 3))
        self.assertIn("N1", items)
        self.assertIn("N3", items)

    def test_items_for_code_unknown_returns_empty(self) -> None:
        mapper, _, _ = self._make_mapper()
        self.assertEqual(mapper.items_for_code((9, 9, 9, 9)), [])

    def test_missing_item_returns_none(self) -> None:
        mapper, _, _ = self._make_mapper()
        self.assertIsNone(mapper.code_for_item("N999"))

    def test_summary_item_count(self) -> None:
        mapper, item_ids, _ = self._make_mapper()
        summary = mapper.summary()
        self.assertEqual(summary.item_count, len(item_ids))

    def test_summary_collision_count(self) -> None:
        mapper, _, _ = self._make_mapper()
        summary = mapper.summary()
        # N1 and N3 collide → 1 collided code
        self.assertEqual(summary.collided_code_count, 1)

    def test_summary_unique_code_count(self) -> None:
        mapper, _, _ = self._make_mapper()
        summary = mapper.summary()
        # 5 items, 1 collision → 4 unique codes
        self.assertEqual(summary.unique_code_count, 4)

    def test_save_and_load_roundtrip(self) -> None:
        mapper, _, _ = self._make_mapper()
        with tempfile.TemporaryDirectory() as tmp:
            mapper.save(tmp)
            restored = SemanticIDMapper.load(tmp)
            # Files must exist
            self.assertTrue((Path(tmp) / "item_to_code.json").exists())
            self.assertTrue((Path(tmp) / "code_to_items.json").exists())

        self.assertEqual(restored.code_for_item("N1"), (0, 1, 2, 3))
        self.assertIn("N1", restored.items_for_code((0, 1, 2, 3)))
        self.assertEqual(restored.summary().item_count, 5)

    def test_nearest_codes_returns_list(self) -> None:
        mapper, _, _ = self._make_mapper()
        nearest = mapper.nearest_codes((0, 1, 2, 3), limit=3)
        self.assertIsInstance(nearest, list)
        self.assertLessEqual(len(nearest), 3)
        # Exact match should be first
        self.assertEqual(nearest[0], (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
