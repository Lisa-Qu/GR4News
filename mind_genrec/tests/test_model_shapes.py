"""Unit tests for model tensor shapes."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.genrec import ARSemanticIdGenerator, GeneratorConfig
from mind_genrec.model.lazy_ar_decoder import LazyARDecoderConfig, LazyAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig


class TestHistorySequenceEncoder(unittest.TestCase):
    def _make_encoder(self, input_dim: int = 32, hidden_dim: int = 32) -> HistorySequenceEncoder:
        cfg = UserEncoderConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            max_history_length=20,
        )
        return HistorySequenceEncoder(cfg)

    def test_user_state_shape(self) -> None:
        encoder = self._make_encoder()
        batch, seq_len, dim = 3, 5, 32
        embeddings = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        user_state, _ = encoder(embeddings, mask)
        self.assertEqual(user_state.shape, (batch, 32))

    def test_encoded_history_shape(self) -> None:
        encoder = self._make_encoder()
        batch, seq_len, dim = 2, 7, 32
        embeddings = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        _, encoded = encoder(embeddings, mask)
        self.assertEqual(encoded.shape, (batch, seq_len, 32))

    def test_partial_mask_does_not_crash(self) -> None:
        encoder = self._make_encoder()
        batch, seq_len, dim = 2, 6, 32
        embeddings = torch.randn(batch, seq_len, dim)
        mask = torch.tensor([[True, True, True, False, False, False],
                              [True, True, True, True, True, False]], dtype=torch.bool)
        user_state, _ = encoder(embeddings, mask)
        self.assertEqual(user_state.shape, (batch, 32))

    def test_wrong_ndim_raises(self) -> None:
        encoder = self._make_encoder()
        with self.assertRaises(ValueError):
            encoder(torch.randn(3, 32), torch.ones(3, dtype=torch.bool))

    def test_truncates_long_history(self) -> None:
        encoder = self._make_encoder()
        # seq_len=25 > max_history_length=20, should not crash
        embeddings = torch.randn(1, 25, 32)
        mask = torch.ones(1, 25, dtype=torch.bool)
        user_state, _ = encoder(embeddings, mask)
        self.assertEqual(user_state.shape, (1, 32))


class TestARDecoder(unittest.TestCase):
    def _make_decoder(self, hidden_dim: int = 32) -> CodeAutoregressiveDecoder:
        cfg = ARDecoderConfig(
            hidden_dim=hidden_dim,
            codebook_size=16,
            code_length=4,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
        return CodeAutoregressiveDecoder(cfg)

    def test_forward_logits_shape(self) -> None:
        decoder = self._make_decoder()
        batch = 3
        user_state = torch.randn(batch, 32)
        target_codes = torch.randint(0, 16, (batch, 4))
        logits = decoder(user_state, target_codes)
        # [batch, code_length, codebook_size]
        self.assertEqual(logits.shape, (batch, 4, 16))

    def test_greedy_decode_shape(self) -> None:
        decoder = self._make_decoder()
        batch = 2
        user_state = torch.randn(batch, 32)
        codes = decoder.greedy_decode(user_state)
        self.assertEqual(codes.shape, (batch, 4))

    def test_greedy_decode_values_in_range(self) -> None:
        decoder = self._make_decoder()
        user_state = torch.randn(4, 32)
        codes = decoder.greedy_decode(user_state)
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < 16).all())


class TestLazyARDecoder(unittest.TestCase):
    def _make_decoder(self, hidden_dim: int = 32) -> LazyAutoregressiveDecoder:
        cfg = LazyARDecoderConfig(
            hidden_dim=hidden_dim,
            codebook_size=16,
            code_length=4,
            num_heads=4,
            num_layers=3,
            dropout=0.0,
            parallel_layers=1,
        )
        return LazyAutoregressiveDecoder(cfg)

    def test_forward_logits_shape(self) -> None:
        decoder = self._make_decoder()
        batch = 2
        user_state = torch.randn(batch, 32)
        target_codes = torch.randint(0, 16, (batch, 4))
        logits = decoder(user_state, target_codes)
        self.assertEqual(logits.shape, (batch, 4, 16))

    def test_greedy_decode_shape(self) -> None:
        decoder = self._make_decoder()
        user_state = torch.randn(3, 32)
        codes = decoder.greedy_decode(user_state)
        self.assertEqual(codes.shape, (3, 4))


class TestARSemanticIdGenerator(unittest.TestCase):
    def _make_generator(self, decoder_type: str = "ar") -> ARSemanticIdGenerator:
        cfg = GeneratorConfig(
            input_embedding_dim=32,
            decoder_type=decoder_type,
            hidden_dim=32,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            code_length=4,
            codebook_size=16,
            max_history_length=20,
            lazy_parallel_layers=1,
        )
        return ARSemanticIdGenerator(cfg)

    def test_forward_logits_shape_ar(self) -> None:
        model = self._make_generator("ar")
        batch, seq_len, dim = 2, 5, 32
        embeddings = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        target_codes = torch.randint(0, 16, (batch, 4))
        logits = model(embeddings, mask, target_codes)
        self.assertEqual(logits.shape, (batch, 4, 16))

    def test_forward_logits_shape_lazy_ar(self) -> None:
        model = self._make_generator("lazy_ar")
        batch, seq_len, dim = 2, 5, 32
        embeddings = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        target_codes = torch.randint(0, 16, (batch, 4))
        logits = model(embeddings, mask, target_codes)
        self.assertEqual(logits.shape, (batch, 4, 16))

    def test_compute_loss_scalar(self) -> None:
        model = self._make_generator()
        batch = 3
        logits = torch.randn(batch, 4, 16)
        target_codes = torch.randint(0, 16, (batch, 4))
        loss = ARSemanticIdGenerator.compute_loss(logits, target_codes)
        self.assertEqual(loss.shape, ())  # scalar
        self.assertFalse(torch.isnan(loss))

    def test_predict_codes_shape(self) -> None:
        model = self._make_generator()
        batch, seq_len, dim = 2, 4, 32
        embeddings = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        codes = model.predict_codes(embeddings, mask)
        self.assertEqual(codes.shape, (batch, 4))

    def test_unsupported_decoder_type_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            cfg = GeneratorConfig(input_embedding_dim=32, decoder_type="transformer_xl")
            ARSemanticIdGenerator(cfg)


if __name__ == "__main__":
    unittest.main()
