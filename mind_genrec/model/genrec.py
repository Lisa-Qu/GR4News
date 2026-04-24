"""Generative retrieval model interfaces and a neutral placeholder model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.beam_search import BeamSearchResult, SemanticCodeBeamSearch
from mind_genrec.model.code_trie import CodeTrie
from mind_genrec.model.lazy_ar_decoder import LazyARDecoderConfig, LazyAutoregressiveDecoder
from mind_genrec.model.semantic_id_mapper import SemanticIDMapper
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig


@dataclass(frozen=True)
class GeneratedCandidate:
    """One generated candidate returned by the model layer."""

    news_id: str
    score: float
    semantic_id: list[int] | None = None


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for the semantic-ID generator."""

    input_embedding_dim: int
    decoder_type: str = "ar"
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    code_length: int = 4
    codebook_size: int = 256
    max_history_length: int = 50
    lazy_parallel_layers: int | None = None


class GenRecModel(Protocol):
    """Serving-facing interface for the generator."""

    @property
    def model_name(self) -> str:
        """Human-readable active model name."""

    @property
    def is_placeholder(self) -> bool:
        """Whether the current implementation is only a placeholder."""

    def recommend(self, history: list[str], top_k: int) -> list[GeneratedCandidate]:
        """Generate top-k candidates from clicked history."""


def _default_generator_model_name(config: GeneratorConfig, *, beam: bool) -> str:
    prefix = "lazy-ar-semantic-generator" if config.decoder_type == "lazy_ar" else "ar-semantic-generator"
    return f"{prefix}-beam" if beam else prefix


class ARSemanticIdGenerator(nn.Module):
    """Predict semantic codes from clicked history with a configurable decoder."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.user_encoder = HistorySequenceEncoder(
            UserEncoderConfig(
                input_dim=config.input_embedding_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                max_history_length=config.max_history_length,
            )
        )
        if config.decoder_type == "ar":
            self.decoder = CodeAutoregressiveDecoder(
                ARDecoderConfig(
                    hidden_dim=config.hidden_dim,
                    codebook_size=config.codebook_size,
                    code_length=config.code_length,
                    num_heads=config.num_heads,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                )
            )
        elif config.decoder_type == "lazy_ar":
            self.decoder = LazyAutoregressiveDecoder(
                LazyARDecoderConfig(
                    hidden_dim=config.hidden_dim,
                    codebook_size=config.codebook_size,
                    code_length=config.code_length,
                    num_heads=config.num_heads,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    parallel_layers=config.lazy_parallel_layers,
                )
            )
        else:
            raise NotImplementedError(
                f"Unsupported decoder_type={config.decoder_type!r}. "
                "Supported values are 'ar' and 'lazy_ar'."
            )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return teacher-forced logits over semantic code tokens."""

        user_state, _ = self.user_encoder(history_embeddings, history_mask)
        return self.decoder(user_state, target_codes)

    @staticmethod
    def compute_loss(logits: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over all semantic code positions."""

        if logits.ndim != 3:
            raise ValueError("logits must have shape [batch, code_length, vocab_size]")
        if target_codes.ndim != 2:
            raise ValueError("target_codes must have shape [batch, code_length]")
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_codes.reshape(-1),
        )

    @torch.no_grad()
    def predict_codes(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy decode semantic codes."""

        user_state, _ = self.user_encoder(history_embeddings, history_mask)
        return self.decoder.greedy_decode(user_state)

    @torch.no_grad()
    def predict_codes_with_scores(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decode semantic codes and return average log-probability scores."""

        user_state, _ = self.user_encoder(history_embeddings, history_mask)
        return self.decoder.greedy_decode_with_scores(user_state)

    @torch.no_grad()
    def predict_topk_codes_with_scores(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        top_k: int,
        beam_width: int | None = None,
    ) -> list[list[BeamSearchResult]]:
        """Beam-search semantic codes from the user state."""

        user_state, _ = self.user_encoder(history_embeddings, history_mask)
        search = SemanticCodeBeamSearch(self.decoder)
        return search.search(user_state, top_k=top_k, beam_width=beam_width)



class SemanticIdGreedyRetriever:
    """Serving-time wrapper around a trained AR semantic-code generator."""

    def __init__(
        self,
        *,
        model: ARSemanticIdGenerator,
        mapper: SemanticIDMapper,
        item_embeddings: np.ndarray,
        item_ids: list[str],
        device: torch.device,
        model_name: str = "ar-semantic-generator",
        fallback_code_limit: int = 5,
    ) -> None:
        self._model = model.eval()
        self._mapper = mapper
        self._device = device
        self._model_name = model_name
        self._fallback_code_limit = fallback_code_limit
        self._item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32, device=device)
        self._item_ids = item_ids
        self._item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        semantic_artifact_dir: str | Path,
        mapper: SemanticIDMapper,
        device: str | torch.device = "cpu",
        model_name: str | None = None,
    ) -> "SemanticIdGreedyRetriever":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = GeneratorConfig(**checkpoint["model_config"])
        model = ARSemanticIdGenerator(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        artifact_dir = Path(semantic_artifact_dir)
        item_embeddings = np.load(artifact_dir / "item_embeddings.npy")
        item_ids_path = artifact_dir / "item_ids.json"
        item_ids = json.loads(item_ids_path.read_text(encoding="utf-8"))

        return cls(
            model=model,
            mapper=mapper,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            device=torch.device(device),
            model_name=model_name or _default_generator_model_name(model_config, beam=False),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_placeholder(self) -> bool:
        return False

    def recommend(self, history: list[str], top_k: int) -> list[GeneratedCandidate]:
        history_indices = [
            self._item_to_index[item_id]
            for item_id in history
            if item_id in self._item_to_index
        ]
        history_indices = history_indices[-self._model.config.max_history_length :]
        if not history_indices:
            return []

        history_embeddings = self._item_embeddings[history_indices].unsqueeze(0)
        history_mask = torch.ones(
            1,
            len(history_indices),
            dtype=torch.bool,
            device=self._device,
        )
        codes, log_scores = self._model.predict_codes_with_scores(history_embeddings, history_mask)
        predicted_code = tuple(int(value) for value in codes[0].tolist())
        predicted_score = float(log_scores[0].item())

        candidates: list[GeneratedCandidate] = []
        exact_items = self._mapper.items_for_code(predicted_code)
        if exact_items:
            for item_id in exact_items[:top_k]:
                candidates.append(
                    GeneratedCandidate(
                        news_id=item_id,
                        score=predicted_score,
                        semantic_id=list(predicted_code),
                    )
                )
            return candidates

        for rank, code in enumerate(
            self._mapper.nearest_codes(predicted_code, limit=self._fallback_code_limit)
        ):
            distance_penalty = float(
                SemanticIDMapper._hamming_distance(predicted_code, code)  # type: ignore[attr-defined]
            )
            for item_id in self._mapper.items_for_code(code):
                candidates.append(
                    GeneratedCandidate(
                        news_id=item_id,
                        score=predicted_score - distance_penalty - 0.01 * rank,
                        semantic_id=list(code),
                    )
                )
                if len(candidates) >= top_k:
                    return candidates
        return candidates


class SemanticIdBeamSearchRetriever:
    """Serving-time wrapper that uses beam search over semantic codes."""

    def __init__(
        self,
        *,
        model: ARSemanticIdGenerator,
        mapper: SemanticIDMapper,
        item_embeddings: np.ndarray,
        item_ids: list[str],
        device: torch.device,
        model_name: str = "ar-semantic-generator-beam",
        beam_width: int = 8,
        fallback_code_limit: int = 5,
        trie: CodeTrie | None = None,
    ) -> None:
        self._model = model.eval()
        self._mapper = mapper
        self._device = device
        self._model_name = model_name
        self._beam_width = beam_width
        self._fallback_code_limit = fallback_code_limit
        self._trie = trie or CodeTrie.from_code_to_items(mapper.code_to_items)
        self._item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32, device=device)
        self._item_ids = item_ids
        self._item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        semantic_artifact_dir: str | Path,
        mapper: SemanticIDMapper,
        device: str | torch.device = "cpu",
        model_name: str | None = None,
        beam_width: int = 8,
    ) -> "SemanticIdBeamSearchRetriever":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = GeneratorConfig(**checkpoint["model_config"])
        model = ARSemanticIdGenerator(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        artifact_dir = Path(semantic_artifact_dir)
        item_embeddings = np.load(artifact_dir / "item_embeddings.npy")
        item_ids = json.loads((artifact_dir / "item_ids.json").read_text(encoding="utf-8"))

        return cls(
            model=model,
            mapper=mapper,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            device=torch.device(device),
            model_name=model_name or _default_generator_model_name(model_config, beam=True),
            beam_width=beam_width,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_placeholder(self) -> bool:
        return False

    def recommend(self, history: list[str], top_k: int) -> list[GeneratedCandidate]:
        history_indices = [
            self._item_to_index[item_id]
            for item_id in history
            if item_id in self._item_to_index
        ]
        history_indices = history_indices[-self._model.config.max_history_length :]
        if not history_indices:
            return []

        history_embeddings = self._item_embeddings[history_indices].unsqueeze(0)
        history_mask = torch.ones(
            1,
            len(history_indices),
            dtype=torch.bool,
            device=self._device,
        )
        search = SemanticCodeBeamSearch(self._model.decoder, trie=self._trie)
        user_state, _ = self._model.user_encoder(history_embeddings, history_mask)
        beam_results = search.search(
            user_state,
            top_k=top_k,
            beam_width=max(self._beam_width, top_k),
        )[0]

        merged: dict[str, GeneratedCandidate] = {}
        for beam_rank, beam in enumerate(beam_results):
            # Trie guarantees every code maps to at least one real item.
            # Keep fallback for the no-trie code path (backward compat).
            exact_items = self._mapper.items_for_code(beam.code)
            if exact_items:
                self._merge_candidates(
                    merged,
                    item_ids=exact_items,
                    score=beam.score - 0.01 * beam_rank,
                    semantic_code=beam.code,
                    limit=top_k,
                )
                continue

            for code_rank, fallback_code in enumerate(
                self._mapper.nearest_codes(beam.code, limit=self._fallback_code_limit)
            ):
                distance_penalty = float(self._mapper._hamming_distance(beam.code, fallback_code))
                self._merge_candidates(
                    merged,
                    item_ids=self._mapper.items_for_code(fallback_code),
                    score=beam.score - distance_penalty - 0.01 * beam_rank - 0.005 * code_rank,
                    semantic_code=fallback_code,
                    limit=top_k,
                )
                if len(merged) >= top_k:
                    break

        ranked = sorted(merged.values(), key=lambda item: (-item.score, item.news_id))
        return ranked[:top_k]

    @staticmethod
    def _merge_candidates(
        merged: dict[str, GeneratedCandidate],
        *,
        item_ids: list[str],
        score: float,
        semantic_code: tuple[int, ...],
        limit: int,
    ) -> None:
        for item_id in item_ids:
            current = merged.get(item_id)
            candidate = GeneratedCandidate(
                news_id=item_id,
                score=score,
                semantic_id=list(semantic_code),
            )
            if current is None or candidate.score > current.score:
                merged[item_id] = candidate
            if len(merged) >= limit:
                return


class StubGenerativeRetriever:
    """Neutral placeholder used before training.

    This keeps the service path and API stable while the actual MIND training
    pipeline is still under construction.
    """

    def __init__(self, model_name: str = "stub-genrec") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_placeholder(self) -> bool:
        return True

    def recommend(self, history: list[str], top_k: int) -> list[GeneratedCandidate]:
        # Returning no candidates is more honest than echoing history items or
        # fabricating IDs that look like a trained next-click model output.
        _ = history
        _ = top_k
        return []
