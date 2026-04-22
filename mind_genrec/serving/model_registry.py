"""Model and catalog registry for the backend-first service path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import os
import torch

from mind_genrec.baseline import (
    CheckpointedTwoTowerRetriever,
    StubTwoTowerRetriever,
    TwoTowerRetriever,
)
from mind_genrec.data import InMemoryMindCatalog, MindCatalog
from mind_genrec.model import (
    GenRecModel,
    SemanticIDMapper,
    SemanticIdBeamSearchRetriever,
    SemanticIdGreedyRetriever,
    StubGenerativeRetriever,
)


@dataclass
class ModelBundle:
    """All runtime dependencies needed by the retrieval service."""

    generator: GenRecModel
    baseline: TwoTowerRetriever | None
    catalog: MindCatalog
    semantic_mapper: SemanticIDMapper | None = None
    semantic_mapping_loaded: bool = False
    runtime_signature: str = "default"

    @property
    def catalog_size(self) -> int:
        """Return the number of catalog entries currently loaded."""
        return len(self.catalog.list_item_ids())

    @property
    def uses_placeholder_components(self) -> bool:
        """Return whether any major runtime component is still a placeholder."""
        baseline_placeholder = bool(
            self.baseline is not None and getattr(self.baseline, "is_placeholder", False)
        )
        return bool(getattr(self.generator, "is_placeholder", False)) or baseline_placeholder

    @property
    def ready(self) -> bool:
        """Backward-compatible alias for service readiness."""
        return self.service_ready

    @property
    def generator_ready(self) -> bool:
        """Return whether the generator path can serve real results."""
        return (
            not bool(getattr(self.generator, "is_placeholder", False))
            and self.catalog_size > 0
            and self.semantic_mapping_loaded
        )

    @property
    def baseline_ready(self) -> bool:
        """Return whether the baseline path can serve real results."""
        return (
            self.baseline is not None
            and not bool(getattr(self.baseline, "is_placeholder", False))
            and self.catalog_size > 0
        )

    @property
    def service_ready(self) -> bool:
        """Return whether at least one real retrieval path is available."""
        return self.generator_ready or self.baseline_ready

    @property
    def semantic_unique_code_count(self) -> int:
        if self.semantic_mapper is None:
            return 0
        return self.semantic_mapper.summary().unique_code_count

    @property
    def semantic_collided_code_count(self) -> int:
        if self.semantic_mapper is None:
            return 0
        return self.semantic_mapper.summary().collided_code_count

    def missing_components(self) -> list[str]:
        """Return a human-readable list of missing runtime artifacts."""
        missing: list[str] = []
        if getattr(self.generator, "is_placeholder", False):
            missing.append("generator checkpoint")
        if self.catalog_size == 0:
            missing.append("item catalog")
        if not self.semantic_mapping_loaded:
            missing.append("semantic ID mapping")
        if self.baseline is None:
            missing.append("baseline retriever")
        elif getattr(self.baseline, "is_placeholder", False):
            missing.append("baseline checkpoint")
        return missing


class ModelRegistry:
    """Keeps track of the active model bundle.

    The first implementation intentionally loads deterministic placeholders so
    the API contract can be reviewed before training is introduced.
    """

    def __init__(
        self,
        *,
        news_jsonl_path: str | None = None,
        semantic_artifact_dir: str | None = None,
        generator_checkpoint_path: str | None = None,
        baseline_checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._lock = Lock()
        self._active_bundle = self._build_bundle(
            news_jsonl_path=news_jsonl_path or os.getenv("MIND_GENREC_NEWS_JSONL"),
            semantic_artifact_dir=semantic_artifact_dir or os.getenv("MIND_GENREC_SEMANTIC_DIR"),
            generator_checkpoint_path=generator_checkpoint_path or os.getenv("MIND_GENREC_GENERATOR_CKPT"),
            baseline_checkpoint_path=baseline_checkpoint_path or os.getenv("MIND_GENREC_BASELINE_CKPT"),
            device=device or os.getenv("MIND_GENREC_DEVICE"),
        )

    def _build_default_bundle(self) -> ModelBundle:
        return ModelBundle(
            generator=StubGenerativeRetriever(),
            baseline=StubTwoTowerRetriever(),
            catalog=InMemoryMindCatalog(),
            semantic_mapper=None,
            semantic_mapping_loaded=False,
            runtime_signature="default",
        )

    def _build_bundle(
        self,
        *,
        news_jsonl_path: str | None,
        semantic_artifact_dir: str | None,
        generator_checkpoint_path: str | None,
        baseline_checkpoint_path: str | None,
        device: str | None,
    ) -> ModelBundle:
        bundle = self._build_default_bundle()
        signature_parts = [
            f"news={news_jsonl_path or ''}",
            f"semantic={semantic_artifact_dir or ''}",
            f"generator={generator_checkpoint_path or ''}",
            f"baseline={baseline_checkpoint_path or ''}",
            f"device={device or ''}",
        ]

        if news_jsonl_path:
            news_path = Path(news_jsonl_path)
            if news_path.exists():
                bundle.catalog = InMemoryMindCatalog.from_jsonl(news_path)

        if semantic_artifact_dir:
            artifact_dir = Path(semantic_artifact_dir)
            item_path = artifact_dir / "item_to_code.json"
            code_path = artifact_dir / "code_to_items.json"
            if item_path.exists() and code_path.exists():
                bundle.semantic_mapper = SemanticIDMapper.load(artifact_dir)
                bundle.semantic_mapping_loaded = True
                runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

                if generator_checkpoint_path:
                    checkpoint_path = Path(generator_checkpoint_path)
                    if checkpoint_path.exists():
                        bundle.generator = SemanticIdBeamSearchRetriever.from_checkpoint(
                            checkpoint_path=checkpoint_path,
                            semantic_artifact_dir=artifact_dir,
                            mapper=bundle.semantic_mapper,
                            device=runtime_device,
                        )
                if baseline_checkpoint_path:
                    checkpoint_path = Path(baseline_checkpoint_path)
                    if checkpoint_path.exists():
                        bundle.baseline = CheckpointedTwoTowerRetriever.from_checkpoint(
                            checkpoint_path=checkpoint_path,
                            semantic_artifact_dir=artifact_dir,
                            device=runtime_device,
                        )

        bundle.runtime_signature = "|".join(signature_parts)
        return bundle

    def get_active_bundle(self) -> ModelBundle:
        with self._lock:
            return self._active_bundle

    def set_active_bundle(self, bundle: ModelBundle) -> None:
        with self._lock:
            self._active_bundle = bundle
