"""Mapping between `news_id` and discrete semantic code sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _code_key(code: tuple[int, ...] | list[int]) -> str:
    return "_".join(str(part) for part in code)


@dataclass(frozen=True)
class SemanticMappingSummary:
    """High-level statistics for an exported semantic ID mapping."""

    item_count: int
    unique_code_count: int
    collided_code_count: int
    max_collision_size: int


class SemanticIDMapper:
    """Bi-directional semantic code lookup."""

    def __init__(
        self,
        *,
        item_to_code: dict[str, tuple[int, ...]],
        code_to_items: dict[tuple[int, ...], list[str]],
    ) -> None:
        self._item_to_code = item_to_code
        self._code_to_items = code_to_items

    @classmethod
    def from_codes(cls, item_ids: list[str], codes: np.ndarray) -> "SemanticIDMapper":
        indices = np.asarray(codes, dtype=np.int32)
        if indices.ndim != 2:
            raise ValueError("codes must have shape [n_items, code_length]")
        if len(item_ids) != indices.shape[0]:
            raise ValueError("item_ids length must match codes row count")

        item_to_code: dict[str, tuple[int, ...]] = {}
        code_to_items: dict[tuple[int, ...], list[str]] = {}
        for item_id, code_row in zip(item_ids, indices.tolist(), strict=True):
            code_tuple = tuple(int(part) for part in code_row)
            item_to_code[item_id] = code_tuple
            code_to_items.setdefault(code_tuple, []).append(item_id)
        return cls(item_to_code=item_to_code, code_to_items=code_to_items)

    @property
    def item_to_code(self) -> dict[str, tuple[int, ...]]:
        return self._item_to_code

    @property
    def code_to_items(self) -> dict[tuple[int, ...], list[str]]:
        return self._code_to_items

    def code_for_item(self, item_id: str) -> tuple[int, ...] | None:
        return self._item_to_code.get(item_id)

    def items_for_code(self, code: tuple[int, ...] | list[int]) -> list[str]:
        return list(self._code_to_items.get(tuple(code), []))

    def nearest_codes(
        self,
        code: tuple[int, ...] | list[int],
        *,
        limit: int = 5,
    ) -> list[tuple[int, ...]]:
        """Return existing codes ordered by Hamming distance to the target code."""

        target = tuple(code)
        ranked = sorted(
            self._code_to_items.keys(),
            key=lambda candidate: (self._hamming_distance(target, candidate), candidate),
        )
        return ranked[:limit]

    def summary(self) -> SemanticMappingSummary:
        collision_sizes = [len(items) for items in self._code_to_items.values()]
        collided = [size for size in collision_sizes if size > 1]
        return SemanticMappingSummary(
            item_count=len(self._item_to_code),
            unique_code_count=len(self._code_to_items),
            collided_code_count=len(collided),
            max_collision_size=max(collision_sizes, default=0),
        )

    def save(self, output_dir: str | Path) -> None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        item_payload = {
            item_id: list(code) for item_id, code in sorted(self._item_to_code.items())
        }
        code_payload = {
            _code_key(code): items for code, items in sorted(self._code_to_items.items())
        }
        summary_payload = self.summary().__dict__

        (target_dir / "item_to_code.json").write_text(
            json.dumps(item_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (target_dir / "code_to_items.json").write_text(
            json.dumps(code_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (target_dir / "semantic_id_summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "SemanticIDMapper":
        source_dir = Path(directory)
        item_payload = json.loads((source_dir / "item_to_code.json").read_text(encoding="utf-8"))
        code_payload = json.loads((source_dir / "code_to_items.json").read_text(encoding="utf-8"))

        item_to_code = {
            item_id: tuple(int(part) for part in code)
            for item_id, code in item_payload.items()
        }
        code_to_items = {
            tuple(int(part) for part in key.split("_")): list(items)
            for key, items in code_payload.items()
        }
        return cls(item_to_code=item_to_code, code_to_items=code_to_items)

    @staticmethod
    def _hamming_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        if len(left) != len(right):
            raise ValueError("Codes must have the same length for Hamming distance")
        return sum(1 for l_token, r_token in zip(left, right, strict=True) if l_token != r_token)
