"""Core data types and parsing helpers for MIND preprocessing and serving."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Protocol


@dataclass(frozen=True)
class NewsItem:
    """Canonical news record used by the serving layer."""

    news_id: str
    category: str = ""
    subcategory: str = ""
    title: str = ""
    abstract: str = ""
    url: str = ""
    title_entities: str = ""
    abstract_entities: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class BehaviorImpression:
    """One candidate item inside a MIND impression list."""

    news_id: str
    clicked: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class BehaviorRecord:
    """Normalized behavior row from MIND."""

    impression_id: str
    user_id: str
    timestamp: str
    history: list[str]
    impressions: list[BehaviorImpression]
    split: str = ""

    def clicked_news_ids(self) -> list[str]:
        """Return clicked items when labels are present."""
        return [item.news_id for item in self.impressions if item.clicked is True]

    def candidate_news_ids(self) -> list[str]:
        """Return all impression candidate ids."""
        return [item.news_id for item in self.impressions]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "impression_id": self.impression_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "history": self.history,
            "impressions": [item.to_dict() for item in self.impressions],
            "split": self.split,
        }


@dataclass(frozen=True)
class TrainingSample:
    """One normalized training example for next-click generation."""

    sample_id: str
    impression_id: str
    user_id: str
    split: str
    history: list[str]
    candidate_news_ids: list[str]
    target_news_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return asdict(self)


class MindCatalog(Protocol):
    """Read-only item catalog interface used by the service layer."""

    def get_item(self, news_id: str) -> NewsItem | None:
        """Return metadata for one item."""

    def list_item_ids(self) -> list[str]:
        """Return all known item ids."""


class InMemoryMindCatalog:
    """Simple in-memory item catalog.

    This is the default serving-time catalog placeholder until the real MIND
    preprocessing pipeline is added.
    """

    def __init__(self, items: Dict[str, NewsItem] | None = None) -> None:
        self._items = items or {}

    @classmethod
    def from_records(cls, records: Iterable[NewsItem]) -> "InMemoryMindCatalog":
        return cls({record.news_id: record for record in records})

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "InMemoryMindCatalog":
        """Load a preprocessed news corpus from JSONL.

        Expected JSONL keys:
        - `news_id`
        - `category`
        - `subcategory`
        - `title`
        - `abstract`
        """

        items: Dict[str, NewsItem] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                news_id = payload["news_id"]
                items[news_id] = NewsItem(
                    news_id=news_id,
                    category=payload.get("category", ""),
                    subcategory=payload.get("subcategory", ""),
                    title=payload.get("title", ""),
                    abstract=payload.get("abstract", ""),
                )
        return cls(items)

    def get_item(self, news_id: str) -> NewsItem | None:
        return self._items.get(news_id)

    def list_item_ids(self) -> list[str]:
        return list(self._items.keys())


def iter_news_tsv(path: str | Path) -> Iterator[NewsItem]:
    """Yield normalized `NewsItem` records from raw MIND `news.tsv`.

    Expected column order in MIND:
    `news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities`
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw:
                continue
            fields = raw.split("\t")
            if len(fields) < 8:
                fields.extend([""] * (8 - len(fields)))
            yield NewsItem(
                news_id=fields[0],
                category=fields[1],
                subcategory=fields[2],
                title=fields[3],
                abstract=fields[4],
                url=fields[5],
                title_entities=fields[6],
                abstract_entities=fields[7],
            )


def parse_impression_token(token: str) -> BehaviorImpression:
    """Parse one MIND impression token such as `N123-1` or `N123`."""

    token = token.strip()
    if not token:
        raise ValueError("Empty impression token")
    if "-" not in token:
        return BehaviorImpression(news_id=token, clicked=None)
    news_id, label = token.rsplit("-", 1)
    if label == "1":
        clicked = True
    elif label == "0":
        clicked = False
    else:
        clicked = None
    return BehaviorImpression(news_id=news_id, clicked=clicked)


def iter_behavior_tsv(path: str | Path, split: str = "") -> Iterator[BehaviorRecord]:
    """Yield normalized `BehaviorRecord` rows from raw MIND `behaviors.tsv`."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw:
                continue
            fields = raw.split("\t")
            if len(fields) < 5:
                fields.extend([""] * (5 - len(fields)))
            impression_id, user_id, timestamp, history_raw, impressions_raw = fields[:5]
            history = history_raw.split() if history_raw.strip() else []
            impressions = [
                parse_impression_token(token)
                for token in impressions_raw.split()
                if token.strip()
            ]
            yield BehaviorRecord(
                impression_id=impression_id,
                user_id=user_id,
                timestamp=timestamp,
                history=history,
                impressions=impressions,
                split=split,
            )


def build_training_samples(
    behaviors: Iterable[BehaviorRecord],
    *,
    split: str,
    max_history_length: int = 50,
    skip_empty_history: bool = True,
    skip_unlabeled: bool = True,
) -> Iterator[TrainingSample]:
    """Create one `TrainingSample` per clicked target in the impression list."""

    for record in behaviors:
        history = record.history[-max_history_length:] if max_history_length > 0 else list(record.history)
        if skip_empty_history and not history:
            continue
        clicked_news_ids = record.clicked_news_ids()
        if skip_unlabeled and not clicked_news_ids:
            continue
        candidate_news_ids = record.candidate_news_ids()
        for target_news_id in clicked_news_ids:
            yield TrainingSample(
                sample_id=f"{record.impression_id}:{target_news_id}",
                impression_id=record.impression_id,
                user_id=record.user_id,
                split=split or record.split,
                history=history,
                candidate_news_ids=candidate_news_ids,
                target_news_id=target_news_id,
            )


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any] | NewsItem | BehaviorRecord | TrainingSample]) -> None:
    """Write dictionaries or dataclass-backed records to JSONL."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "to_dict"):
                payload = record.to_dict()  # type: ignore[assignment]
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records as dictionaries."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                yield json.loads(raw)
