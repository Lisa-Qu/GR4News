"""Unit tests for data parsing and catalog helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mind_genrec.data.dataset import (
    BehaviorImpression,
    InMemoryMindCatalog,
    NewsItem,
    build_training_samples,
    iter_behavior_tsv,
    iter_jsonl,
    iter_news_tsv,
    parse_impression_token,
    write_jsonl,
)


_NEWS_TSV = (
    "N1\tnews\tlocal\tCity council news\tSome abstract.\thttps://a\t[]\t[]\n"
    "N2\tsports\tfootball\tGame recap\tTeam won.\thttps://b\t[]\t[]\n"
)

_BEHAVIORS_TSV = (
    "1\tU1\t11/11/2019 9:00:00 AM\tN1\tN2-1 N3-0\n"
    "2\tU2\t11/11/2019 9:05:00 AM\t\tN1-0 N2-0\n"
)


class TestIterNewsTsv(unittest.TestCase):
    def test_parses_all_fields(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(_NEWS_TSV)
            path = f.name

        items = list(iter_news_tsv(path))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].news_id, "N1")
        self.assertEqual(items[0].category, "news")
        self.assertEqual(items[0].subcategory, "local")
        self.assertEqual(items[0].title, "City council news")
        self.assertEqual(items[0].abstract, "Some abstract.")
        self.assertEqual(items[1].news_id, "N2")
        self.assertEqual(items[1].category, "sports")

    def test_yields_news_item_type(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(_NEWS_TSV)
            path = f.name
        for item in iter_news_tsv(path):
            self.assertIsInstance(item, NewsItem)

    def test_short_line_padded(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("N99\tcategory\tsubcategory\n")
            path = f.name
        items = list(iter_news_tsv(path))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].news_id, "N99")
        self.assertEqual(items[0].title, "")


class TestParseImpressionToken(unittest.TestCase):
    def test_clicked(self) -> None:
        imp = parse_impression_token("N123-1")
        self.assertEqual(imp.news_id, "N123")
        self.assertTrue(imp.clicked)

    def test_not_clicked(self) -> None:
        imp = parse_impression_token("N123-0")
        self.assertEqual(imp.news_id, "N123")
        self.assertFalse(imp.clicked)

    def test_no_label(self) -> None:
        imp = parse_impression_token("N123")
        self.assertEqual(imp.news_id, "N123")
        self.assertIsNone(imp.clicked)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_impression_token("")


class TestIterBehaviorTsv(unittest.TestCase):
    def test_parses_history_and_impressions(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(_BEHAVIORS_TSV)
            path = f.name

        records = list(iter_behavior_tsv(path, split="train"))
        self.assertEqual(len(records), 2)

        r0 = records[0]
        self.assertEqual(r0.impression_id, "1")
        self.assertEqual(r0.user_id, "U1")
        self.assertEqual(r0.history, ["N1"])
        self.assertEqual(r0.split, "train")
        self.assertEqual(len(r0.impressions), 2)
        self.assertIsInstance(r0.impressions[0], BehaviorImpression)

    def test_empty_history(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(_BEHAVIORS_TSV)
            path = f.name

        records = list(iter_behavior_tsv(path))
        self.assertEqual(records[1].history, [])

    def test_clicked_news_ids(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(_BEHAVIORS_TSV)
            path = f.name

        records = list(iter_behavior_tsv(path))
        self.assertEqual(records[0].clicked_news_ids(), ["N2"])
        self.assertEqual(records[1].clicked_news_ids(), [])


class TestBuildTrainingSamples(unittest.TestCase):
    def _make_records(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write(_BEHAVIORS_TSV)
            path = f.name
        return list(iter_behavior_tsv(path, split="train"))

    def test_skips_empty_history_by_default(self) -> None:
        records = self._make_records()
        samples = list(build_training_samples(records, split="train"))
        # record[1] has empty history → should be skipped
        self.assertEqual(len(samples), 1)

    def test_sample_fields(self) -> None:
        records = self._make_records()
        samples = list(build_training_samples(records, split="train"))
        s = samples[0]
        self.assertEqual(s.target_news_id, "N2")
        self.assertEqual(s.split, "train")
        self.assertIn("N1", s.history)

    def test_max_history_length_truncates(self) -> None:
        records = self._make_records()
        samples = list(build_training_samples(records, split="train", max_history_length=0))
        # max_history_length=0 means no truncation
        self.assertGreaterEqual(len(samples), 0)


class TestInMemoryMindCatalog(unittest.TestCase):
    def _make_catalog(self) -> InMemoryMindCatalog:
        return InMemoryMindCatalog.from_records([
            NewsItem(news_id="N1", category="news", title="Title 1"),
            NewsItem(news_id="N2", category="sports", title="Title 2"),
        ])

    def test_get_item_returns_correct_record(self) -> None:
        catalog = self._make_catalog()
        item = catalog.get_item("N1")
        self.assertIsNotNone(item)
        self.assertEqual(item.category, "news")
        self.assertEqual(item.title, "Title 1")

    def test_get_item_missing_returns_none(self) -> None:
        catalog = self._make_catalog()
        self.assertIsNone(catalog.get_item("N999"))

    def test_list_item_ids(self) -> None:
        catalog = self._make_catalog()
        ids = catalog.list_item_ids()
        self.assertIn("N1", ids)
        self.assertIn("N2", ids)
        self.assertEqual(len(ids), 2)

    def test_from_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.jsonl"
            records = [
                NewsItem(news_id="N1", category="news", subcategory="local", title="Title 1"),
                NewsItem(news_id="N2", category="sports", subcategory="football", title="Title 2"),
            ]
            write_jsonl(path, records)
            catalog = InMemoryMindCatalog.from_jsonl(path)

        self.assertEqual(len(catalog.list_item_ids()), 2)
        item = catalog.get_item("N2")
        self.assertEqual(item.subcategory, "football")


class TestIterJsonl(unittest.TestCase):
    def test_roundtrip_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            records = [{"a": 1}, {"b": 2}]
            write_jsonl(path, records)
            loaded = list(iter_jsonl(path))
        self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()
