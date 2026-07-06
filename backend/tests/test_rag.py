from unittest.mock import MagicMock

import requests

from app import rag
from app.config import TOP_K


def hit(text, score, source="doc.pdf", page=None):
    return {"text": text, "source": source, "page": page, "score": score}


def match(text, score, source="doc.pdf"):
    return {"score": score, "metadata": {"text": text, "source": source}}


def test_filter_drops_below_min_score():
    hits = rag._filter_hits([match("a", 0.5), match("b", 0.1)])
    assert [h["text"] for h in hits] == ["a"]


def test_filter_relative_cutoff():
    hits = rag._filter_hits([match("a", 0.9), match("b", 0.6), match("c", 0.25)])
    # 0.25 < 0.9 * 0.35 -> dropped even though above MIN_SCORE
    assert [h["text"] for h in hits] == ["a", "b"]


def test_dedup_keeps_higher_scored_twin():
    text = "the quick brown fox jumps over the lazy dog every single day"
    hits = [hit(text, 0.9), hit(text + " indeed", 0.8), hit("completely different content here now", 0.7)]
    out = rag._dedup(hits)
    assert len(out) == 2
    assert out[0]["score"] == 0.9


def test_source_prefix_deterministic():
    assert rag.source_prefix("a.pdf") == rag.source_prefix("a.pdf")
    assert rag.source_prefix("a.pdf") != rag.source_prefix("b.pdf")
    assert len(rag.source_prefix("a.pdf")) == 16


def test_rerank_falls_back_on_failure(monkeypatch):
    def boom(url, payload):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(rag, "_jina", boom)
    hits = [hit(f"t{i}", 0.9 - i * 0.1) for i in range(8)]
    out = rag._rerank("q", hits)
    assert out == hits[:TOP_K]


def test_rerank_reorders(monkeypatch):
    monkeypatch.setattr(rag, "_jina", lambda url, payload: {"results": [{"index": 2}, {"index": 0}]})
    hits = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    out = rag._rerank("q", hits)
    assert [h["text"] for h in out] == ["c", "a"]


def test_retrieve_passes_source_filter(monkeypatch):
    index = MagicMock()
    index.query.return_value = {"matches": []}
    monkeypatch.setattr(rag, "_get_index", lambda: index)
    monkeypatch.setattr(rag, "embed_query", lambda q: [0.0] * 4)

    rag.retrieve("q", namespace="ns", source="doc.pdf")
    assert index.query.call_args.kwargs["filter"] == {"source": {"$eq": "doc.pdf"}}

    rag.retrieve("q", namespace="ns")
    assert index.query.call_args.kwargs["filter"] is None


def test_store_chunks_replaces_old_vectors(monkeypatch):
    index = MagicMock()
    page = MagicMock()
    page.vectors = [MagicMock(id="old#0"), MagicMock(id="old#1")]
    index.list.return_value = iter([page])
    monkeypatch.setattr(rag, "_get_index", lambda: index)
    monkeypatch.setattr(rag, "embed_passages", lambda texts: [[0.0] * 4 for _ in texts])

    chunks = [{"text": "hello", "page": 3}, {"text": "world", "page": None}]
    stored = rag.store_chunks(chunks, source="doc.pdf", namespace="ns")

    assert stored == 2
    index.delete.assert_called_once_with(ids=["old#0", "old#1"], namespace="ns")
    vectors = index.upsert.call_args.kwargs["vectors"]
    prefix = rag.source_prefix("doc.pdf")
    assert [v["id"] for v in vectors] == [f"{prefix}#0", f"{prefix}#1"]
    assert vectors[0]["metadata"]["page"] == 3
    assert "page" not in vectors[1]["metadata"]
