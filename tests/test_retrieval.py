from dataclasses import dataclass

import numpy as np

from app.rag import retrieve as retrieve_module
from app.rag.retrieve import Retriever
from app.storage.metadata_store import ChunkMeta


@dataclass(frozen=True)
class _FakeHit:
    chunk_id: str
    score: float


class _FakeVectorStore:
    def __init__(self, hits):
        self._hits = list(hits)
        self.calls = []

    def search(self, query_embedding, top_k=5):
        self.calls.append({"shape": tuple(query_embedding.shape), "top_k": top_k})
        return self._hits[:top_k]


class _FakeMetadataStore:
    def __init__(self, by_id):
        self._by_id = dict(by_id)

    def get(self, chunk_id):
        return self._by_id.get(chunk_id)


class _FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        # Shape (1, dim), enough for retrieval flow.
        assert len(texts) == 1
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def test_retrieve_returns_filtered_ranked_chunks(monkeypatch):
    monkeypatch.setattr(retrieve_module, "SentenceTransformer", _FakeEmbedder)

    vector_store = _FakeVectorStore(
        [
            _FakeHit("c1", 0.93),
            _FakeHit("c_missing", 0.90),  # missing metadata should be skipped
            _FakeHit("c2", 0.20),  # filtered by min_score
        ]
    )
    metadata_store = _FakeMetadataStore(
        {
            "c1": ChunkMeta(
                chunk_id="c1",
                doc_id="docA",
                text="alpha",
                page_start=1,
                page_end=1,
                chunk_index=0,
            ),
            "c2": ChunkMeta(
                chunk_id="c2",
                doc_id="docA",
                text="beta",
                page_start=2,
                page_end=2,
                chunk_index=1,
            ),
        }
    )

    retriever = Retriever(vector_store=vector_store, metadata_store=metadata_store)
    out = retriever.retrieve("What is alpha?", top_k=3, min_score=0.5)

    assert len(out) == 1
    assert out[0].chunk_id == "c1"
    assert out[0].doc_id == "docA"
    assert out[0].text == "alpha"
    assert out[0].score == 0.93

    assert vector_store.calls
    assert vector_store.calls[0]["top_k"] == 3
    assert vector_store.calls[0]["shape"] == (1, 4)


def test_retrieve_dedupes_by_default(monkeypatch):
    monkeypatch.setattr(retrieve_module, "SentenceTransformer", _FakeEmbedder)

    vector_store = _FakeVectorStore(
        [
            _FakeHit("c1", 0.95),
            _FakeHit("c1", 0.94),
            _FakeHit("c2", 0.90),
        ]
    )
    metadata_store = _FakeMetadataStore(
        {
            "c1": ChunkMeta(chunk_id="c1", doc_id="doc", text="t1", chunk_index=0),
            "c2": ChunkMeta(chunk_id="c2", doc_id="doc", text="t2", chunk_index=1),
        }
    )

    retriever = Retriever(vector_store=vector_store, metadata_store=metadata_store)

    deduped = retriever.retrieve("q", top_k=10, dedupe=True)
    not_deduped = retriever.retrieve("q", top_k=10, dedupe=False)

    assert [c.chunk_id for c in deduped] == ["c1", "c2"]
    assert [c.chunk_id for c in not_deduped] == ["c1", "c1", "c2"]


def test_retrieve_blank_query_short_circuits(monkeypatch):
    monkeypatch.setattr(retrieve_module, "SentenceTransformer", _FakeEmbedder)

    vector_store = _FakeVectorStore([_FakeHit("c1", 0.9)])
    metadata_store = _FakeMetadataStore({})
    retriever = Retriever(vector_store=vector_store, metadata_store=metadata_store)

    assert retriever.retrieve("   ") == []
    assert vector_store.calls == []


def test_retrieve_texts_returns_only_text(monkeypatch):
    monkeypatch.setattr(retrieve_module, "SentenceTransformer", _FakeEmbedder)

    vector_store = _FakeVectorStore([_FakeHit("c1", 0.9), _FakeHit("c2", 0.8)])
    metadata_store = _FakeMetadataStore(
        {
            "c1": ChunkMeta(chunk_id="c1", doc_id="doc", text="first", chunk_index=0),
            "c2": ChunkMeta(chunk_id="c2", doc_id="doc", text="second", chunk_index=1),
        }
    )
    retriever = Retriever(vector_store=vector_store, metadata_store=metadata_store)

    assert retriever.retrieve_texts("query", top_k=2) == ["first", "second"]


def test_retrieve_filters_by_doc_ids(monkeypatch):
    monkeypatch.setattr(retrieve_module, "SentenceTransformer", _FakeEmbedder)

    vector_store = _FakeVectorStore([_FakeHit("c1", 0.9), _FakeHit("c2", 0.8)])
    metadata_store = _FakeMetadataStore(
        {
            "c1": ChunkMeta(chunk_id="c1", doc_id="docA", text="from A", chunk_index=0),
            "c2": ChunkMeta(chunk_id="c2", doc_id="docB", text="from B", chunk_index=1),
        }
    )
    retriever = Retriever(vector_store=vector_store, metadata_store=metadata_store)

    only_a = retriever.retrieve("query", top_k=2, doc_ids=["docA"])
    only_b_texts = retriever.retrieve_texts("query", top_k=2, doc_ids=["docB"])

    assert [c.doc_id for c in only_a] == ["docA"]
    assert only_b_texts == ["from B"]
