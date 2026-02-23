from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from fastapi import HTTPException

# Keep API tests fast by stubbing heavy model imports before app import.
if "sentence_transformers" not in sys.modules:
    fake_module = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            raise RuntimeError("Dummy embedder should not be used in API tests.")

    fake_module.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = fake_module

from app import main as api_main
from app.rag.generate import GenerationResult, INSUFFICIENT_CONTEXT_MSG
from app.rag.ingest import IngestResult
from app.rag.retrieve import RetrievedChunk
from app.rag.schemas import ChatRequest, DeleteDocumentsRequest
from app.storage.metadata_store import ChunkMeta, MetadataStore
from app.storage.vector_store import VectorStore


@pytest.fixture()
def api_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data_dir = tmp_path / "data"
    uploads_dir = data_dir / "uploads"
    faiss_index_path = data_dir / "faiss.index"
    faiss_ids_path = data_dir / "faiss_ids.json"
    metadata_path = data_dir / "metadata.json"

    monkeypatch.setattr(api_main, "DATA_DIR", data_dir)
    monkeypatch.setattr(api_main, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(api_main, "FAISS_INDEX_PATH", faiss_index_path)
    monkeypatch.setattr(api_main, "FAISS_IDS_PATH", faiss_ids_path)
    monkeypatch.setattr(api_main, "METADATA_PATH", metadata_path)

    return {
        "data_dir": data_dir,
        "uploads_dir": uploads_dir,
        "faiss_index_path": faiss_index_path,
        "faiss_ids_path": faiss_ids_path,
        "metadata_path": metadata_path,
    }


def _seed_metadata(metadata_path: Path, metas: list[ChunkMeta]) -> None:
    store = MetadataStore(metadata_path)
    store.add_chunkmeta_list(metas, save=True)


def _seed_vectors(index_path: Path, ids_path: Path, chunk_ids: list[str]) -> None:
    vectors = np.eye(8, dtype=np.float32)[: len(chunk_ids)]
    store = VectorStore(index_path=index_path, ids_path=ids_path, dim=8, normalize=True)
    store.add(vectors, chunk_ids, save=True)


class _FakeUploadFile:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload

    async def close(self) -> None:
        return None


def test_upload_document_and_list_documents(
    api_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"%PDF-1.4 fake"
    expected_doc_id = api_main._build_doc_id("sample.pdf", payload)

    def fake_ingest_pdf(
        pdf_path,
        *,
        doc_id=None,
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        faiss_index_path="data/faiss.index",
        faiss_ids_path="data/faiss_ids.json",
        metadata_path="data/metadata.json",
        embedding_device=None,
        log_timing=True,
    ):
        assert Path(pdf_path).exists()
        _seed_metadata(
            Path(metadata_path),
            [
                ChunkMeta(
                    chunk_id=f"{doc_id}::p0001::c0000",
                    doc_id=str(doc_id),
                    text="chunk text",
                    page_start=1,
                    page_end=1,
                    chunk_index=0,
                )
            ],
        )
        return IngestResult(doc_id=str(doc_id), pages=1, chunks=1, index_size=1)

    monkeypatch.setattr(api_main, "ingest_pdf", fake_ingest_pdf)

    upload = _FakeUploadFile(filename="sample.pdf", payload=payload)
    out = asyncio.run(api_main.upload_document(upload))
    assert out.doc_id == expected_doc_id
    assert out.pages == 1
    assert out.chunks == 1
    assert out.index_size == 1

    docs = api_main.list_documents()
    assert docs.documents == [expected_doc_id]

def test_chat_returns_answer_and_citations(
    api_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_metadata(
        api_paths["metadata_path"],
        [
            ChunkMeta(
                chunk_id="docA::p0001::c0000",
                doc_id="docA",
                text="Paris is the capital of France.",
                page_start=1,
                page_end=1,
                chunk_index=0,
            )
        ],
    )

    def fake_generate(*args, **kwargs):
        return GenerationResult(
            answer="Paris is the capital of France [1]\n\nSources: [1]",
            contexts=[
                RetrievedChunk(
                    chunk_id="docA::p0001::c0000",
                    doc_id="docA",
                    text="Paris is the capital of France.",
                    page_start=1,
                    page_end=1,
                    chunk_index=0,
                    score=0.91,
                )
            ],
            model="llama3.2",
        )

    monkeypatch.setattr(api_main, "_build_retriever", lambda metadata_store: object())
    monkeypatch.setattr(api_main, "generate", fake_generate)

    body = api_main.chat(ChatRequest(question="What is the capital of France?"))

    assert body.answer.startswith("Paris is the capital")
    assert body.refusal is False
    assert body.model == "llama3.2"
    assert len(body.citations) == 1
    assert body.citations[0].doc == "docA"
    assert body.citations[0].page == 1
    assert body.citations[0].chunk_id == "docA::p0001::c0000"


def test_chat_doc_ids_filtering_and_validation(
    api_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_metadata(
        api_paths["metadata_path"],
        [
            ChunkMeta(chunk_id="docA::c0000", doc_id="docA", text="a", chunk_index=0),
            ChunkMeta(chunk_id="docB::c0000", doc_id="docB", text="b", chunk_index=0),
        ],
    )

    captured: dict[str, object] = {}

    def fake_generate(*args, **kwargs):
        captured["doc_ids"] = kwargs.get("doc_ids")
        return GenerationResult(answer="ok", contexts=[], model="llama3.2")

    monkeypatch.setattr(api_main, "_build_retriever", lambda metadata_store: object())
    monkeypatch.setattr(api_main, "generate", fake_generate)

    out = api_main.chat(
        ChatRequest(question="q", doc_ids=["docB", "docA", "docA", " "]),
    )
    assert out.answer == "ok"
    assert captured["doc_ids"] == ["docA", "docB"]

    with pytest.raises(HTTPException) as exc:
        api_main.chat(ChatRequest(question="q", doc_ids=["missing_doc"]))
    assert exc.value.status_code == 404
    assert "Unknown doc_ids" in exc.value.detail


def test_chat_refusal_when_no_context(
    api_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_metadata(
        api_paths["metadata_path"],
        [ChunkMeta(chunk_id="docA::c0000", doc_id="docA", text="a", chunk_index=0)],
    )

    monkeypatch.setattr(api_main, "_build_retriever", lambda metadata_store: object())
    monkeypatch.setattr(
        api_main,
        "generate",
        lambda *args, **kwargs: GenerationResult(
            answer=INSUFFICIENT_CONTEXT_MSG,
            contexts=[],
            model="llama3.2",
        ),
    )

    body = api_main.chat(ChatRequest(question="q"))
    assert body.answer == INSUFFICIENT_CONTEXT_MSG
    assert body.refusal is True
    assert body.citations == []


def test_delete_documents_success(api_paths: dict[str, Path]) -> None:
    _seed_metadata(
        api_paths["metadata_path"],
        [
            ChunkMeta(chunk_id="docA::c0000", doc_id="docA", text="a0", chunk_index=0),
            ChunkMeta(chunk_id="docA::c0001", doc_id="docA", text="a1", chunk_index=1),
            ChunkMeta(chunk_id="docB::c0000", doc_id="docB", text="b0", chunk_index=0),
        ],
    )
    _seed_vectors(
        api_paths["faiss_index_path"],
        api_paths["faiss_ids_path"],
        ["docA::c0000", "docA::c0001", "docB::c0000"],
    )

    out = api_main.delete_documents(DeleteDocumentsRequest(doc_ids=["docA"]))
    assert out.deleted_doc_ids == ["docA"]
    assert out.deleted_chunks == 2
    assert out.remaining_docs == 1
    assert out.remaining_chunks == 1
    assert out.index_size == 1
    assert api_main.list_documents().documents == ["docB"]


def test_delete_documents_validation_errors(api_paths: dict[str, Path]) -> None:
    _seed_metadata(
        api_paths["metadata_path"],
        [ChunkMeta(chunk_id="docA::c0000", doc_id="docA", text="a0", chunk_index=0)],
    )
    _seed_vectors(
        api_paths["faiss_index_path"],
        api_paths["faiss_ids_path"],
        ["docA::c0000"],
    )

    with pytest.raises(HTTPException) as exc:
        api_main.delete_documents(DeleteDocumentsRequest(doc_ids=["missing_doc"]))
    assert exc.value.status_code == 404
    assert "Unknown doc_ids" in exc.value.detail

    with pytest.raises(HTTPException) as exc2:
        api_main.delete_documents(DeleteDocumentsRequest(doc_ids=[" "]))
    assert exc2.value.status_code == 400
    assert "doc_ids cannot be empty" in exc2.value.detail


def test_delete_documents_returns_500_on_index_ids_inconsistency(
    api_paths: dict[str, Path],
) -> None:
    _seed_metadata(
        api_paths["metadata_path"],
        [ChunkMeta(chunk_id="docA::c0000", doc_id="docA", text="a0", chunk_index=0)],
    )
    # Create only ids mapping without FAISS index to simulate storage inconsistency.
    api_paths["faiss_ids_path"].write_text('["docA::c0000"]', encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        api_main.delete_documents(DeleteDocumentsRequest(doc_ids=["docA"]))
    assert exc.value.status_code == 500
    assert "Delete failed" in exc.value.detail
    assert "out of sync" in exc.value.detail

    # Failure should not delete metadata.
    assert api_main.list_documents().documents == ["docA"]
