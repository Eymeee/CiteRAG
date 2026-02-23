import json

import numpy as np
import pytest

from app.rag.delete import hard_delete_documents
from app.storage.metadata_store import ChunkMeta, MetadataStore
from app.storage.vector_store import VectorStore


def _seed_corpus(metadata_path, index_path, ids_path) -> None:
    chunk_ids = ["docA::c0000", "docA::c0001", "docB::c0000"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vector_store = VectorStore(
        index_path=index_path,
        ids_path=ids_path,
        dim=3,
        normalize=True,
    )
    vector_store.add(embeddings, chunk_ids, save=True)

    metadata_store = MetadataStore(metadata_path)
    metadata_store.add_chunkmeta_list(
        [
            ChunkMeta(chunk_id="docA::c0000", doc_id="docA", text="a0", chunk_index=0),
            ChunkMeta(chunk_id="docA::c0001", doc_id="docA", text="a1", chunk_index=1),
            ChunkMeta(chunk_id="docB::c0000", doc_id="docB", text="b0", chunk_index=0),
        ],
        save=True,
    )


def test_hard_delete_documents_rebuilds_index(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    index_path = tmp_path / "faiss.index"
    ids_path = tmp_path / "faiss_ids.json"
    _seed_corpus(metadata_path, index_path, ids_path)

    result = hard_delete_documents(
        doc_ids=["docA"],
        metadata_path=metadata_path,
        faiss_index_path=index_path,
        faiss_ids_path=ids_path,
    )

    assert result.deleted_doc_ids == ["docA"]
    assert result.deleted_chunks == 2
    assert result.remaining_docs == 1
    assert result.remaining_chunks == 1
    assert result.index_size == 1

    kept_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    assert kept_ids == ["docB::c0000"]
    assert MetadataStore(metadata_path).list_docs() == ["docB"]

    store = VectorStore(index_path=index_path, ids_path=ids_path, dim=3, normalize=True)
    hits = store.search(np.array([0.0, 0.0, 3.0], dtype=np.float32), top_k=1)
    assert hits[0].chunk_id == "docB::c0000"


def test_hard_delete_documents_rejects_unknown_doc_id(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    index_path = tmp_path / "faiss.index"
    ids_path = tmp_path / "faiss_ids.json"
    _seed_corpus(metadata_path, index_path, ids_path)

    with pytest.raises(ValueError, match="Unknown doc_ids"):
        hard_delete_documents(
            doc_ids=["missing_doc"],
            metadata_path=metadata_path,
            faiss_index_path=index_path,
            faiss_ids_path=ids_path,
        )
