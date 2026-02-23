from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.storage.metadata_store import MetadataStore


@dataclass(frozen=True)
class DeleteDocumentsResult:
    requested_doc_ids: list[str]
    deleted_doc_ids: list[str]
    deleted_chunks: int
    remaining_docs: int
    remaining_chunks: int
    index_size: int


def hard_delete_documents(
    *,
    doc_ids: Sequence[str],
    metadata_path: str | Path = "data/metadata.json",
    faiss_index_path: str | Path = "data/faiss.index",
    faiss_ids_path: str | Path = "data/faiss_ids.json",
) -> DeleteDocumentsResult:
    requested_doc_ids = sorted({d.strip() for d in doc_ids if d and d.strip()})
    if not requested_doc_ids:
        raise ValueError("doc_ids cannot be empty.")

    metadata_store = MetadataStore(metadata_path)
    if len(metadata_store) == 0:
        raise ValueError("No indexed documents found.")

    available_doc_ids = set(metadata_store.list_docs())
    unknown_doc_ids = [d for d in requested_doc_ids if d not in available_doc_ids]
    if unknown_doc_ids:
        raise ValueError(f"Unknown doc_ids: {', '.join(unknown_doc_ids)}")

    deleted_chunk_ids = [
        meta.chunk_id for meta in metadata_store.all() if meta.doc_id in requested_doc_ids
    ]
    deleted_chunk_id_set = set(deleted_chunk_ids)

    index_size = _rebuild_index_without_chunk_ids(
        index_path=faiss_index_path,
        ids_path=faiss_ids_path,
        deleted_chunk_ids=deleted_chunk_id_set,
    )

    for chunk_id in deleted_chunk_ids:
        metadata_store.delete(chunk_id, save=False)
    metadata_store.save()

    return DeleteDocumentsResult(
        requested_doc_ids=requested_doc_ids,
        deleted_doc_ids=requested_doc_ids,
        deleted_chunks=len(deleted_chunk_ids),
        remaining_docs=len(metadata_store.list_docs()),
        remaining_chunks=len(metadata_store),
        index_size=index_size,
    )


def _rebuild_index_without_chunk_ids(
    *,
    index_path: str | Path,
    ids_path: str | Path,
    deleted_chunk_ids: set[str],
) -> int:
    index_path = Path(index_path)
    ids_path = Path(ids_path)

    if not index_path.exists() and not ids_path.exists():
        return 0
    if index_path.exists() != ids_path.exists():
        raise ValueError("FAISS index and ids mapping are out of sync on disk.")

    index = faiss.read_index(str(index_path))
    ids = _load_ids(ids_path)

    if index.ntotal != len(ids):
        raise ValueError(
            f"Index vectors ({index.ntotal}) != ids ({len(ids)}) while rebuilding index."
        )

    kept_positions: list[int] = []
    kept_ids: list[str] = []
    for pos, chunk_id in enumerate(ids):
        if chunk_id in deleted_chunk_ids:
            continue
        kept_positions.append(pos)
        kept_ids.append(chunk_id)

    rebuilt_index = _new_empty_index(dim=int(index.d), metric_type=int(index.metric_type))
    if kept_positions:
        vectors = _extract_vectors(index=index, positions=kept_positions, dim=int(index.d))
        rebuilt_index.add(vectors)

    faiss.write_index(rebuilt_index, str(index_path))
    _save_ids(ids_path, kept_ids)
    return len(kept_ids)


def _new_empty_index(*, dim: int, metric_type: int) -> faiss.Index:
    if metric_type == faiss.METRIC_L2:
        return faiss.IndexFlatL2(dim)
    return faiss.IndexFlatIP(dim)


def _extract_vectors(*, index: faiss.Index, positions: list[int], dim: int) -> np.ndarray:
    vectors = np.empty((len(positions), dim), dtype=np.float32)
    for out_pos, source_pos in enumerate(positions):
        vectors[out_pos] = index.reconstruct(int(source_pos))
    return vectors


def _load_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError("ids file must contain a JSON list of strings.")
    return raw


def _save_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
