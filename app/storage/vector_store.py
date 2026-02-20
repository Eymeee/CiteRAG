from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float # higher is better for cosine/inner product


class VectorStore:
    """
    Minimal FAISS vector store for RAG.

    - Uses cosine similarity via inner product + L2 normalization.
    - Stores chunk_ids in a parallel JSON file (positional mapping).
    - No deletes/updates to keep it simple and reliable.
    """
    def __init__(
        self, 
        index_path: str | Path, 
        ids_path: str | Path, 
        *, dim: int, normalize: bool = True
        ):

        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self.dim = int(dim)
        self.normalize = bool(normalize)

        self._index: faiss.Index = faiss.IndexFlatIP(self.dim)
        self._ids: list[str] = []

        # Auto-load if present
        if self.index_path.exists() and self.ids_path.exists():
            self.load()


    def __len__(self) -> int :
        return len(self._ids)
        

    @property
    def ids(self) -> list[str]:
        return self._ids


    def add(self, embeddings: np.ndarray, chunk_ids: Sequence[str], *, save: bool = True) -> None:
        """
        Add embeddings + their chunk_ids.

        embeddings: shape (n, dim), dtype float32/64
        chunk_ids:  length n, each is a stable string id (that matches metadata store)
        """
        vecs = _to_float32_2d(embeddings, dim=self.dim)
        ids = list(chunk_ids)

        if vecs.shape[0] != len(ids):
            raise ValueError(
                f"\n --- embeddings rows ({vecs.shape[0]}) != chunk_ids ({len(ids)})"
            )
        if not ids: 
            return

        if self.normalize:
            faiss.normalize_L2(vecs)

        self._index.add(vecs)
        self._ids.extend(ids)

        if save:
            self.save()


    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if len(self._ids) == 0:
            return []

        q = _to_float32_2d(query_embedding, dim=self.dim)
        if q.shape[0] != 1:
            raise ValueError("query_embedding must be a single vector (dim,) or (1, dim)")

        if self.normalize:
            faiss.normalize_L2(q)

        scores, idxs = self._index.search(q, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._ids) :
                continue
            results.append(
                SearchResult(
                    chunk_id=self._ids[idx], score=float(score)
                )
            )
            
        return results


    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.ids_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_path))

        tmp = self.ids_path.with_suffix(self.ids_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._ids, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.ids_path)
    

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.ids_path.exists():
            raise FileNotFoundError(f"IDs file not found: {self.ids_path}")

        index = faiss.read_index(str(self.index_path))
        ids = json.loads(self.ids_path.read_text(encoding="utf-8"))

        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError("ids_path must contain a JSON list of strings")

        if index.d != self.dim:
            raise ValueError(f"Index dim ({index.d}) != expected dim ({self.dim})")

        if index.ntotal != len(ids):
            raise ValueError(
                f"Index vectors ({index.ntotal}) != ids ({len(ids)})"
            )

        self._index = index
        self._ids = ids


# -------------------------------------------
#              Helper functions 
# -------------------------------------------

def _to_float32_2d(arr: np.ndarray, *, dim: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")

    if a.shape[1] != dim:
        raise ValueError(f"Expected dim={dim}, got {a.shape[1]}")

    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a