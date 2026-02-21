from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    score: float


class Retriever:
    """
    Minimal retrieval:
    - embed query (SentenceTransformer)
    - VectorStore.search() -> list[SearchResult]
    - MetadataStore.get(chunk_id) -> ChunkMeta
    - return ranked RetrievedChunk list
    """
    def __init__(
        self,
        *,
        vector_store,
        metadata_store,
        embed_model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.embedder = SentenceTransformer(embed_model_name, device=device)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        dedupe: bool = True,
    ) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []

        q_vec = self.embedder.encode([query])  # shape (1, dim)
        hits = self.vector_store.search(q_vec, top_k=top_k)

        out: list[RetrievedChunk] = []
        seen: set[str] = set()

        for hit in hits:
            if min_score is not None and hit.score < min_score:
                continue

            if dedupe and hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)

            meta = self.metadata_store.get(hit.chunk_id)
            if meta is None:
                continue

            out.append(
                RetrievedChunk(
                    chunk_id=meta.chunk_id,
                    doc_id=meta.doc_id,
                    text=meta.text,
                    page_start=meta.page_start,
                    page_end=meta.page_end,
                    chunk_index=meta.chunk_index,
                    score=float(hit.score),
                )
            )

        return out

    def retrieve_texts(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[str]:
        """
        Convenience helper for prompt construction: returns only chunk texts in rank order.
        """
        return [c.text for c in self.retrieve(query, top_k=top_k, min_score=min_score)]