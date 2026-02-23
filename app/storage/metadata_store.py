from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ChunkMeta:
    """
    Minimal metadata needed to map a retrieved vector back to its source + citations
    """
    chunk_id: str # unique identifier string for this chunk
    doc_id: str
    text: str

    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_index: int = 0 # the order of the chunk within the document


class MetadataStore:
    """
    Simple JSON metadata store:

    - In-memory dict for O(1) lookups by chunk_id
    - Single JSON file on disk for persistence
    - Atomic writes to avoid corruption
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._by_id: Dict[str, ChunkMeta] = {}

        if self.path.exists():
            self.load()


    def __len__(self) -> int:
        return len(self._by_id)


    def add_chunkmeta_list(self, metas: List[ChunkMeta], *, save: bool = True) -> None:
        for meta in metas:
            self._by_id[meta.chunk_id] = meta

        if save: 
            self.save()


    def get(self, chunk_id: str) -> Optional[ChunkMeta]:
        return self._by_id.get(chunk_id)


    def all(self) -> List[ChunkMeta]:
        return list(self._by_id.values())


    def list_docs(self) -> List[str]:
        metas = list(self._by_id.values())
        doc_ids = {m.doc_id for m in metas}
        return sorted(doc_ids)


    def delete(self, chunk_id: str, *, save: bool = True) -> bool:
        existed = chunk_id in self._by_id
        if existed:
            del self._by_id[chunk_id]
            if save:
                self.save()
        
        return existed


    def clear(self, *, save: bool = True) -> None:
        self._by_id.clear()
        if save:
            self.save()


    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))

        chunks = data.get("chunks") or {}
        by_id: Dict[str, ChunkMeta] = {}

        for chunk_id, raw in chunks.items():
            by_id[chunk_id] = ChunkMeta(
                chunk_id=chunk_id,
                doc_id=str(raw["doc_id"]),
                text=raw.get("text", ""),
                page_start=raw.get("page_start"),
                page_end=raw.get("page_end"),
                chunk_index=int(raw.get("chunk_index", 0)),
            )

        self._by_id = by_id


    def save(self) -> None:
        payload = {
            "chunks": {cid: _chunkmeta_to_dict(cm) for cid, cm in self._by_id.items()},
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


# -------------------------------------------
#              Helper functions 
# -------------------------------------------

def _chunkmeta_to_dict(cm: ChunkMeta) -> dict:
    d = asdict(cm)
    d.pop("chunk_id", None)  # chunk_id is the dict key
    return d
