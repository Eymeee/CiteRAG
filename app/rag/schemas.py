from __future__ import annotations

from pydantic import BaseModel, Field


class UploadDocumentResponse(BaseModel):
    doc_id: str
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)
    index_size: int = Field(ge=0)


class DocumentsResponse(BaseModel):
    documents: list[str]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[str] | None = None
    collection_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=50)
    min_score: float | None = None


class Citation(BaseModel):
    id: int = Field(ge=1, description="1-based citation number like [1], [2].")
    doc: str
    page: int | None = Field(default=None, ge=1)
    chunk_id: str
    text_preview: str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refusal: bool = False
    model: str | None = None
