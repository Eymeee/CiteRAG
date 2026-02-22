from __future__ import annotations

from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import build_prompt_parts
from .retrieve import RetrievedChunk

INSUFFICIENT_CONTEXT_MSG = (
    "I don't have enough information in the uploaded documents to answer that."
)


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    contexts: list[RetrievedChunk]
    model: str


def generate(
    question: str,
    *,
    retriever,
    model: str,
    base_url: str = "http://localhost:11434",
    top_k: int = 6,
    min_score: float | None = None,
    max_context_chars: int = 12_000,
    max_chunk_chars: int = 1_500,
    temperature: float = 0.2,
    num_ctx: int | None = None,
    include_score_in_sources: bool = False,
) -> GenerationResult:
    q = (question or "").strip()
    if not q:
        return GenerationResult(answer="", contexts=[], model=model)

    contexts = retriever.retrieve(q, top_k=top_k, min_score=min_score)
    if not contexts:
        return GenerationResult(
            answer=INSUFFICIENT_CONTEXT_MSG,
            contexts=[],
            model=model,
        )

    parts = build_prompt_parts(
        q,
        contexts,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        include_score=include_score_in_sources,
    )

    llm_kwargs: dict = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
    }
    if num_ctx is not None:
        llm_kwargs["num_ctx"] = int(num_ctx)

    llm = ChatOllama(**llm_kwargs)

    resp = llm.invoke(
        [
            SystemMessage(content=parts.system),
            HumanMessage(content=parts.user),
        ]
    )

    answer = (resp.content or "").strip()
    return GenerationResult(answer=answer, contexts=contexts, model=model)