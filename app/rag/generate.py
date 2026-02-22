from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

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


def _candidate_base_urls(base_url: str) -> list[str]:
    primary = (base_url or "").strip() or "http://localhost:11434"
    parts = urlsplit(primary)
    hostname = (parts.hostname or "").lower()
    candidates: list[str] = [primary]

    def _with_host(new_host: str) -> str:
        netloc = new_host
        if parts.port:
            netloc = f"{new_host}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    if hostname in {"localhost", "127.0.0.1"}:
        for host in ("127.0.0.1", "localhost", "host.docker.internal"):
            url = _with_host(host)
            if url not in candidates:
                candidates.append(url)
    elif hostname == "host.docker.internal":
        for host in ("localhost", "127.0.0.1"):
            url = _with_host(host)
            if url not in candidates:
                candidates.append(url)
    return candidates


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    needles = [
        "connection refused",
        "failed to establish a new connection",
        "[errno 111]",
        "connect error",
        "max retries exceeded",
        "name or service not known",
        "temporary failure in name resolution",
    ]
    return any(n in msg for n in needles)


def _invoke_ollama_with_fallback(
    *,
    messages: list,
    llm_kwargs: dict,
) -> tuple[object, str]:
    original_base_url = str(llm_kwargs.get("base_url") or "http://localhost:11434")
    candidates = _candidate_base_urls(original_base_url)
    last_exc: Exception | None = None

    for candidate in candidates:
        local_kwargs = dict(llm_kwargs)
        local_kwargs["base_url"] = candidate
        llm = ChatOllama(**local_kwargs)
        try:
            return llm.invoke(messages), candidate
        except Exception as exc:
            last_exc = exc
            if not _is_connection_error(exc):
                raise
            continue

    tried = ", ".join(candidates)
    detail = str(last_exc) if last_exc is not None else "unknown connection error"
    raise RuntimeError(
        "Could not connect to Ollama. "
        f"Tried base URLs: {tried}. "
        "Make sure Ollama is running (`ollama serve`) and the model is available "
        f"(`ollama pull {llm_kwargs.get('model', '')}`). "
        f"Original error: {detail}"
    )


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
    doc_ids: Sequence[str] | None = None,
) -> GenerationResult:
    q = (question or "").strip()
    if not q:
        return GenerationResult(answer="", contexts=[], model=model)

    contexts = retriever.retrieve(
        q,
        top_k=top_k,
        min_score=min_score,
        doc_ids=doc_ids,
    )
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

    messages = [
        SystemMessage(content=parts.system),
        HumanMessage(content=parts.user),
    ]
    resp, _used_base_url = _invoke_ollama_with_fallback(
        messages=messages,
        llm_kwargs=llm_kwargs,
    )

    answer = (resp.content or "").strip()
    return GenerationResult(answer=answer, contexts=contexts, model=model)
