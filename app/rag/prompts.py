from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence


_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class PromptParts:
    system: str
    user: str


SYSTEM_PROMPT = """You are CiteRAG.

Rules:
- Answer the QUESTION using ONLY the SOURCES provided.
- If the SOURCES do not contain enough information, say you don't know.
- Cite sources using bracket numbers like [1], [2] for every factual claim.
- Do not invent citations. Only use citations that exist in SOURCES.
"""


USER_PROMPT_TEMPLATE = """SOURCES:
{sources}

QUESTION:
{question}

Write a helpful answer with citations.
"""


def format_contexts(
    chunks: Sequence,
    *,
    max_context_chars: int = 12_000,
    max_chunk_chars: int = 1_500,
    include_score: bool = False,
) -> str:
    """
    Convert retrieved chunks into a prompt-ready SOURCES block.

    Output format:
      [1] (doc_id p.X-Y) chunk text...
      [2] (doc_id p.Z)   chunk text...

    Notes:
    - Keeps incoming order (so pass already-ranked chunks).
    - Enforces per-chunk and total budget guardrails.
    """
    lines: list[str] = []
    used = 0

    for i, c in enumerate(chunks, start=1):
        doc_id = getattr(c, "doc_id", "unknown")

        ps = getattr(c, "page_start", None)
        pe = getattr(c, "page_end", None)

        if ps is None:
            prov = f"({doc_id})"
        elif pe in (None, ps):
            prov = f"({doc_id} p.{ps})"
        else:
            prov = f"({doc_id} p.{ps}-{pe})"

        header = f"[{i}] {prov}"
        if include_score:
            header += f" score={float(getattr(c, 'score', 0.0)):.4f}"

        text = (getattr(c, "text", "") or "").strip()
        text = _WS.sub(" ", text)

        if len(text) > max_chunk_chars:
            text = text[: max_chunk_chars - 1].rstrip() + "…"

        block = f"{header} {text}".strip()

        next_len = len(block) + (1 if lines else 0)  # newline if not first
        if used + next_len > max_context_chars:
            break

        lines.append(block)
        used += next_len

    return "\n".join(lines)


def build_prompt_parts(
    question: str,
    chunks: Sequence,
    *,
    max_context_chars: int = 12_000,
    max_chunk_chars: int = 1_500,
    include_score: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
) -> PromptParts:
    """
    Returns (system, user) prompt strings for chat-style generation.
    """
    q = (question or "").strip()
    sources = format_contexts(
        chunks,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        include_score=include_score,
    )
    user = USER_PROMPT_TEMPLATE.format(sources=sources, question=q)
    return PromptParts(system=system_prompt, user=user)


def build_single_prompt(
    question: str,
    chunks: Sequence,
    *,
    max_context_chars: int = 12_000,
    max_chunk_chars: int = 1_500,
    include_score: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Convenience for non-chat completion APIs: returns one combined prompt.
    """
    parts = build_prompt_parts(
        question,
        chunks,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        include_score=include_score,
        system_prompt=system_prompt,
    )
    return f"{parts.system}\n\n{parts.user}"