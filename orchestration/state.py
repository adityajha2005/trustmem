"""LangGraph state for multi-agent memory coordination."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from memory.schema import Memory


def _merge_records(existing: list[Memory], new: list[Memory]) -> list[Memory]:
    seen = {r.id for r in existing}
    out = list(existing)
    for r in new:
        if r.id not in seen:
            out.append(r)
            seen.add(r.id)
    return out


class CoordinationState(TypedDict, total=False):
    """Shared graph state across collaborating agents."""

    task_id: str
    active_agent_id: str
    pending_writes: Annotated[list[Memory], _merge_records]
    retrieval_query: str
    retrieval_hits: list[dict[str, Any]]
    open_contradictions: list[str]
    messages: Annotated[list[str], lambda a, b: a + b]
