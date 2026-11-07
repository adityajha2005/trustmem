"""Trust-aware scoring for collaborative retrieval."""

from __future__ import annotations

from memory.schema import Memory
from retrieval.context import RetrievalContext


def trust_weight(record: Memory, ctx: RetrievalContext) -> float:
    """
    Weight a candidate by stored confidence and agent-relative policy.

    Self-authored local memories are fully trusted; shared memories use confidence.
    """
    if record.agent == ctx.agent_id:
        return 1.0
    if record.confidence < ctx.min_trust:
        return 0.0
    # TODO: Decay trust over time; boost corroborated multi-agent writes
    return record.confidence
