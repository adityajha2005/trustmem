"""LangGraph workflow stub for collaborative memory coordination."""

from __future__ import annotations

from typing import Any

from orchestration.state import CoordinationState
from utils.logging import get_logger

logger = get_logger(__name__)

_HAS_LANGGRAPH = False
try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None  # type: ignore[misc, assignment]
    END = None  # type: ignore[misc, assignment]


def _node_ingest(state: CoordinationState) -> dict[str, Any]:
    """Apply pending writes from the active agent. TODO: validation + lineage."""
    logger.debug("ingest: %d pending writes", len(state.get("pending_writes", [])))
    return {"messages": ["ingest_complete"]}


def _node_retrieve(state: CoordinationState) -> dict[str, Any]:
    """Run collaborative retrieval for the active query. TODO: wire pipeline."""
    logger.debug("retrieve: query=%r", state.get("retrieval_query"))
    return {"retrieval_hits": [], "messages": ["retrieve_complete"]}


def _node_reconcile(state: CoordinationState) -> dict[str, Any]:
    """Check consistency and open contradictions. TODO: contradiction registry."""
    return {"messages": ["reconcile_complete"]}


def build_coordination_graph():  # noqa: ANN201
    """
    Build LangGraph StateGraph: ingest -> retrieve -> reconcile -> END.

    Returns None when langgraph is unavailable (import-safe for lightweight tests).
    """
    if not _HAS_LANGGRAPH:
        logger.warning("LangGraph not available; coordination graph disabled")
        return None

    graph: StateGraph = StateGraph(CoordinationState)
    graph.add_node("ingest", _node_ingest)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("reconcile", _node_reconcile)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "reconcile")
    graph.add_edge("reconcile", END)
    return graph.compile()
