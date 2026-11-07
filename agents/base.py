"""Reference agent client wiring repository + retrieval pipeline."""

from __future__ import annotations

from uuid import UUID

from memory.schema import Memory
from retrieval.context import RetrievalContext
from retrieval.retriever import CollaborativeMemoryRetriever, RetrievalResult, build_retriever
from storage.repository import MemoryRepository


class BaseAgentClient:
    def __init__(
        self,
        agent_id: str,
        repository: MemoryRepository,
        *,
        task_id: str | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._task_id = task_id
        self._repo = repository
        self._retriever = build_retriever(repository)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def write(self, memory: Memory) -> Memory:
        if memory.agent != self._agent_id:
            memory = memory.model_copy(update={"agent": self._agent_id})
        return self._repo.add(memory)

    def read(
        self,
        query: str,
        ctx: RetrievalContext | None = None,
        *,
        top_k: int = 5,
    ) -> RetrievalResult:
        context = ctx or RetrievalContext(agent_id=self._agent_id, task_id=self._task_id)
        return self._retriever.retrieve(query, context, top_k=top_k)

    def resolve_contradiction(self, claim_id: UUID, resolution: str) -> None:
        # TODO: Contradiction resolution — apply policy via store.supersede / archive
        raise NotImplementedError(f"resolve_contradiction({claim_id}, {resolution})")
