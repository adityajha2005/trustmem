"""Memory lineage traversal and active-state resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from memory.schema import Memory
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LineageIndex:
    """
    In-memory index over a memory set using parent_memory and supersedes links.

    Supports planner → coder → verifier workflows by resolving the latest
    trusted revision in a supersession chain.
    """

    memories: list[Memory] = field(default_factory=list)
    _by_id: dict[UUID, Memory] = field(default_factory=dict, init=False)
    _successor_of: dict[UUID, Memory] = field(default_factory=dict, init=False)
    _superseded_ids: set[UUID] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._by_id = {m.id: m for m in self.memories}
        for memory in self.memories:
            if memory.supersedes is not None:
                self._successor_of[memory.supersedes] = memory
                self._superseded_ids.add(memory.supersedes)

    @classmethod
    def from_memories(cls, memories: list[Memory]) -> LineageIndex:
        return cls(memories=list(memories))

    def get(self, memory_id: UUID) -> Memory | None:
        return self._by_id.get(memory_id)

    def is_superseded(self, memory_id: UUID) -> bool:
        """True when another memory has supersedes pointing at this id."""
        return memory_id in self._superseded_ids

    def is_stale(self, memory: Memory) -> bool:
        """Archived or superseded by a newer memory."""
        return memory.archived or self.is_superseded(memory.id)

    def get_successor(self, memory_id: UUID) -> Memory | None:
        return self._successor_of.get(memory_id)

    def ancestors(self, memory_id: UUID) -> list[Memory]:
        """
        Walk predecessors via supersedes and parent_memory (oldest first).

        TODO: Graph-based contradiction reasoning over branching lineage.
        """
        chain: list[Memory] = []
        visited: set[UUID] = set()
        current = self._by_id.get(memory_id)
        while current is not None and current.id not in visited:
            visited.add(current.id)
            predecessors: list[Memory] = []
            if current.supersedes is not None:
                pred = self._by_id.get(current.supersedes)
                if pred is not None:
                    predecessors.append(pred)
            if current.parent_memory is not None and current.parent_memory != current.supersedes:
                parent = self._by_id.get(current.parent_memory)
                if parent is not None and parent.id not in visited:
                    predecessors.append(parent)
            if not predecessors:
                break
            predecessors.sort(key=lambda m: m.timestamp)
            current = predecessors[0]
            chain.append(current)
        chain.reverse()
        return chain

    def descendants(self, memory_id: UUID) -> list[Memory]:
        """Follow supersession forward (newest last)."""
        result: list[Memory] = []
        current_id = memory_id
        visited: set[UUID] = set()
        while current_id not in visited:
            visited.add(current_id)
            successor = self._successor_of.get(current_id)
            if successor is None:
                break
            result.append(successor)
            current_id = successor.id
        return result

    def get_memory_lineage(self, memory_id: UUID) -> list[Memory]:
        """Full chain: ancestors → focal memory → descendants."""
        focal = self._by_id.get(memory_id)
        if focal is None:
            return []
        upstream = self.ancestors(memory_id)
        downstream = self.descendants(memory_id)
        return [*upstream, focal, *downstream]

    def get_latest_memory(self, memory_id: UUID) -> Memory | None:
        """
        Follow supersession links to the tip of the chain.

        Prefers the newest non-archived revision; falls back to archived tip.
        """
        current = self._by_id.get(memory_id)
        if current is None:
            return None

        best = current
        while True:
            successor = self._successor_of.get(current.id)
            if successor is None:
                break
            current = successor
            if not current.archived:
                best = current

        return best

    def get_active_memories(self, *, include_archived: bool = False) -> list[Memory]:
        """
        Memories that represent current state: not superseded, optionally not archived.

        Retrieval should prefer this set for latest trusted state.
        """
        active = [
            m
            for m in self.memories
            if not self.is_superseded(m.id) and (include_archived or not m.archived)
        ]
        active.sort(key=lambda m: m.timestamp, reverse=True)
        return active

    def roots(self) -> list[Memory]:
        """Memories that do not supersede another (chain heads)."""
        return [m for m in self.memories if m.supersedes is None]

    def group_by_lineage_root(self) -> dict[UUID, list[Memory]]:
        """Group memories by the root of their supersession chain."""
        groups: dict[UUID, list[Memory]] = {}
        for memory in self.memories:
            root = self._lineage_root_id(memory.id)
            groups.setdefault(root, []).append(memory)
        for chain in groups.values():
            chain.sort(key=lambda m: m.timestamp)
        return groups

    def _lineage_root_id(self, memory_id: UUID) -> UUID:
        current_id = memory_id
        visited: set[UUID] = set()
        while current_id not in visited:
            visited.add(current_id)
            memory = self._by_id.get(current_id)
            if memory is None or memory.supersedes is None:
                return current_id
            current_id = memory.supersedes
        return memory_id


def filter_lineage_active(memories: list[Memory], *, include_archived: bool = False) -> list[Memory]:
    """Lineage-aware retrieval helper: keep only non-superseded memories."""
    return LineageIndex.from_memories(memories).get_active_memories(include_archived=include_archived)


def prefer_latest_per_chain(memories: list[Memory]) -> list[Memory]:
    """
    Collapse each supersession chain to its latest revision for retrieval.

    TODO: Lineage-aware retrieval — boost corroborating branches, penalize orphans.
    """
    index = LineageIndex.from_memories(memories)
    tips: list[Memory] = []
    seen: set[UUID] = set()
    for memory in memories:
        latest = index.get_latest_memory(memory.id)
        if latest is not None and latest.id not in seen:
            tips.append(latest)
            seen.add(latest.id)
    tips.sort(key=lambda m: m.timestamp, reverse=True)
    return tips
