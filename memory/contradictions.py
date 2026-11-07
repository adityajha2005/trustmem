"""Contradiction registry and conflict resolution for evolving memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from memory.lineage import LineageIndex
from memory.schema import Memory
from memory.temporal import prefer_latest_trusted
from utils.logging import get_logger

logger = get_logger(__name__)


class ResolutionStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    DEFERRED = "deferred"


@dataclass
class ContradictionClaim:
    """Conflicting memories that cannot all hold under current beliefs."""

    id: UUID = field(default_factory=uuid4)
    memory_ids: list[UUID] = field(default_factory=list)
    detected_by_agent_id: str | None = None
    confidence: float = 0.5
    resolution_status: ResolutionStatus = ResolutionStatus.OPEN
    winner_id: UUID | None = None
    notes: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_open(self) -> bool:
        return self.resolution_status == ResolutionStatus.OPEN


class ContradictionRegistry:
    """
    Track and resolve contradictions across collaborative agents.

    Archive-aware: archived memories are excluded from winner selection.
    """

    def __init__(self) -> None:
        self._claims: dict[UUID, ContradictionClaim] = {}

    @property
    def claims(self) -> list[ContradictionClaim]:
        return list(self._claims.values())

    @property
    def open_claims(self) -> list[ContradictionClaim]:
        return [c for c in self._claims.values() if c.is_open]

    def register(
        self,
        memory_ids: list[UUID],
        *,
        detected_by: str | None = None,
        confidence: float = 0.5,
        notes: str = "",
    ) -> ContradictionClaim:
        """
        Record a contradiction between two or more memories.

        TODO: Probabilistic contradiction resolution — confidence-weighted voting.
        TODO: Graph-based contradiction reasoning — infer implicit conflicts.
        """
        if len(memory_ids) < 2:
            raise ValueError("Contradiction requires at least two memory ids")
        claim = ContradictionClaim(
            memory_ids=list(memory_ids),
            detected_by_agent_id=detected_by,
            confidence=confidence,
            notes=notes,
        )
        self._claims[claim.id] = claim
        logger.info("Registered contradiction %s (%d memories)", claim.id, len(memory_ids))
        return claim

    def get(self, claim_id: UUID) -> ContradictionClaim | None:
        return self._claims.get(claim_id)

    def resolve_conflicts(
        self,
        claim_id: UUID,
        lineage: LineageIndex,
        *,
        winner_id: UUID | None = None,
        requesting_agent: str | None = None,
    ) -> ContradictionClaim:
        """
        Resolve an open contradiction by selecting a winning memory.

        Default policy: latest trusted non-archived memory in the claim set.
        Verifier overrides can pass winner_id explicitly.
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            raise KeyError(f"Unknown contradiction claim: {claim_id}")
        if not claim.is_open:
            return claim

        candidates: list[Memory] = []
        for mem_id in claim.memory_ids:
            memory = lineage.get(mem_id)
            if memory is None or memory.archived:
                continue
            latest = lineage.get_latest_memory(mem_id)
            if latest is not None and not latest.archived:
                candidates.append(latest)

        if winner_id is not None:
            winner = lineage.get(winner_id)
            if winner is None:
                raise ValueError(f"Winner memory not found: {winner_id}")
        else:
            winner = prefer_latest_trusted(candidates, requesting_agent=requesting_agent)
            if winner is None:
                raise ValueError(f"No resolvable candidates for claim {claim_id}")

        claim.resolution_status = ResolutionStatus.RESOLVED
        claim.winner_id = winner.id
        logger.info("Resolved contradiction %s -> winner %s", claim_id, winner.id)
        return claim

    def contested_memory_ids(self, *, include_resolved: bool = False) -> set[UUID]:
        """All memory ids participating in open (or all) claims."""
        ids: set[UUID] = set()
        for claim in self._claims.values():
            if claim.is_open or include_resolved:
                ids.update(claim.memory_ids)
        return ids

    def filter_for_retrieval(
        self,
        memories: list[Memory],
        lineage: LineageIndex,
        *,
        include_archived: bool = False,
        prefer_winners: bool = True,
    ) -> tuple[list[Memory], int]:
        """
        Archive-aware contradiction filtering for retrieval.

        - Drops archived unless include_archived.
        - Drops superseded (stale) revisions via lineage.
        - For open claims: keep latest trusted winner per conflict when prefer_winners=True.
        """
        start_count = len(memories)
        pool = [m for m in memories if include_archived or not m.archived]

        active_ids = {m.id for m in lineage.get_active_memories(include_archived=include_archived)}
        pool = [m for m in pool if m.id in active_ids]

        drop_ids: set[UUID] = set()
        if self.open_claims:
            for claim in self.open_claims:
                claim_memories = [
                    m
                    for mid in claim.memory_ids
                    if (m := lineage.get(mid)) is not None and (include_archived or not m.archived)
                ]
                if not claim_memories:
                    continue
                if prefer_winners:
                    winner = prefer_latest_trusted(
                        claim_memories,
                        requesting_agent=claim.detected_by_agent_id,
                    )
                    if winner is not None:
                        drop_ids.update(m.id for m in claim_memories if m.id != winner.id)
                else:
                    drop_ids.update(m.id for m in claim_memories)

        filtered = [m for m in pool if m.id not in drop_ids]
        return filtered, start_count - len(filtered)

    def mark_contested(self, records: list[Memory], claim: ContradictionClaim) -> list[Memory]:
        """Lower confidence on open-claim participants until resolved."""
        if not claim.is_open:
            return records
        contested = set(claim.memory_ids)
        return [
            r.model_copy(update={"confidence": min(r.confidence, 0.25)})
            if r.id in contested
            else r
            for r in records
        ]
