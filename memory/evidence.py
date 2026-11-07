"""Evidence attached to claims: verification events, verdicts, and the stage-4 evidence filter.

Rules (frozen with Benchmark C, see docs/BENCHMARK_C.md):
- Only external sources (test, config, human) decide anything. Agent-sourced checks are stored
  but never support, refute, or move reliability.
- For a claim_key, the latest round of external events (newest timestamp) gives the verdict.
  If that round disagrees with itself, the claim is contested and nothing is decided.
- A verdict refutes only notes written at or before it; a newer claim is unverified, not wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, field_validator

from memory.schema import Memory


class SourceKind(StrEnum):
    TEST = "test"
    CONFIG = "config"
    HUMAN = "human"
    AGENT = "agent"


EXTERNAL = frozenset({SourceKind.TEST, SourceKind.CONFIG, SourceKind.HUMAN})


class VerificationEvent(BaseModel):
    claim_key: str
    value: str
    source_kind: SourceKind
    source_id: str
    at: datetime

    @field_validator("at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)

    @property
    def is_external(self) -> bool:
        return self.source_kind in EXTERNAL


@dataclass(frozen=True)
class Verdict:
    value: str | None = None  # the verified value, if the latest round agrees
    at: datetime | None = None
    contested: bool = False  # the latest round of external sources disagrees


def topic_of(claim_key: str | None) -> str | None:
    return claim_key.split(".", 1)[0] if claim_key else None


def expired(memory: Memory, at: datetime) -> bool:
    return memory.valid_until is not None and memory.valid_until <= at


class Evidence:
    """In-memory view over verification events (load them from SQLiteMemoryStore.list_events)."""

    def __init__(self, events: list[VerificationEvent] | None = None) -> None:
        self.events: list[VerificationEvent] = list(events or [])

    def add(self, event: VerificationEvent) -> None:
        self.events.append(event)

    def verdict(self, claim_key: str, now: datetime | None = None) -> Verdict:
        ext = [e for e in self.events if e.claim_key == claim_key and e.is_external and (now is None or e.at <= now)]
        if not ext:
            return Verdict()
        latest = max(e.at for e in ext)
        values = {e.value for e in ext if e.at == latest}
        if len(values) > 1:
            return Verdict(at=latest, contested=True)
        return Verdict(value=values.pop(), at=latest)

    def supports(self, memory: Memory, now: datetime | None = None) -> bool:
        if not memory.claim_key:
            return False
        return self.verdict(memory.claim_key, now).value == memory.claim_value

    def filter(self, memories: list[Memory], now: datetime) -> tuple[list[Memory], int]:
        """Keep each agent's latest note per claim, then drop expired and refuted notes."""
        latest: dict[tuple[str, str], Memory] = {}
        for m in memories:
            if m.claim_key:
                key = (m.agent, m.claim_key)
                if key not in latest or m.timestamp > latest[key].timestamp:
                    latest[key] = m
        kept = []
        for m in memories:
            if m.claim_key:
                if latest[(m.agent, m.claim_key)] is not m or expired(m, now):
                    continue
                v = self.verdict(m.claim_key, now)
                if v.value is not None and v.value != m.claim_value and m.timestamp <= v.at:
                    continue
            kept.append(m)
        return kept, len(memories) - len(kept)
