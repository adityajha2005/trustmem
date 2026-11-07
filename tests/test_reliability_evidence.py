"""Adaptive trust and evidence rules: deterministic unit tests for every frozen rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory.evidence import Evidence, SourceKind, VerificationEvent
from memory.reliability import BetaReliability, EWMAReliability, judge, learn
from memory.schema import Memory
from retrieval.context import RetrievalContext
from retrieval.scoring import trust_score
from storage import init_db, session_scope
from storage.sqlite_store import SQLiteMemoryStore
from utils.config import AppConfig

T0 = datetime(2026, 1, 1, tzinfo=UTC)
PRIOR = {"planner": 0.60, "coder": 0.70, "verifier": 0.95}


def note(agent: str, value: str, minutes: int = 0, key: str = "api.rate_limit", valid_for: int | None = None) -> Memory:
    return Memory(
        content=f"{agent}: {value}", agent=agent, claim_key=key, claim_value=value,
        timestamp=T0 + timedelta(minutes=minutes),
        valid_until=None if valid_for is None else T0 + timedelta(minutes=minutes + valid_for),
    )


def event(value: str, minutes: int = 100, kind: SourceKind = SourceKind.TEST, key: str = "api.rate_limit") -> VerificationEvent:
    return VerificationEvent(claim_key=key, value=value, source_kind=kind, source_id=kind.value, at=T0 + timedelta(minutes=minutes))


# --- reliability ------------------------------------------------------------


def test_cold_start_is_the_role_prior() -> None:
    rel = BetaReliability(PRIOR)
    assert rel.trust("verifier", "api") == pytest.approx(0.95)
    assert rel.trust("stranger") == 0.5


def test_one_mistake_is_bounded_by_prior_strength() -> None:
    rel = BetaReliability(PRIOR)
    rel.observe("verifier", None, correct=False)
    assert rel.trust("verifier") == pytest.approx(3.8 / 5.0)  # 0.95 -> 0.76, not to 0


def test_decay_tracks_a_reliable_agent_turning_unreliable() -> None:
    fast, slow = BetaReliability(PRIOR, retention=0.9), BetaReliability(PRIOR, retention=1.0)
    for rel in (fast, slow):
        for _ in range(20):
            rel.observe("verifier", None, True)
        for _ in range(3):
            rel.observe("verifier", None, False)
    assert fast.trust("verifier") < slow.trust("verifier")


def test_topic_trust_backs_off_to_agent_trust_when_sparse() -> None:
    rel = BetaReliability(PRIOR, topic_threshold=3)
    rel.observe("coder", "db", True)
    rel.observe("coder", "infra", False)
    t = rel._topic[("coder", "infra")]
    w = (t.a + t.b - 4) / 3
    assert w == pytest.approx(1 / 3)
    assert rel.trust("coder", "infra") == pytest.approx(w * t.mean + (1 - w) * rel.trust("coder"))
    assert rel.trust("coder", "db") > rel.trust("coder", "infra")
    assert BetaReliability(PRIOR, per_topic=False).trust("coder", "db") == pytest.approx(0.7)


def test_ewma_starts_at_prior_and_moves_by_rate() -> None:
    rel = EWMAReliability(PRIOR, rate=0.3)
    rel.observe("verifier", "api", False)
    assert rel.trust("verifier", "api") == pytest.approx(0.665)
    assert rel.trust("verifier", "db") == pytest.approx(0.95)


# --- what gets judged (feedback-loop guards) ----------------------------------


def test_agent_sourced_events_never_change_reliability() -> None:
    rel = BetaReliability(PRIOR)
    learn(rel, event("300", kind=SourceKind.AGENT), [note("verifier", "1000")])
    assert judge(event("300", kind=SourceKind.AGENT), [note("verifier", "1000")]) == []
    assert rel.trust("verifier") == pytest.approx(0.95)


def test_repeating_a_claim_counts_once() -> None:
    notes = [note("coder", "1000", minutes=10 * i) for i in range(5)]
    assert judge(event("300"), notes) == [("coder", False)]


def test_self_correction_is_judged_on_the_latest_claim() -> None:
    notes = [note("verifier", "1000", 0), note("verifier", "600", 10), note("verifier", "300", 20)]
    assert judge(event("300"), notes) == [("verifier", True)]


def test_expired_claim_and_later_claim_are_not_judged() -> None:
    assert judge(event("300", minutes=100), [note("verifier", "1000", 0, valid_for=30)]) == []
    assert judge(event("300", minutes=100), [note("coder", "1000", 200)]) == []


# --- evidence -----------------------------------------------------------------


def test_latest_round_decides_and_disagreement_is_contested() -> None:
    ev = Evidence([event("1000", 10), event("300", 50)])
    assert ev.verdict("api.rate_limit").value == "300"
    ev.add(event("600", 50, kind=SourceKind.CONFIG))
    v = ev.verdict("api.rate_limit")
    assert v.contested and v.value is None
    assert Evidence([event("300", kind=SourceKind.AGENT)]).verdict("api.rate_limit").value is None


def test_filter_drops_refuted_expired_and_superseded_versions() -> None:
    ev = Evidence([event("300", minutes=50)])
    notes = [
        note("planner", "1000", 0),  # refuted by the event at minute 50
        note("verifier", "600", 10), note("verifier", "300", 20),  # older version dropped
        note("coder", "1000", 60),  # written after the event: unverified, kept
        note("planner", "x", 0, key="db.engine", valid_for=30),  # expired at query time
        Memory(content="no claim", agent="planner"),
    ]
    kept, dropped = ev.filter(notes, now=T0 + timedelta(minutes=90))
    assert [(m.agent, m.claim_value) for m in kept] == [("verifier", "300"), ("coder", "1000"), ("planner", None)]
    assert dropped == 3


def test_supported_note_gets_full_trust_others_keep_theirs() -> None:
    ev = Evidence([event("300", minutes=50)])
    ctx = RetrievalContext(agent_id="reader", evidence=ev, trust=lambda m: 0.42)
    assert trust_score(note("planner", "300"), ctx) == 1.0
    assert trust_score(note("planner", "999", key="db.engine"), ctx) == 0.42


# --- storage ------------------------------------------------------------------


def test_claim_fields_and_events_persist() -> None:
    config = AppConfig(database_url="sqlite:///:memory:")
    init_db(config)
    with session_scope(config) as session:
        store = SQLiteMemoryStore(session)
        m = store.create(note("verifier", "300", valid_for=30))
        back = store.get(m.id)
        assert (back.claim_key, back.claim_value, back.valid_until) == ("api.rate_limit", "300", m.valid_until)
        store.add_event(event("300"))
        assert store.list_events("api.rate_limit") == [event("300")]
