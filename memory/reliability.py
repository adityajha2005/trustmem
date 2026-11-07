"""Agent reliability learned from external verification outcomes.

BetaReliability: Beta-Bernoulli per agent and per (agent, topic). The role trust is the prior
(alpha0 = n0 * r, beta0 = n0 * (1 - r)); each external verification first discounts the
accumulated evidence toward that prior by `retention`, then adds one outcome. Ranking uses a
blend of topic and agent means, weighted by how much topic evidence exists.

EWMAReliability: the simpler baseline, kept per (agent, topic), starting at the role prior.

Only `learn()` changes state, and it ignores agent-sourced events. Retrieval never calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory.evidence import VerificationEvent, expired, topic_of
from memory.schema import Memory

DEFAULT_PRIOR = 0.5  # agents with no role prior


@dataclass
class Beta:
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)


class BetaReliability:
    def __init__(
        self,
        prior: dict[str, float],
        *,
        prior_strength: float = 4.0,
        retention: float = 0.9,
        topic_threshold: float = 3.0,
        per_topic: bool = True,
    ) -> None:
        self.prior = prior
        self.n0 = prior_strength
        self.retention = retention
        self.topic_threshold = topic_threshold
        self.per_topic = per_topic
        self._agent: dict[str, Beta] = {}
        self._topic: dict[tuple[str, str], Beta] = {}

    def _start(self, agent: str) -> Beta:
        r = self.prior.get(agent, DEFAULT_PRIOR)
        return Beta(self.n0 * r, self.n0 * (1 - r))

    def _update(self, beta: Beta, start: Beta, correct: bool) -> None:
        lam = self.retention
        beta.a = lam * beta.a + (1 - lam) * start.a + (1.0 if correct else 0.0)
        beta.b = lam * beta.b + (1 - lam) * start.b + (0.0 if correct else 1.0)

    def observe(self, agent: str, topic: str | None, correct: bool) -> None:
        start = self._start(agent)
        self._update(self._agent.setdefault(agent, self._start(agent)), start, correct)
        if self.per_topic and topic:
            self._update(self._topic.setdefault((agent, topic), self._start(agent)), start, correct)

    def trust(self, agent: str, topic: str | None = None) -> float:
        overall = self._agent.get(agent, self._start(agent)).mean
        if not (self.per_topic and topic and (agent, topic) in self._topic):
            return overall
        t = self._topic[(agent, topic)]
        w = min(1.0, (t.a + t.b - self.n0) / self.topic_threshold)
        return w * t.mean + (1 - w) * overall


class EWMAReliability:
    def __init__(self, prior: dict[str, float], *, rate: float = 0.3) -> None:
        self.prior = prior
        self.rate = rate
        self._t: dict[tuple[str, str | None], float] = {}

    def observe(self, agent: str, topic: str | None, correct: bool) -> None:
        cur = self.trust(agent, topic)
        self._t[(agent, topic)] = (1 - self.rate) * cur + self.rate * (1.0 if correct else 0.0)

    def trust(self, agent: str, topic: str | None = None) -> float:
        return self._t.get((agent, topic), self.prior.get(agent, DEFAULT_PRIOR))


def judge(event: VerificationEvent, memories: list[Memory]) -> list[tuple[str, bool]]:
    """One outcome per agent: its latest unexpired claim on the key, written before the event."""
    if not event.is_external:
        return []
    latest: dict[str, Memory] = {}
    for m in memories:
        if m.claim_key == event.claim_key and m.timestamp <= event.at and not expired(m, event.at):
            if m.agent not in latest or m.timestamp > latest[m.agent].timestamp:
                latest[m.agent] = m
    return [(agent, m.claim_value == event.value) for agent, m in sorted(latest.items())]


def learn(reliability: BetaReliability | EWMAReliability, event: VerificationEvent, memories: list[Memory]) -> None:
    for agent, correct in judge(event, memories):
        reliability.observe(agent, topic_of(event.claim_key), correct)
