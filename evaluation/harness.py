"""Benchmark harness: deterministic execution, structured manifests."""

from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Union

from evaluation.metrics import CollaborativeMetrics
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reports & manifests
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """
    Structured output of a single benchmark scenario.

    - ``metrics``  : numeric outcomes keyed by metric name.
    - ``expected`` : minimum thresholds; ``passed`` is True when every metric
                     meets/exceeds its threshold.
    - ``queries``  : per-query traces (query, retrieved ids, hit details).
    - ``notes``    : free-form scenario metadata (e.g. expected lineage chain).
    """

    scenario: str
    metrics: dict[str, float] = field(default_factory=dict)
    expected: dict[str, float | tuple[str, float] | list] = field(default_factory=dict)
    queries: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        for name, spec in self.expected.items():
            value = self.metrics.get(name, 0.0)
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                op, target = spec
                target = float(target)
                if op == ">=" and not value >= target:
                    return False
                if op == "<=" and not value <= target:
                    return False
                if op == ">" and not value > target:
                    return False
                if op == "<" and not value < target:
                    return False
                if op == "==" and value != target:
                    return False
            else:
                if value < float(spec):
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "metrics": self.metrics,
            "expected": self.expected,
            "passed": self.passed,
            "queries": self.queries,
            "notes": self.notes,
        }


@dataclass
class RunManifest:
    """Reproducibility envelope for a single scenario invocation."""

    scenario: str
    seed: int
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, float] = field(default_factory=dict)
    expected: dict[str, float] = field(default_factory=dict)
    queries: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    passed: bool = True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved run manifest to %s", path)


# ---------------------------------------------------------------------------
# Scenario protocol & harness
# ---------------------------------------------------------------------------


ScenarioResult = Union[BenchmarkReport, CollaborativeMetrics]


class BenchmarkScenario(Protocol):
    """Protocol every scenario must satisfy."""

    name: str
    expected: dict[str, float]

    def setup(self) -> None: ...

    def run(self) -> ScenarioResult: ...

    def teardown(self) -> None: ...


def _detect_git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


class EvaluationHarness:
    """
    Run registered scenarios deterministically and persist manifests.

    Scenarios may return either:
      - ``BenchmarkReport`` (preferred, structured), or
      - ``CollaborativeMetrics`` (legacy aggregate).

    The harness adapts both into a uniform ``RunManifest`` written under
    ``runs_dir`` (default ``experiments/runs/``).
    """

    def __init__(self, runs_dir: Path | str | None = None) -> None:
        self._runs_dir = Path(runs_dir) if runs_dir else Path("experiments/runs")
        self._scenarios: dict[str, BenchmarkScenario] = {}

    def register(self, scenario: BenchmarkScenario) -> None:
        self._scenarios[scenario.name] = scenario

    def list_scenarios(self) -> list[str]:
        return sorted(self._scenarios)

    def get(self, name: str) -> BenchmarkScenario:
        scenario = self._scenarios.get(name)
        if scenario is None:
            raise KeyError(f"Unknown scenario: {name}")
        return scenario

    def run(
        self,
        name: str,
        *,
        seed: int = 42,
        config: dict[str, Any] | None = None,
    ) -> RunManifest:
        scenario = self.get(name)
        _seed_everything(seed)
        logger.info("Running benchmark scenario %s (seed=%d)", name, seed)

        manifest = RunManifest(
            scenario=name,
            seed=seed,
            config_snapshot=config or {},
            git_sha=_detect_git_sha(),
        )

        try:
            scenario.setup()
            result = scenario.run()
        finally:
            scenario.teardown()

        if isinstance(result, BenchmarkReport):
            manifest.metrics = dict(result.metrics)
            manifest.expected = dict(result.expected)
            manifest.queries = list(result.queries)
            manifest.notes = dict(result.notes)
            manifest.passed = result.passed
        elif isinstance(result, CollaborativeMetrics):
            manifest.metrics = {
                "recall_at_k": result.recall_at_k,
                "cross_agent_consistency": result.cross_agent_consistency,
                "contradiction_resolution_rate": result.contradiction_resolution_rate,
                "lineage_completeness": result.lineage_completeness,
                "trust_calibration_error": result.trust_calibration_error,
            }
            manifest.passed = True
        else:
            raise TypeError(
                f"Scenario {name!r} returned unsupported type {type(result).__name__}"
            )

        out = self._runs_dir / f"{name}_seed{seed}.json"
        manifest.save(out)
        return manifest

    def run_all(self, *, seed: int = 42) -> dict[str, RunManifest]:
        return {name: self.run(name, seed=seed) for name in self.list_scenarios()}
