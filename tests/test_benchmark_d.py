"""Benchmark D: offline tests with mock agents. No test makes a live API call."""

from __future__ import annotations

import json
import pytest

from experiments import d_agents, d_env, d_run

CFG = d_run.load_config()


def _tasks(n: int = 10) -> list[dict]:
    return d_env.generate_tasks("pilot", "test-seed", n)


# --- controlled environment and ground truth -------------------------------------------------


def test_frozen_task_files_match_their_generator() -> None:
    for split, s in CFG["splits"].items():
        on_disk = d_env.load_tasks(split)
        regen = d_env.generate_tasks(split, s["seed"], s["tasks"])
        assert [json.dumps(t, sort_keys=True) for t in regen] == [json.dumps(t, sort_keys=True) for t in on_disk]
        assert len(on_disk) == s["tasks"] and all(t["origin"] == "controlled_env" for t in on_disk)


def test_pilot_and_full_tasks_do_not_overlap() -> None:
    sig = lambda t: (t["scenario"], t["claim_key"], json.dumps(t["timeline"]), t["project"])  # noqa: E731
    assert not {sig(t) for t in d_env.load_tasks("pilot")} & {sig(t) for t in d_env.load_tasks("full")}


@pytest.mark.parametrize("key", list(d_env.CLAIMS))
def test_check_reads_back_every_value_of_every_key(key: str) -> None:
    topic = key.split(".", 1)[0]
    background = {k: d_env.CLAIMS[k]["values"][0] for k in d_env.keys_for(topic)}
    for v in d_env.CLAIMS[key]["values"]:
        text = d_env.render(topic, {**background, key: v}, "2026-01-01T00:00Z")
        assert d_env.check(topic, text, key) == d_env.normalize(key, v) is not None


def test_normalize_is_strict_about_ambiguity() -> None:
    assert d_env.normalize("api.rate_limit_per_min", "1,000 req/min") == "1000"
    assert d_env.normalize("api.rate_limit_per_min", "300 (was 1000)") is None
    assert d_env.normalize("deps.requests_version", "requests 2.31.0") == d_env.normalize("deps.requests_version", "2.31")
    assert d_env.normalize("config.cache_backend", "In memory") == "inmemory"
    assert d_env.normalize("config.cache_backend", "redis, not memcached") is None
    assert d_env.normalize("impl.date_format", "ISO 8601 strings") == "iso8601"


def test_views_match_agent_steps_and_never_name_the_truth() -> None:
    for t in d_env.load_tasks("full"):
        agent_steps = [i for i, s in enumerate(t["steps"]) if s not in ("CHANGE", "CHECK")]
        assert len(agent_steps) == len(t["views"])
        for i, view in zip(agent_steps, t["views"], strict=True):
            doc = d_env.view_document(t, view, d_env.step_minute(i))
            if doc:
                assert "current" not in doc["title"].lower() and "stale" not in doc["title"].lower()


def test_ground_truth_follows_the_timeline() -> None:
    t = next(t for t in d_env.load_tasks("full") if t["scenario"] == "d08_expiring_claims")
    override = t["timeline"][1]["value"]
    assert d_env.truth_at(t, 50) == d_env.normalize(t["claim_key"], override)
    assert d_env.truth_at(t, d_env.QUERY_MIN) == t["truth"] != d_env.truth_at(t, 50)
    assert t["stale_values"] == [d_env.normalize(t["claim_key"], override)]


# --- agent layer ----------------------------------------------------------------------------


def test_parse_reply_keeps_free_text_and_flags_bad_keys() -> None:
    r = d_agents.parse_reply('ok {"claim_key": "api.rate_limit_per_min", "claim_value": "300/min", "expires_at": null,'
                             ' "note": "Limit dropped to 300.", "other_note": "x"}', "api")
    assert (r["claim_key"], r["claim_value"], r["valid_until"]) == ("api.rate_limit_per_min", "300", None)
    bad = d_agents.parse_reply('{"claim_key": "api.limit", "claim_value": "300", "note": "n"}', "api")
    assert bad["claim_key"] is None and bad["claim_value"] is None
    with pytest.raises(d_agents.ReplyError):
        d_agents.parse_reply("no json here", "api")
    exp = d_agents.parse_reply('{"claim_key": "api.rate_limit_per_min", "claim_value": "2000",'
                               ' "expires_at": "2026-03-02T10:40Z", "note": "temp"}', "api")
    assert exp["valid_until"] == "2026-03-02T10:40:00+00:00"


def test_returned_model_must_match() -> None:
    assert d_agents.returned_model_ok("gpt-6-luna", "gpt-6-luna")
    assert d_agents.returned_model_ok("gpt-6-luna", "gpt-6-luna-2026-09-21")
    assert not d_agents.returned_model_ok("gpt-6-luna", "gpt-6-luna-mini")
    assert not d_agents.returned_model_ok("grok-4.7", "grok-3-mini")


class FakeAPI:
    """Stands in for the provider SDK: returns canned replies with usage, never touches the network."""

    def __init__(self, provider: str, model_id: str, returned: str | None = None, text: str | None = None) -> None:
        self.provider, self.model_id = provider, model_id
        self.returned, self.text = returned or model_id, text
        self.mock = d_agents.MockClient(model_id)

    def complete(self, messages, params):
        r = self.mock.complete(messages, params)
        return d_agents.Reply(self.text or r.text, self.returned,
                              {"prompt_tokens": 1000, "completion_tokens": 500, "reasoning_tokens": 200, "cached_tokens": 100})


def test_ledger_records_usage_cost_and_stops_on_model_substitution(tmp_path) -> None:
    model = CFG["models"]["gpt-6.1-sol"]
    ledger = d_agents.Ledger.open(tmp_path, tmp_path / "m", cap_usd=10)
    msgs = d_agents.build_messages("planner", _tasks(1)[0], "2026-03-02T09:10Z", None, [])
    _, rec = ledger.call(FakeAPI("openai", "gpt-6.1-sol"), model, msgs, {"task_id": "t"}, 0)
    assert rec["cost_usd"] == pytest.approx((900 * 2.0 + 100 * 0.10 + 500 * 10.0) / 1e6)
    assert rec["model_returned"] == "gpt-6.1-sol" and rec["prompt_sha256"] == d_agents.messages_sha256(msgs)
    with pytest.raises(d_agents.ModelMismatch):
        ledger.call(FakeAPI("openai", "gpt-6.1-sol", returned="gpt-6-luna"), model, msgs, {"task_id": "t"}, 0)
    assert len((tmp_path / "m" / "agent_calls.jsonl").read_text().splitlines()) == 2  # the substituted call is kept


def test_budget_cap_is_enforced_across_sessions(tmp_path) -> None:
    model = CFG["models"]["gpt-6.1-sol"]
    msgs = d_agents.build_messages("planner", _tasks(1)[0], "2026-03-02T09:10Z", None, [])
    ledger = d_agents.Ledger.open(tmp_path, tmp_path / "a", cap_usd=0.03)
    ledger.call(FakeAPI("openai", "gpt-6.1-sol"), model, msgs, {}, 0)
    reopened = d_agents.Ledger.open(tmp_path, tmp_path / "b", cap_usd=0.03)  # another model, same run
    assert reopened.spent == pytest.approx(ledger.spent)
    with pytest.raises(d_agents.BudgetExceeded):
        for _ in range(10):
            reopened.call(FakeAPI("openai", "gpt-6.1-sol"), model, msgs, {}, 0)


def test_live_generation_refuses_unverified_prices_and_mock_run_ids(tmp_path) -> None:
    with pytest.raises(SystemExit, match="price not verified"):
        d_run.generate("pilot", "gpt-6-luna", "pilot-test", cfg=CFG, runs_dir=tmp_path, tasks=_tasks(1))
    with pytest.raises(SystemExit, match="mock"):
        d_run.generate("pilot", "gpt-6-luna", "pilot-test", mock=True, cfg=CFG, runs_dir=tmp_path, tasks=_tasks(1))


def test_openai_client_stops_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        d_agents.OpenAICompatClient("openai", "gpt-6-luna")


def test_unparseable_reply_is_retried_then_recorded_not_dropped(tmp_path) -> None:
    tasks = _tasks(2)
    client = d_agents.MockClient("gpt-6-luna", bad_reply_every=1)
    d_run.generate("pilot", "gpt-6-luna", "mock-x", client=client, mock=True, cfg=CFG, runs_dir=tmp_path, tasks=tasks)
    writes = d_run._jsonl(tmp_path / "mock-x" / "gpt-6-luna" / "writes.jsonl")
    assert writes and all(not w["parse_ok"] and len(w["calls"]) == CFG["calls"]["max_parse_attempts"] for w in writes)


# --- generation + replay --------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    tasks = _tasks(10)
    d_run.generate("pilot", "gpt-6-luna", "mock-run", mock=True, cfg=CFG, runs_dir=runs, tasks=tasks)
    d_run.generate("pilot", "grok-4.7", "mock-run", client=d_agents.MockClient("grok-4.7", never_expires=True),
                   mock=True, cfg=CFG, runs_dir=runs, tasks=tasks)
    return runs, tasks


def test_generation_is_resumable_and_records_harness_checks(mock_run) -> None:
    runs, tasks = mock_run
    md = runs / "mock-run" / "gpt-6-luna"
    before = (md / "agent_calls.jsonl").read_text()
    d_run.generate("pilot", "gpt-6-luna", "mock-run", mock=True, cfg=CFG, runs_dir=runs, tasks=tasks)
    assert (md / "agent_calls.jsonl").read_text() == before  # nothing re-run
    checks = d_run._jsonl(md / "checks.jsonl")
    assert {c["origin"] for c in checks} == {"harness_check"}
    assert sum(c["phase"] == "after_query" for c in checks) == len(tasks)
    for c in checks:
        t = next(t for t in tasks if t["task_id"] == c["task_id"])
        assert c["value"] == d_env.truth_at(t, c["minute"])


def test_replay_is_offline_deterministic_and_prefix_stable(mock_run, monkeypatch) -> None:
    """Scores for the first k tasks don't change when later tasks exist: no future leakage."""
    runs, tasks = mock_run
    md = runs / "mock-run" / "gpt-6-luna"
    writes, checks = d_run._jsonl(md / "writes.jsonl"), d_run._jsonl(md / "checks.jsonl")
    for method in ("learned", "learned_evidence"):
        full, snaps = d_run.replay(method, tasks, writes, checks, CFG)
        again, _ = d_run.replay(method, tasks, writes, checks, CFG)
        prefix, _ = d_run.replay(method, tasks[:5], writes, checks, CFG)
        strip = lambda rs: [{k: v for k, v in r.items() if k != "latency_ms"} for r in rs]  # noqa: E731
        assert strip(full) == strip(again)
        assert strip(full[:5]) == strip(prefix)
        assert len(snaps) == len(tasks)


def test_trust_learns_only_from_post_query_harness_checks(mock_run) -> None:
    runs, tasks = mock_run
    md = runs / "mock-run" / "gpt-6-luna"
    writes, checks = d_run._jsonl(md / "writes.jsonl"), d_run._jsonl(md / "checks.jsonl")
    rows, _ = d_run.replay("learned", tasks, writes, checks, CFG)
    assert rows[0]["trust_at_query"] == CFG["role_prior"]  # cold start is the prior
    no_checks = [c for c in checks if c["phase"] != "after_query"] + [
        {**c, "source_kind": "agent"} for c in checks if c["phase"] == "after_query"]
    flat, _ = d_run.replay("learned", tasks, writes, no_checks, CFG)
    assert all(r["trust_at_query"] == CFG["role_prior"] for r in flat)  # agent-sourced events never move trust
    static, _ = d_run.replay("static", tasks, writes, checks, CFG)
    assert all(r["trust_at_query"] == CFG["role_prior"] for r in static)


def test_evaluate_writes_all_artifacts_without_secrets(mock_run, monkeypatch) -> None:
    runs, tasks = mock_run
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    summary = d_run.evaluate("mock-run", runs_dir=runs, cfg=CFG, tasks=tasks)
    out = runs / "mock-run" / "eval-v1"
    for f in ("summary.json", "manifest.json", "report.md"):
        assert (out / f).exists()
    for m in ("gpt-6-luna", "grok-4.7"):
        rows = d_run._jsonl(out / m / "scores.jsonl")
        assert {r["method"] for r in rows} == set(CFG["methods"]) and len(rows) == 4 * len(tasks)
    # each model has its own memory database and trust state: evaluating it next to another
    # model gives exactly the scores it gets alone
    luna = runs / "mock-run" / "gpt-6-luna"
    alone, _ = d_run.replay("learned", tasks, d_run._jsonl(luna / "writes.jsonl"), d_run._jsonl(luna / "checks.jsonl"), CFG)
    together = [r for r in d_run._jsonl(out / "gpt-6-luna" / "scores.jsonl") if r["method"] == "learned"]
    drop = lambda rs: [{k: v for k, v in r.items() if k not in ("latency_ms", "model_name")} for r in rs]  # noqa: E731
    assert drop(alone) == drop(together)
    g = summary["retrieval"]["groups"]
    assert g["ALL|dense"]["correct@1"]["n"] == 2 * len(tasks)
    sums = (runs / "mock-run" / "SHA256SUMS").read_text()
    assert "agent_calls.jsonl" in sums and "scores.jsonl" in sums
    with pytest.raises(SystemExit, match="exists"):
        d_run.evaluate("mock-run", runs_dir=runs, cfg=CFG, tasks=tasks)  # old artifacts are never overwritten
    (runs / "mock-run" / "gpt-6-luna" / "leak.txt").write_text("key=sk-test-not-a-real-key-000000000000")
    with pytest.raises(SystemExit, match="credential"):
        d_run.evaluate("mock-run", eval_name="eval-leak", runs_dir=runs, cfg=CFG, tasks=tasks)


def test_wilson_and_bootstrap() -> None:
    assert d_run.wilson(0, 0) is None
    lo, hi = d_run.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    rows = [{"task_id": f"t{i}", "model_name": "m", "method": meth, "correct@1": meth == "a" or i % 2 == 0}
            for i in range(20) for meth in ("a", "b")]
    c = d_run.paired_bootstrap(rows, "a", "b", 500, 1)
    assert c["diff"] == 0.5 and c["ci95"][0] > 0 and c == d_run.paired_bootstrap(rows, "a", "b", 500, 1)

