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
    assert rec["cost_usd"] == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)  # cached billed at full input price
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
    unverified = {**CFG, "models": {**CFG["models"], "gpt-6-luna": {**CFG["models"]["gpt-6-luna"], "price_verified": False}}}
    with pytest.raises(SystemExit, match="price not verified"):
        d_run.generate("pilot", "gpt-6-luna", "pilot-test", cfg=unverified, runs_dir=tmp_path, tasks=_tasks(1))
    with pytest.raises(SystemExit, match="mock"):
        d_run.generate("pilot", "gpt-6-luna", "pilot-test", mock=True, cfg=CFG, runs_dir=tmp_path, tasks=_tasks(1))


def test_only_the_provider_key_is_read_from_the_credentials_file(tmp_path, monkeypatch) -> None:
    import os

    creds = tmp_path / "creds.env"
    creds.write_text("XAI_API_KEY=x-file\nOPENAI_API_KEY=o-file\nMEMORY_SEMANTIC_WEIGHT=0.99\nXAI_BASE_URL=https://us.api.x.ai/v1\n")
    monkeypatch.delenv("MEMORY_SEMANTIC_WEIGHT", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert d_agents.load_api_key("xai", creds) == "x-file"
    assert d_agents.load_api_key("openai", creds) == "o-file"
    assert "MEMORY_SEMANTIC_WEIGHT" not in os.environ and "XAI_API_KEY" not in os.environ
    with pytest.raises(SystemExit, match="not found"):
        d_agents.load_api_key("xai", tmp_path / "missing.env")
    xai = d_agents.OpenAICompatClient("xai", "grok-4.7", d_agents.load_api_key("xai", creds))
    oai = d_agents.OpenAICompatClient("openai", "gpt-6-luna", d_agents.load_api_key("openai", creds))
    # each provider gets its own key; the endpoint is fixed (a base-URL setting in the file is ignored)
    assert (xai._client.api_key, str(xai._client.base_url).rstrip("/")) == ("x-file", "https://api.x.ai/v1")
    assert (oai._client.api_key, str(oai._client.base_url).rstrip("/")) == ("o-file", "https://api.openai.com/v1")
    with pytest.raises(SystemExit, match="no API key"):
        d_agents.OpenAICompatClient("openai", "gpt-6-luna", "")


def test_regional_surcharge_is_priced() -> None:
    assert d_agents.endpoint_multiplier("https://api.x.ai/v1") == 1.0
    assert d_agents.endpoint_multiplier("https://us.api.x.ai/v1") == 1.1
    m = {**CFG["models"]["grok-4.7"], "price_multiplier": 1.1}
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000, "cached_tokens": 0}
    assert d_agents.cost_usd(usage, m) == pytest.approx(1.1 * (1000 * 2 + 1000 * 6) / 1e6)


def test_replay_retrieval_settings_are_pinned_against_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_SEMANTIC_WEIGHT", "0.99")
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "some/other-model")
    config = d_run.pinned_config("t", tmp_path, CFG)
    assert config.semantic_weight == 0.45 and config.embedding_model == "BAAI/bge-small-en-v1.5"


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



# --- adaptive-trust stress experiment (secondary, exploratory) --------------------------------

SCFG = d_run.load_config("stress")


def _ideal_trust(per_topic: bool) -> list[dict]:
    """The real BetaReliability over the frozen stress stream, with idealised agents (a current view
    is right, a stale or mirror view is wrong). Trust at each query, as replay computes it."""
    from memory.reliability import BetaReliability

    r = BetaReliability(SCFG["role_prior"], prior_strength=SCFG["prior_strength"], retention=SCFG["retention"],
                        topic_threshold=SCFG["topic_threshold"], per_topic=per_topic)
    out, wrong = [], 0
    for t in d_env.load_tasks("stress"):
        out.append({"t": t, "wrong_before": wrong, "v": r.trust("verifier", t["topic"]),
                    "low": r.trust(t["lower_agent"], t["topic"]), "coder": r.trust("coder", "deps"),
                    "planner": r.trust("planner", "deps"), "v_config": r.trust("verifier", "config")})
        for role, view in zip(t["steps"], t["views"], strict=True):
            r.observe(role, t["topic"], view == "current")
            wrong += role == "verifier" and t["topic"] == "deps" and view != "current"
    return out


def _first(trace: list[dict], pred) -> int | None:
    return next((x["wrong_before"] for x in trace if x["t"]["phase"] == 2 and x["t"]["topic"] == "deps" and pred(x)), None)


def test_frozen_stress_tasks_match_generator_and_design() -> None:
    s = SCFG["stress"]
    on_disk = d_env.load_tasks("stress")
    regen = d_env.generate_stress_tasks(s["seed"], s["phase1"], s["phase2"])
    assert [json.dumps(t, sort_keys=True) for t in regen] == [json.dumps(t, sort_keys=True) for t in on_disk]
    assert len(on_disk) == 25 and all(t["independent"] and t["origin"] == "controlled_env" for t in on_disk)
    p1 = [t for t in on_disk if t["phase"] == 1]
    assert [t["phase"] for t in on_disk] == [1] * len(p1) + [2] * (25 - len(p1))  # phase 1 strictly first
    for t in on_disk:
        assert t["steps"] == [t["lower_agent"], "verifier"] and t["lower_agent"] in ("planner", "coder")
        if t["phase"] == 1:
            assert t["views"][1] == "current"
        elif t["topic"] == "deps":
            assert t["views"] == ["current", "mirror"]
        else:
            assert t["views"] == ["stale", "current"]
    # stress and pilot/full tasks share nothing but the claim bank
    assert not {t["task_id"] for t in on_disk} & {t["task_id"] for t in d_env.load_tasks("pilot")}


def test_mirror_view_shows_the_previous_value_and_its_sync_date() -> None:
    t = next(t for t in d_env.load_tasks("stress") if t["views"] == ["current", "mirror"])
    doc = d_env.view_document(t, "mirror", 20)
    assert d_env.check("deps", doc["text"], t["claim_key"]) == t["timeline"][0]["value"] != t["truth"]
    assert "last synced " + d_env._stamp(t["mirror_frozen_day"], 0) in doc["text"]
    assert "stale" not in doc["title"] and "out of date" not in doc["text"]


def test_reliability_gives_topic_specific_transitions_on_the_frozen_stream() -> None:
    """Expected transitions under idealised agents (frozen in docs/BENCHMARK_D_ADAPTIVE_TRUST.md)."""
    topic = _ideal_trust(per_topic=True)
    assert _first(topic, lambda x: x["v"] < SCFG["role_prior"]["verifier"]) == 1
    assert _first(topic, lambda x: x["v"] < x["coder"]) == 3
    assert _first(topic, lambda x: x["v"] < x["planner"]) == 5
    assert _first(topic, lambda x: x["v"] < x["low"]) == 3
    p2 = [x for x in topic if x["t"]["phase"] == 2]
    assert min(x["v_config"] for x in p2) > 0.95  # the control topic keeps the verifier trusted
    agent_only = _ideal_trust(per_topic=False)
    assert _first(agent_only, lambda x: x["v"] < x["coder"]) == 5
    assert _first(agent_only, lambda x: x["v"] < x["low"]) == 7
    # without per-topic trust, deps failures leak into config: the verifier falls below the coder there
    cfg_p2 = [x for x in agent_only if x["t"]["phase"] == 2 and x["t"]["topic"] == "config"]
    assert min(x["v_config"] for x in cfg_p2) < 0.5
    assert any(x["v"] < x["low"] for x in cfg_p2)


def test_stress_config_is_separate_from_the_primary_benchmark() -> None:
    assert list(CFG["methods"]) == ["dense", "static", "learned", "learned_evidence"]  # pilot-v2 untouched
    assert list(SCFG["methods"]) == [*CFG["methods"], "learned_no_topic"]
    assert SCFG["methods"]["learned_no_topic"]["per_topic"] is False
    assert SCFG["budget_usd"] == {"stress": 5.0}
    for k in ("models", "retrieval_pinned", "role_prior", "prior_strength", "retention", "topic_threshold", "top_k"):
        assert SCFG[k] == CFG[k]
    assert d_run.runs_for("stress") == d_env.STRESS_DIR / "runs" != d_run.RUNS


@pytest.fixture(scope="module")
def stress_run(tmp_path_factory):
    runs = tmp_path_factory.mktemp("stress_runs")
    tasks = d_env.load_tasks("stress")
    d_run.generate("stress", "gpt-6-luna", "mock-stress", mock=True, cfg=SCFG, runs_dir=runs, tasks=tasks)
    return runs, tasks


def test_stress_agents_are_independent_and_capped_at_five_dollars(stress_run) -> None:
    runs, _ = stress_run
    prompts = d_run._jsonl(runs / "mock-stress" / "prompts.jsonl")
    assert prompts and all(p["messages"][1]["content"].endswith("Team notes so far (oldest first):\n(none)") for p in prompts)
    capped = {**SCFG, "budget_usd": {"stress": 1e-9}}
    with pytest.raises(d_agents.BudgetExceeded):  # the stress cap comes from the stress block
        d_run.generate("stress", "grok-4.7", "capped", client=FakeAPI("xai", "grok-4.7"), cfg=capped, runs_dir=runs,
                       tasks=d_env.load_tasks("stress"))


def test_stress_evaluation_separates_trust_changes_from_ranking_changes(stress_run) -> None:
    runs, tasks = stress_run
    summary = d_run.evaluate("mock-stress", runs_dir=runs, cfg=SCFG, split="stress", tasks=tasks)
    a = summary["stress"]["gpt-6-luna"]
    learned, agent_only = a["transitions"]["learned"], a["transitions"]["learned_no_topic"]
    # the mock agent reads its document, so it behaves like the idealised agents above
    assert a["verifier_wrong_deps_checks_total"] == 12
    assert learned["trust_swap_vs_coder"]["incorrect_verifier_checks_before"] == 3
    assert learned["trust_swap_vs_planner"]["incorrect_verifier_checks_before"] == 5
    assert agent_only["trust_swap_vs_task_lower_agent"]["incorrect_verifier_checks_before"] == 7
    assert "ranking_first_top1_role_differs_from_static" in learned  # measured separately from trust
    assert set(a["ranking_changed_vs_static_top1"]) == {"learned", "learned_evidence", "learned_no_topic"}
    assert a["topic_control"]["learned"]["min_verifier_config_trust_phase2"] > 0.95
    rows = d_run._jsonl(runs / "mock-stress" / "eval-v1" / "gpt-6-luna" / "scores.jsonl")
    assert {r["method"] for r in rows} == set(SCFG["methods"]) and len(rows) == 5 * len(tasks)
    assert "Adaptive-trust stress analysis" in (runs / "mock-stress" / "eval-v1" / "report.md").read_text()
    assert (runs / "mock-stress" / "SHA256SUMS").exists()
