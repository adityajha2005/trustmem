"""Benchmark D, layer B: the REAL LLM agents (planner, coder, verifier) and their call ledger.

Every call is recorded with provider, requested and returned model id, role, prompt hash,
sampling params, run and task ids, timestamp, latency, token usage and cost. A returned model
id that doesn't match the requested one stops the run (no silent substitution). Spending is
checked against a hard cap before every call. API keys are read from the environment by the
OpenAI SDK client and never written anywhere.

`MockClient` reads the prompt the same way a model would (it sees only the prompt), so the
whole pipeline can be tested offline without any API call.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from experiments.d_env import CLAIMS, keys_for, normalize

PROVIDERS = {  # same endpoints and key variables as experiments/real_run.py
    "xai": ("XAI_API_KEY", "XAI_BASE_URL", "https://api.x.ai/v1"),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1"),
}

ROLE_DESC = {
    "planner": "You research the project and propose facts for the team to build on.",
    "coder": "You implement changes and record the facts you built against.",
    "verifier": "You check claims against tests, repository state, configuration and documentation.",
}

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are the {role} agent on a small software team. {role_desc}
    You know only what is in the documents and team notes shown to you. They may be out of
    date or wrong; use your own judgement.
    Write ONE note to the team's shared memory about the fact you are asked about. Other
    agents will search this memory later. Reply with JSON only:
    {{"claim_key": "<one key from the list>",
      "claim_value": "<the value you believe is current, value only>",
      "expires_at": "<ISO-8601 time if the value is temporary, otherwise null>",
      "note": "<your note, one or two sentences, in your own words>",
      "other_note": "<one short note about some other aspect of the project>"}}"""
)

USER_PROMPT = textwrap.dedent(
    """\
    Current time: {now}
    Project: {project} service
    Fact to record: {desc}
    Claim keys you may use:
    {keys}

    Documents you can see:
    {documents}

    Team notes so far (oldest first):
    {notes}"""
)


def prompt_templates_sha256() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + USER_PROMPT + json.dumps(ROLE_DESC, sort_keys=True)).encode()).hexdigest()


def build_messages(role: str, task: dict[str, Any], now: str, document: dict[str, str] | None,
                   notes: list[dict[str, Any]]) -> list[dict[str, str]]:
    keys = "\n".join(f"- {k}: {CLAIMS[k]['desc']}" for k in keys_for(task["topic"]))
    docs = f"=== {document['title']} ===\n{document['text'].rstrip()}" if document else "(none)"
    shown = "\n".join(f"- [{n['at']}] {n['role']}: {n['note']}" for n in notes) or "(none)"
    user = USER_PROMPT.format(now=now, project=task["project"], desc=CLAIMS[task["claim_key"]]["desc"],
                              keys=keys, documents=docs, notes=shown)
    return [{"role": "system", "content": SYSTEM_PROMPT.format(role=role, role_desc=ROLE_DESC[role])},
            {"role": "user", "content": user}]


def messages_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


class ReplyError(ValueError):
    pass


def parse_reply(text: str, topic: str) -> dict[str, Any]:
    """Extract the agent's JSON. Raises ReplyError if it is unusable; the raw text is kept either way."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ReplyError("no JSON object in reply")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReplyError(f"invalid JSON: {exc}") from exc
    note = str(data.get("note") or "").strip()
    if not note or data.get("claim_value") in (None, ""):
        raise ReplyError("missing note or claim_value")
    raw_key = str(data.get("claim_key") or "").strip()
    key = raw_key if raw_key in keys_for(topic) else None
    expires = data.get("expires_at")
    valid_until = None
    if expires not in (None, "", "null"):
        try:
            valid_until = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
            valid_until = valid_until.replace(tzinfo=UTC) if valid_until.tzinfo is None else valid_until.astimezone(UTC)
        except ValueError:
            valid_until = None
    return {
        "claim_key_raw": raw_key, "claim_key": key,
        "claim_value_raw": str(data["claim_value"]),
        "claim_value": normalize(key, data["claim_value"]) if key else None,
        "expires_at_raw": None if expires in (None, "", "null") else str(expires),
        "valid_until": valid_until.isoformat() if valid_until else None,
        "note": note, "other_note": str(data.get("other_note") or "").strip(),
    }


# --------------------------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    returned_model: str
    usage: dict[str, int]
    finish_reason: str | None = None


class Client(Protocol):
    provider: str
    model_id: str

    def complete(self, messages: list[dict[str, str]], params: dict[str, Any]) -> Reply: ...


class TransientError(RuntimeError):
    pass


class OpenAICompatClient:
    """xAI and OpenAI through the OpenAI SDK, with the keys already in the environment."""

    def __init__(self, provider: str, model_id: str) -> None:
        key_env, url_env, url_default = PROVIDERS[provider]
        if not os.getenv(key_env, "").strip():
            raise SystemExit(f"{key_env} is not set. Stopping: credentials are not created or changed by this tool.")
        from openai import OpenAI

        self.provider, self.model_id = provider, model_id
        self._client = OpenAI(base_url=os.getenv(url_env, url_default), max_retries=0)

    def complete(self, messages: list[dict[str, str]], params: dict[str, Any]) -> Reply:
        import openai

        try:
            resp = self._client.chat.completions.create(model=self.model_id, messages=messages, **params)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
            raise TransientError(type(exc).__name__) from exc
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
        usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                 "reasoning_tokens": getattr(details, "reasoning_tokens", None) or 0, "cached_tokens": cached or 0}
        return Reply(resp.choices[0].message.content or "", resp.model, usage, resp.choices[0].finish_reason)


class MockClient:
    """Offline stand-in. Reads the fact's line from the first document shown (so a stale view
    yields the stale value), else repeats the last team note. Deterministic."""

    provider = "mock"

    def __init__(self, model_id: str = "mock-1", *, bad_reply_every: int = 0, never_expires: bool = False) -> None:
        self.model_id = model_id
        self.bad_reply_every = bad_reply_every
        self.never_expires = never_expires
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], params: dict[str, Any]) -> Reply:
        self.calls += 1
        user = messages[1]["content"]
        usage = {"prompt_tokens": len(user) // 4, "completion_tokens": 60, "reasoning_tokens": 0, "cached_tokens": 0}
        if self.bad_reply_every and self.calls % self.bad_reply_every == 0:
            return Reply("Sorry, here is my note without JSON.", self.model_id, usage, "stop")
        desc = re.search(r"^Fact to record: (.+)$", user, re.M).group(1)
        key = next(k for k, c in CLAIMS.items() if c["desc"] == desc and f"- {k}:" in user)
        label = CLAIMS[key]["label"]
        docs = user.split("Documents you can see:\n", 1)[1].split("\n\nTeam notes so far", 1)[0]
        value, expires = None, None
        for line in docs.splitlines():
            if label in line and not line.startswith("# NOTE"):
                value = normalize(key, line.split("==", 1)[-1] if "==" in line else line.split(label, 1)[1])
                if value is not None:
                    break
        m = re.search(r"temporary override until (\S+);", docs)
        if m and not self.never_expires:
            expires = m.group(1)
        if value is None:
            notes = re.findall(r"^- \[[^]]+\] \w+: (.+)$", user.split("Team notes so far", 1)[1], re.M)
            value = next((normalize(key, n.split(" is ", 1)[-1]) for n in reversed(notes)), None) or "unknown"
        reply = {"claim_key": key, "claim_value": value, "expires_at": expires,
                 "note": f"{label} is {value}", "other_note": "Keep the README in sync with the deploy steps."}
        return Reply(json.dumps(reply), self.model_id, usage, "stop")


# --------------------------------------------------------------------------------------------
# Ledger: cost, budget, model identity, trace records
# --------------------------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    pass


class ModelMismatch(RuntimeError):
    pass


def returned_model_ok(requested: str, returned: str) -> bool:
    """Exact id, or the same id with a dated snapshot suffix (e.g. -2026-09-21 / -0921)."""
    return returned == requested or re.fullmatch(re.escape(requested) + r"-\d{4}(-\d{2}-\d{2})?", returned or "") is not None


def cost_usd(usage: dict[str, int], m: dict[str, Any]) -> float:
    cached = usage.get("cached_tokens", 0)
    fresh = max(usage["prompt_tokens"] - cached, 0)
    return (fresh * m["price_in"] + cached * m.get("price_cached_in", m["price_in"]) + usage["completion_tokens"] * m["price_out"]) / 1e6


def worst_case_usd(messages: list[dict[str, str]], m: dict[str, Any]) -> float:
    est_in = sum(len(x["content"]) for x in messages) // 3 + 50
    max_out = m["params"].get("max_completion_tokens") or m["params"].get("max_tokens") or 4000
    return (est_in * m["price_in"] + max_out * m["price_out"]) / 1e6


@dataclass
class Ledger:
    """Appends one JSON line per API call to `path`. `spent` includes earlier sessions."""

    path: Path
    prompts_path: Path
    cap_usd: float
    spent: float = 0.0
    _prompt_hashes: set[str] = field(default_factory=set)

    @classmethod
    def open(cls, run_dir: Path, model_dir: Path, cap_usd: float) -> Ledger:
        """One ledger per model directory; the cap covers every model in the run directory."""
        model_dir.mkdir(parents=True, exist_ok=True)
        ledger = cls(model_dir / "agent_calls.jsonl", run_dir / "prompts.jsonl", cap_usd)
        for p in run_dir.glob("*/agent_calls.jsonl"):
            ledger.spent += sum(json.loads(line)["cost_usd"] for line in p.open())
        if ledger.prompts_path.exists():
            ledger._prompt_hashes = {json.loads(line)["prompt_sha256"] for line in ledger.prompts_path.open()}
        return ledger

    def store_prompt(self, digest: str, messages: list[dict[str, str]]) -> None:
        if digest in self._prompt_hashes:
            return
        self._prompt_hashes.add(digest)
        with self.prompts_path.open("a") as f:
            f.write(json.dumps({"prompt_sha256": digest, "messages": messages}) + "\n")

    def call(self, client: Client, model: dict[str, Any], messages: list[dict[str, str]], meta: dict[str, Any],
             max_api_retries: int) -> tuple[Reply, dict[str, Any]]:
        if client.provider != "mock" and self.spent + worst_case_usd(messages, model) > self.cap_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f}; next call could exceed the ${self.cap_usd:.2f} cap")
        digest = messages_sha256(messages)
        self.store_prompt(digest, messages)
        for attempt in range(max_api_retries + 1):
            start = time.perf_counter()
            try:
                reply = client.complete(messages, model["params"])
                break
            except TransientError as exc:
                if attempt == max_api_retries:
                    raise
                self._write({**meta, "prompt_sha256": digest, "error": str(exc), "cost_usd": 0.0, "api_attempt": attempt})
                time.sleep(2 ** attempt)
        latency_ms = (time.perf_counter() - start) * 1000
        cost = 0.0 if client.provider == "mock" else cost_usd(reply.usage, model)
        self.spent += cost
        record = {
            **meta, "origin": "llm_agent", "provider": client.provider, "model_requested": model["model_id"],
            "model_returned": reply.returned_model, "params": model["params"], "prompt_sha256": digest,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"), "latency_ms": round(latency_ms, 1),
            "usage": reply.usage, "cost_usd": round(cost, 8), "finish_reason": reply.finish_reason,
            "response_text": reply.text, "api_attempt": attempt,
        }
        self._write(record)
        if not returned_model_ok(model["model_id"], reply.returned_model):
            raise ModelMismatch(f"requested {model['model_id']!r}, provider returned {reply.returned_model!r}")
        return reply, record

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
