"""xAI Grok client for answer generation and LLM-as-judge scoring."""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any

from baselines.techniques import technique_label
from utils.logging import get_logger

logger = get_logger(__name__)


def xai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai: pip install -e '.[llm]'") from exc

    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "XAI_API_KEY not set. Add it to .env.local (see .env.example)."
        )
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def default_model() -> str:
    return os.getenv("XAI_MODEL", "grok-3-mini")


def generate_answer(
    query: str,
    memory_context: str,
    *,
    technique: str,
    model: str | None = None,
) -> str:
    """Answer using only retrieved memory context."""
    client = xai_client()
    model_name = model or default_model()
    system = textwrap.dedent(
        """\
        You answer questions using ONLY the retrieved team memories below.
        Be concise and factual. If memories lack the answer, say you do not have it.
        Do not invent infrastructure details not supported by the memories.
        """
    )
    user = textwrap.dedent(
        f"""\
        Memory backend: {technique_label(technique)}

        Question: {query}

        Retrieved memories:
        {memory_context}
        """
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


def judge_answer(
    *,
    task_id: str,
    query: str,
    required_terms: tuple[str, ...],
    forbidden_terms: tuple[str, ...],
    memory_context: str,
    answer: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    LLM-as-judge: score factual alignment with rubric.

    Returns dict with judge_score, reasoning, hallucination, used_context.
    """
    client = xai_client()
    model_name = model or default_model()
    rubric = {
        "required_facts": list(required_terms),
        "forbidden_leaks": list(forbidden_terms),
    }
    system = textwrap.dedent(
        """\
        You are an evaluation judge for a memory-augmented QA system.
        Score the ANSWER against the rubric using ONLY the retrieved memories as ground truth.
        Reply with JSON only:
        {
          "judge_score": <float 0.0-1.0>,
          "required_terms_found": [<strings>],
          "forbidden_leak": <bool>,
          "used_retrieved_context": <bool>,
          "hallucination": <bool>,
          "reasoning": "<one short paragraph>"
        }
        judge_score rubric:
        - 1.0: all required facts present, no forbidden leak, no hallucination
        - 0.5: partial facts or vague but mostly correct
        - 0.0: wrong, missing key facts, forbidden leak, or clear hallucination
        """
    )
    user = textwrap.dedent(
        f"""\
        Task: {task_id}
        Question: {query}
        Rubric: {json.dumps(rubric)}

        Retrieved memories:
        {memory_context}

        Model answer:
        {answer}
        """
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "{}").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning("LLM judge failed (%s), using keyword fallback", exc)

    return _keyword_judge_fallback(
        required_terms=required_terms,
        forbidden_terms=forbidden_terms,
        answer=answer,
    )


def _keyword_judge_fallback(
    *,
    required_terms: tuple[str, ...],
    forbidden_terms: tuple[str, ...],
    answer: str,
) -> dict[str, Any]:
    from evaluation.llm_fixtures import has_forbidden, term_recall

    recall = term_recall(required_terms, answer)
    leak = has_forbidden(forbidden_terms, answer)
    score = 0.0 if leak else recall
    return {
        "judge_score": round(score, 3),
        "required_terms_found": [t for t in required_terms if t.lower() in answer.lower()],
        "forbidden_leak": leak,
        "used_retrieved_context": recall > 0,
        "hallucination": False,
        "reasoning": "Keyword fallback (judge API parse failed or unavailable).",
    }


def parse_judge_score(judge: dict[str, Any]) -> float:
    try:
        return float(judge.get("judge_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
