"""Grounding repair: re-ask the model to cite, never let it invent.

Measured live 2026-07-19: the model INVERTS the citation contract on some cases.
It attaches PMIDs to registry exercises (output_schema.py:281 says omit for pool
exercises) and omits them on invented ones (prompt_builder.py:609 requires them).
Two runs, same prompt and same deployed commit:

    p52  invented 2, grounded 2/2
    p53  invented 5, grounded 0/5   -- while holding 5 exercise-domain articles
                                        retrieved expressly to ground inventions

So grounding cannot rest on model compliance. The repair re-asks for ONLY the
ungrounded inventions, offering ONLY the already-retrieved pack.

Every safety property below is enforced server-side rather than requested in the
prompt, because the prompt is exactly what already failed.
"""

from __future__ import annotations

import asyncio

import pytest

from app.features.plan_generation.application.ports import LLMResponse
from app.features.plan_generation.application.use_case import GeneratePlanUseCase
from app.features.plan_generation.domain.entities import Exercise, GeneratedPlan, Phase


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _Article:
    def __init__(self, pmid, title="t"):
        self.pmid = pmid
        self.title = title


class _FakeLLM:
    def __init__(self, citations=None, raise_exc=None):
        self.citations = citations or []
        self.raise_exc = raise_exc
        self.calls = 0
        self.last_message = ""

    async def call_with_tool(self, req):
        self.calls += 1
        self.last_message = req.user_message
        if self.raise_exc:
            raise self.raise_exc
        return LLMResponse(
            tool_input={"citations": self.citations},
            raw_text="",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=1,
            model="test",
            estimated_cost_usd=0.0,
        )


def _ex(name, *, invented, pmids=None):
    return Exercise.model_construct(
        name=name,
        ai_suggested=invented,
        exercise_id=None if invented else "REG_1",
        evidence_pmids=list(pmids or []),
        evidence_source="",
        evidence_level="",
    )


def _plan(*exercises):
    return GeneratedPlan.model_construct(
        phases=[Phase.model_construct(exercises=list(exercises))], medication_schedule=[]
    )


def _uc(llm):
    uc = object.__new__(GeneratePlanUseCase)
    uc._llm = llm
    return uc


def _repair(uc, plan, evidence, meta):
    return _run(
        GeneratePlanUseCase._repair_invented_grounding(
            uc, plan=plan, evidence=evidence, cmd=None, safety_meta=meta
        )
    )


# ── the defect it exists to fix ─────────────────────────────────────────────
def test_ungrounded_invented_exercise_is_repaired():
    plan = _plan(_ex("Invented sit-to-stand", invented=True))
    llm = _FakeLLM([{"exercise_name": "Invented sit-to-stand", "pmids": ["111"]}])
    meta: dict = {}
    _repair(_uc(llm), plan, [_Article("111")], meta)
    assert plan.phases[0].exercises[0].evidence_pmids == ["111"]
    assert meta["grounding_repair"]["repaired"] == 1


# ── it must never become a fabrication channel ──────────────────────────────
def test_pmid_outside_the_retrieved_pack_is_rejected():
    """The whole point of the pack is that these PMIDs were really retrieved. A
    PMID the model invents during repair is discarded HERE, before the NCBI gate
    ever sees it."""
    plan = _plan(_ex("Invented", invented=True))
    llm = _FakeLLM([{"exercise_name": "Invented", "pmids": ["999999"]}])
    meta: dict = {}
    _repair(_uc(llm), plan, [_Article("111")], meta)
    assert plan.phases[0].exercises[0].evidence_pmids == []
    assert meta["grounding_repair"]["out_of_pack_pmids_rejected"] == 1
    assert meta["grounding_repair"]["repaired"] == 0


def test_mixed_response_keeps_only_in_pack_pmids():
    plan = _plan(_ex("Invented", invented=True))
    llm = _FakeLLM([{"exercise_name": "Invented", "pmids": ["111", "999999"]}])
    meta: dict = {}
    _repair(_uc(llm), plan, [_Article("111")], meta)
    assert plan.phases[0].exercises[0].evidence_pmids == ["111"]
    assert meta["grounding_repair"]["out_of_pack_pmids_rejected"] == 1


# ── it must not touch anything it was not asked to ─────────────────────────
def test_already_grounded_invented_exercise_is_not_rewritten():
    """The first call's per-exercise judgement is authoritative. Repair may only
    fill gaps, never overwrite."""
    plan = _plan(_ex("Already cited", invented=True, pmids=["111"]))
    llm = _FakeLLM([{"exercise_name": "Already cited", "pmids": ["222"]}])
    _repair(_uc(llm), plan, [_Article("111"), _Article("222")], {})
    assert plan.phases[0].exercises[0].evidence_pmids == ["111"]
    assert llm.calls == 0, "no ungrounded inventions -> no call at all"


def test_registry_exercises_are_never_touched():
    """A registry exercise is cited by guideline source + GRADE, not a PMID."""
    plan = _plan(_ex("Registry", invented=False))
    llm = _FakeLLM([{"exercise_name": "Registry", "pmids": ["111"]}])
    _repair(_uc(llm), plan, [_Article("111")], {})
    assert plan.phases[0].exercises[0].evidence_pmids == []
    assert llm.calls == 0


def test_unknown_exercise_name_in_response_is_ignored():
    plan = _plan(_ex("Invented", invented=True))
    llm = _FakeLLM([{"exercise_name": "Something else entirely", "pmids": ["111"]}])
    _repair(_uc(llm), plan, [_Article("111")], {})
    assert plan.phases[0].exercises[0].evidence_pmids == []


# ── it must never fail a generation that would otherwise succeed ───────────
def test_llm_failure_is_swallowed_and_plan_untouched():
    plan = _plan(_ex("Invented", invented=True))
    llm = _FakeLLM(raise_exc=RuntimeError("provider down"))
    meta: dict = {}
    _repair(_uc(llm), plan, [_Article("111")], meta)
    assert plan.phases[0].exercises[0].evidence_pmids == []
    assert meta["grounding_repair"]["error"] == "RuntimeError"


def test_empty_evidence_pack_makes_no_call():
    plan = _plan(_ex("Invented", invented=True))
    llm = _FakeLLM([{"exercise_name": "Invented", "pmids": ["111"]}])
    _repair(_uc(llm), plan, [], {})
    assert llm.calls == 0, "nothing to cite from -> do not spend a call"


def test_no_llm_wired_is_a_noop():
    plan = _plan(_ex("Invented", invented=True))
    _repair(_uc(None), plan, [_Article("111")], {})
    assert plan.phases[0].exercises[0].evidence_pmids == []


# ── the prompt must permit an honest empty answer ──────────────────────────
def test_repair_prompt_allows_returning_nothing():
    """If the repair demanded a citation per exercise it would manufacture one.
    The instruction must make an empty array the correct answer when nothing
    genuinely supports the exercise."""
    plan = _plan(_ex("Invented", invented=True))
    llm = _FakeLLM([])
    _repair(_uc(llm), plan, [_Article("111")], {})
    assert "EMPTY pmids array" in llm.last_message
    assert "never invent" in llm.last_message.lower()
