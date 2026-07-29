"""Property-based safety-invariant tests for the Warn-Never-Block CRITICAL
acknowledgment gate (Document 1: "property-based safety-invariant tests").

WHY THIS EXISTS
Example tests check a handful of hand-picked plans. This file asserts the
*invariant* across a generated space of plan-result shapes, which is the kind of
test that would have caught the C-6 dual-path asymmetry (the legacy /plan-approval
path gated a Block-severity drug interaction, while the newer ApprovePlanUseCase
path did not) without anyone thinking to write that specific example.

THE INVARIANT (both halves):
  1. ApprovePlanUseCase.execute raises CriticalAckRequired  IF AND ONLY IF
     the plan result carries a Block-severity drug interaction AND the approving
     doctor did not set critical_safety_acknowledged.
  2. When a Block-severity interaction is present and NOT acknowledged, approval
     never proceeds — the job is never claimed and no plan is written (a plan with
     an unacknowledged emergency-level interaction can never reach a patient).

These run against the REAL detector (_result_has_block_ddi) and the REAL use case
with in-memory fakes — no mock of the logic under test — so they genuinely guard
the gate, and they run in the normal (SQLite) test job.

Hypothesis generates: the two recognized list keys (drug_interactions /
drug_exercise_interaction_matrix), severity strings in every case + surrounding
whitespace, non-dict / missing-severity list entries, and the acknowledged flag.
"""
import asyncio

import pytest
from hypothesis import given, settings, strategies as st

from app.features.plan_generation.application.approval_use_case import (
    ApprovePlanUseCase,
    _result_has_block_ddi,
)
from app.features.plan_generation.application.ports import JobRecord
from app.features.plan_generation.domain import (
    ApprovalDecision,
    ApprovalDecisionKind,
    CriticalAckRequired,
)


# ── Fakes (mirrors test_approve_critical_ack.py) ────────────────────────────
class _FakeJobRepo:
    def __init__(self, result):
        self._rec = JobRecord(
            job_id="job1", status="succeeded", progress_step="done",
            result=result, error_code=None, error_message=None,
            created_at="2026-01-01T00:00:00Z", started_at=None, completed_at=None,
        )
        self.claimed = False
        self.marked_approved = False

    async def get(self, *, job_id, doctor_id, clinic_id):
        return self._rec

    async def claim_for_approval(self, *, job_id, doctor_id, clinic_id):
        self.claimed = True
        return True

    async def release_approval_claim(self, *, job_id, doctor_id, clinic_id):
        pass

    async def mark_approved(self, *, job_id, doctor_id, clinic_id, plan_id):
        self.marked_approved = True


class _FakeWrite:
    plan_id = 42
    patient_id = 1
    phase_count = 2
    exercise_count = 2
    medication_count = 1
    daily_task_count = 0


class _FakePlanWriter:
    def __init__(self):
        self.wrote = False

    async def write_approved_plan(self, *, job_id, doctor_id, clinic_id, result_json, approved_by_id):
        self.wrote = True
        return _FakeWrite()


def _decision(acknowledged):
    return ApprovalDecision(
        job_id="job1", doctor_id=1, clinic_id=1,
        kind=ApprovalDecisionKind.APPROVE,
        critical_safety_acknowledged=acknowledged,
    )


# ── Strategies ──────────────────────────────────────────────────────────────
# Severity strings that MUST count as "block" (case-insensitive, trimmed).
_block_severity = st.sampled_from(["block", "Block", "BLOCK", "  block", "block  ", " Block "])
# Severity strings that must NOT count as block.
_non_block_severity = st.sampled_from(
    ["", "ok", "OK", "warn", "Warn", "info", "moderate", "blocker", "blocked", "none", "high"]
)
_interaction_key = st.sampled_from(["drug_interactions", "drug_exercise_interaction_matrix"])

# Arbitrary "noise" list entries that must never be mistaken for a block.
_noise_entry = st.one_of(
    st.none(),
    st.integers(),
    st.text(max_size=5),
    st.fixed_dictionaries({"title": st.text(max_size=8)}),  # no severity key
    st.fixed_dictionaries({"severity": _non_block_severity, "title": st.text(max_size=8)}),
)


def _make_block_di(sev):
    return {"title": "Warfarin + NSAID", "severity": sev, "description": "Bleeding risk."}


@st.composite
def plan_results(draw):
    """Generate a (result_json, expected_has_block) pair across the real shapes."""
    key = draw(_interaction_key)
    noise = draw(st.lists(_noise_entry, max_size=4))
    include_block = draw(st.booleans())
    items = list(noise)
    if include_block:
        items.insert(draw(st.integers(min_value=0, max_value=len(items))),
                     _make_block_di(draw(_block_severity)))
    result = {"plan_metadata": {"patient_id": "1"}, "phases": [], key: items}
    return result, include_block


# ── Property 1: detector correctness across the generated space ─────────────
@settings(max_examples=200, deadline=None)
@given(data=plan_results())
def test_block_ddi_detector_matches_expectation(data):
    result, expected = data
    assert _result_has_block_ddi(result) is expected


# Detector must never crash and must return False on degenerate inputs.
@settings(max_examples=100, deadline=None)
@given(
    blob=st.one_of(
        st.none(),
        st.integers(),
        st.text(max_size=10),
        st.dictionaries(st.text(max_size=6), st.text(max_size=6), max_size=4),
        st.lists(st.text(max_size=6), max_size=4),
    )
)
def test_block_ddi_detector_safe_on_garbage(blob):
    assert _result_has_block_ddi(blob) is False  # type: ignore[arg-type]


# ── Property 2: the gate invariant (the C-6 guard) ──────────────────────────
@settings(max_examples=200, deadline=None)
@given(data=plan_results(), acknowledged=st.booleans())
def test_gate_invariant_raises_iff_block_and_unacknowledged(data, acknowledged):
    result, has_block = data
    repo = _FakeJobRepo(result)
    writer = _FakePlanWriter()
    use_case = ApprovePlanUseCase(repo, writer)

    must_block = has_block and not acknowledged

    async def run():
        return await use_case.execute(_decision(acknowledged))

    if must_block:
        # INVARIANT 1+2: an unacknowledged Block-DDI plan is refused AND never
        # claimed/written — it can never reach a patient.
        with pytest.raises(CriticalAckRequired):
            asyncio.run(run())
        assert repo.claimed is False
        assert writer.wrote is False
    else:
        # No block, or block but acknowledged -> approval proceeds normally.
        outcome = asyncio.run(run())
        assert outcome is not None
        assert repo.marked_approved is True
