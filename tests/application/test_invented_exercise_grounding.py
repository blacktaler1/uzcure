"""Ungrounded invented exercises are WITHHELD, not allowed to sink the plan.

CHARTER: when the registry is insufficient the model must activate external
retrieval and ground the recommendation in retrieved evidence, and must NEVER
fill evidence gaps with model memory. An `ai_suggested` exercise with empty
`evidence_pmids` is model memory by definition and must not reach a patient.

The charter names FOUR permitted fail-closed responses: limiting the output,
requiring clinician review, requesting more clinical information, or abstaining.
An earlier version of this gate RAISED instead, discarding the whole plan.
Measured over 10 production generations it rejected 60% of them — throwing away
grounded registry exercises and grounded inventions alongside the ungrounded
ones. Failing closed is required; destroying a usable plan is not.

So: drop the ungrounded inventions, keep everything else, escalate with a reason.
"""

from __future__ import annotations

from app.features.plan_generation.application.use_case import (
    _build_escalation_reasons,
    _withhold_ungrounded_inventions,
)
from app.features.plan_generation.domain.entities import Exercise, GeneratedPlan, Phase


class _FakeSafety:
    """Minimal stand-in for SafetyCheckResult — only the fields the reason
    builder reads for these assertions."""

    allergy_hits: list = []
    contraindications: list = []
    clamps: list = []
    synthesis: list = []
    red_flags: list = []


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


def _names(plan):
    return [ex.name for ph in plan.phases for ex in ph.exercises]


# ── the charter rule: model memory must not reach the patient ──────────────
def test_ungrounded_invention_is_removed():
    plan = _plan(_ex("Invented, no evidence", invented=True))
    meta: dict = {}
    withheld = _withhold_ungrounded_inventions(plan, meta)
    assert _names(plan) == []
    assert withheld == ["Invented, no evidence"]
    assert meta["ungrounded_inventions_withheld"] == ["Invented, no evidence"]


def test_it_does_not_raise():
    """The whole point of the change: this must never sink the generation."""
    plan = _plan(_ex("Invented", invented=True))
    _withhold_ungrounded_inventions(plan, {})  # must not raise


# ── it must LIMIT, not destroy ─────────────────────────────────────────────
def test_grounded_work_survives_alongside_an_offender():
    """The production shape that motivated this: grounded registry exercises and
    a grounded invention sitting beside ungrounded ones. Discarding the plan
    threw all of it away."""
    plan = _plan(
        _ex("Registry bridging", invented=False),
        _ex("Invented but grounded", invented=True, pmids=["GOOD1"]),
        _ex("Invented, ungrounded", invented=True),
    )
    withheld = _withhold_ungrounded_inventions(plan, {})
    assert _names(plan) == ["Registry bridging", "Invented but grounded"]
    assert withheld == ["Invented, ungrounded"]


def test_registry_exercise_without_pmids_is_kept():
    """A registry exercise is cited by guideline source + GRADE, not a PMID.
    Requiring PMIDs of them would empty every ordinary plan."""
    plan = _plan(_ex("Registry only", invented=False))
    assert _withhold_ungrounded_inventions(plan, {}) == []
    assert _names(plan) == ["Registry only"]


def test_clean_plan_is_untouched_and_records_nothing():
    plan = _plan(
        _ex("Registry", invented=False),
        _ex("Invented grounded", invented=True, pmids=["GOOD1"]),
    )
    meta: dict = {}
    assert _withhold_ungrounded_inventions(plan, meta) == []
    assert _names(plan) == ["Registry", "Invented grounded"]
    assert "ungrounded_inventions_withheld" not in meta


def test_withholding_spans_every_phase():
    plan = GeneratedPlan.model_construct(
        phases=[
            Phase.model_construct(exercises=[_ex("P1 bad", invented=True)]),
            Phase.model_construct(
                exercises=[
                    _ex("P2 ok", invented=True, pmids=["G"]),
                    _ex("P2 bad", invented=True),
                ]
            ),
        ],
        medication_schedule=[],
    )
    withheld = _withhold_ungrounded_inventions(plan, {})
    assert withheld == ["P1 bad", "P2 bad"]
    assert _names(plan) == ["P2 ok"]


# ── the removal must be visible to the doctor ──────────────────────────────
def test_withholding_escalates_with_a_named_reason():
    """A silently thinner plan is its own defect — the doctor must see that the
    plan was LIMITED, not merely short."""
    plan = _plan(_ex("Registry", invented=False))
    reasons = _build_escalation_reasons(plan, _FakeSafety(), withheld_inventions=["A", "B"])
    codes = {r["code"]: r for r in reasons}
    assert "ungrounded_invention_withheld" in codes
    assert codes["ungrounded_invention_withheld"]["count"] == 2


def test_no_withholding_means_no_such_reason():
    plan = _plan(_ex("Registry", invented=False))
    reasons = _build_escalation_reasons(plan, _FakeSafety(), withheld_inventions=[])
    assert "ungrounded_invention_withheld" not in {r["code"] for r in reasons}
