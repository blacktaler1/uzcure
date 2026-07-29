"""Deterministic medication dose-reconciliation gate (external audit #1).

Root of the recurring "apixaban 2.5-5 mg" failure: an ambiguous RANGE dose is
never a safe final medication instruction, and NO gate caught it — prompt
guidance is not enforcement. flag_ambiguous_medication_doses detects a range dose
(format-based, so universal — no drug-specific knowledge), surfaces a mandatory
reconciliation SafetyAlert the LLM cannot omit, and escalates the plan to caution.

Verified BOTH directions: it flags a genuine ambiguous strength range and
escalates; it does NOT flag a clean single dose or a legitimate tablet-COUNT
range; it is idempotent and localized.
"""

from __future__ import annotations

import pytest

from app.features.plan_generation.application.safety_check import (
    flag_ambiguous_medication_doses,
)
from app.features.plan_generation.domain.entities import GeneratedPlan, Verdict


def _plan(meds, lang="en", verdict="proceed"):
    return GeneratedPlan.model_validate(
        {
            "plan_metadata": {
                "patient_name": "X",
                "patient_id": "1",
                "diagnosis_short": "d",
                "start_date": "2026-07-14",
                "reward_day": "S",
                "post_discharge_cap_weeks": 4,
                "post_discharge_note": "n",
            },
            "verdict": verdict,
            "decision": verdict if verdict in ("proceed", "caution") else "proceed",
            "output_language": lang,
            "phases": [
                {
                    "phase_number": 1,
                    "phase_name": "P",
                    "duration": "1w",
                    "goal": "g",
                    "exercises": [{"name": "x"}],
                }
            ],
            "diet_plan": {},
            "emergency_action": "c",
            "actions": {"approve_button": "a", "edit_button": "e", "reject_button": "r"},
            "medication_schedule": meds,
        }
    )


def _med(medication, dose):
    return {
        "time": "AM",
        "medication": medication,
        "dose": dose,
        "instructions": "",
        "duration": "long",
    }


def test_flags_range_in_medication_name_and_escalates():
    # The exact audit case: the range lives in the medication NAME.
    plan = _plan([_med("Apixaban 2.5-5 mg", "1 tablet")])
    flagged = flag_ambiguous_medication_doses(plan)
    assert flagged == ["Apixaban 2.5-5 mg"]
    assert plan.verdict == Verdict.CAUTION
    assert plan.decision == "caution"
    alert = plan.safety_alerts[-1]
    assert "2.5-5 mg" in alert.detail
    assert "reconcile" in alert.detail.lower() or "physician" in alert.detail.lower()


@pytest.mark.parametrize(
    "dose",
    ["2.5-5 mg", "2.5 – 5 mg", "2.5 to 5 mg", "5-10 mcg", "50 - 100 ml"],
)
def test_flags_various_range_formats(dose):
    plan = _plan([_med("Drug", dose)])
    assert flag_ambiguous_medication_doses(plan) == ["Drug"]


@pytest.mark.parametrize(
    "dose",
    [
        "2.5 mg - 5 mg",
        "2.5 mg – 5 mg",
        "2.5 mg to 5 mg",
        "5 mcg - 10 mcg",
        "50 ml - 100 ml",
        "2.5 мг – 5 мг",
        "2.5mg-5mg",
    ],
)
def test_flags_expanded_range_with_unit_after_each_number(dose):
    """#1: the EXPANDED range form repeats the unit after BOTH numbers
    ("2.5 mg – 5 mg"). The gate previously required the separator immediately
    after the first number, so this equally-common form silently passed — no
    reconciliation alert, no caution escalation, defeating the gate whose whole
    purpose is to root the "apixaban 2.5–5 mg" class."""
    plan = _plan([_med("Drug", dose)])
    assert flag_ambiguous_medication_doses(plan) == ["Drug"], f"missed expanded range: {dose!r}"
    assert plan.verdict == Verdict.CAUTION


@pytest.mark.parametrize(
    "medication,dose",
    [
        ("Bisoprolol", "5 mg"),  # clean single strength
        ("Paracetamol", "1-2 tablets as needed"),  # COUNT range, not strength — legit PRN
        ("Vitamin D", "1000 units"),  # single
        ("Insulin", "morning and evening"),  # no numbers
        # Fixed-dose COMBINATION ratios use "/" — the patient takes that exact
        # combined tablet, so they must NOT be flagged as an ambiguous range to
        # reconcile (a range uses a dash / "to" / "до", never "/").
        ("Levodopa/carbidopa 100/25 mg", "1 tablet"),
        ("Co-careldopa", "25/100 mg"),
        ("Co-amoxiclav", "500/125 mg"),
    ],
)
def test_does_not_flag_unambiguous_or_count_ranges(medication, dose):
    plan = _plan([_med(medication, dose)])
    assert flag_ambiguous_medication_doses(plan) == []
    assert plan.verdict == Verdict.PROCEED  # not escalated


def test_localized_russian_alert():
    plan = _plan([_med("Апиксабан 2.5-5 мг", "1 таблетка")], lang="ru")
    assert flag_ambiguous_medication_doses(plan) == ["Апиксабан 2.5-5 мг"]
    assert plan.safety_alerts[-1].alert == "Уточните дозу препарата"


def test_idempotent_no_double_flag():
    plan = _plan([_med("Apixaban", "2.5-5 mg")])
    flag_ambiguous_medication_doses(plan)
    n = len(plan.safety_alerts)
    flag_ambiguous_medication_doses(plan)  # second pass (e.g. regeneration)
    assert len(plan.safety_alerts) == n


def test_never_downgrades_an_abstaining_verdict():
    # An abstaining plan is validly shaped: empty phases + a reason.
    plan = GeneratedPlan.model_validate(
        {
            "plan_metadata": {
                "patient_name": "X",
                "patient_id": "1",
                "diagnosis_short": "d",
                "start_date": "2026-07-14",
                "reward_day": "S",
                "post_discharge_cap_weeks": 4,
                "post_discharge_note": "n",
            },
            "verdict": "contraindicated",
            "abstention_reason": "unstable — withhold plan",
            "output_language": "en",
            "phases": [],
            "diet_plan": {},
            "emergency_action": "c",
            "actions": {"approve_button": "a", "edit_button": "e", "reject_button": "r"},
            "medication_schedule": [_med("Apixaban", "2.5-5 mg")],
        }
    )
    flag_ambiguous_medication_doses(plan)
    assert plan.verdict == Verdict.CONTRAINDICATED  # never downgraded to caution
