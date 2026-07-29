"""An ABSTAINING plan must always escalate the doctor's 'review now' push.

Regression guard for the safety-triage bug where an emergency abstention
(verdict=contraindicated — e.g. suspected DVT/PE, raised ICP, cord compression)
was mislabeled escalation_level='routine' whenever it lacked a Block drug
interaction (so it produced no escalation reason and no urgent-review push).
"""

from __future__ import annotations

import pytest

from app.features.plan_generation.application.safety_check import SafetyCheckResult
from app.features.plan_generation.application.use_case import _build_escalation_reasons
from app.features.plan_generation.domain.entities import GeneratedPlan, Verdict


def _abstaining_plan(verdict: str) -> GeneratedPlan:
    return GeneratedPlan.model_validate(
        {
            "plan_metadata": {
                "patient_name": "X",
                "patient_id": "1",
                "diagnosis_short": "d",
                "start_date": "2026-07-15",
                "reward_day": "S",
                "post_discharge_cap_weeks": 4,
                "post_discharge_note": "n",
            },
            "verdict": verdict,
            "decision": "caution",
            "abstention_reason": "Acute emergency — rehabilitation unsafe until stabilized.",
            "output_language": "en",
            "phases": [],  # abstaining plans carry NO plan body
            "diet_plan": {},
            "emergency_action": "Seek urgent care.",
            "actions": {"approve_button": "a", "edit_button": "e", "reject_button": "r"},
        }
    )


@pytest.mark.parametrize(
    "verdict", ["contraindicated", "insufficient_information", "evidence_unavailable"]
)
def test_abstaining_plan_always_escalates(verdict):
    plan = _abstaining_plan(verdict)
    assert plan.verdict.abstains  # precondition
    reasons = _build_escalation_reasons(plan, SafetyCheckResult())
    codes = [r["code"] for r in reasons]
    assert "plan_withheld" in codes
    # The reason names WHICH abstaining verdict, so the doctor sees why it was held.
    withheld = next(r for r in reasons if r["code"] == "plan_withheld")
    assert withheld["verdict"] == verdict
    # escalation_level is derived as escalated when any reason exists.
    assert bool(reasons) is True


def test_abstention_escalates_even_without_block_ddi():
    """The exact live failure (Case 3/4): emergency abstention, no Block DDI."""
    plan = _abstaining_plan("contraindicated")
    assert not any(di.severity == "Block" for di in plan.drug_interactions)  # no block_ddi
    reasons = _build_escalation_reasons(plan, SafetyCheckResult())
    assert reasons, "abstention with no Block-DDI must still escalate (was 'routine')"
    assert reasons[0]["code"] == "plan_withheld"


def test_proceed_plan_not_escalated_by_abstention_rule():
    """A normal proceeding plan must NOT get the plan_withheld reason."""
    plan = GeneratedPlan.model_validate(
        {
            "plan_metadata": {
                "patient_name": "X",
                "patient_id": "1",
                "diagnosis_short": "d",
                "start_date": "2026-07-15",
                "reward_day": "S",
                "post_discharge_cap_weeks": 4,
                "post_discharge_note": "n",
            },
            "verdict": "proceed",
            "decision": "proceed",
            "output_language": "en",
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
        }
    )
    reasons = _build_escalation_reasons(plan, SafetyCheckResult())
    assert "plan_withheld" not in [r["code"] for r in reasons]
