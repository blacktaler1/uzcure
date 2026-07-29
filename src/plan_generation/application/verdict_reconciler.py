"""Verdict reconciler — soft post-LLM checks.

Runs AFTER the LLM returns a parsed plan, BEFORE exercise resolution and
safety check. Never raises, never blocks generation. Never strips phases —
the rehab plan with phases is the product, even on emergency cases.

What this layer does:
  1. Upgrades verdict from `proceed` → `caution` when the LLM emitted ≥3 Block
     drug_interactions or safety_alerts contain emergency-language patterns —
     LLM contradicted itself, we keep the higher-risk side.
  2. Logs every adjustment for audit.

The reconciler is a *safety net*, not a clinical gate. The LLM remains the
primary decision-maker. This layer only catches obvious LLM self-contradiction.
"""

from __future__ import annotations

import logging
import re

from app.features.plan_generation.domain import GeneratedPlan, Verdict

logger = logging.getLogger(__name__)


_EMERGENCY_PATTERNS = re.compile(
    r"(🚨|emergency|shoshilinch|критич|неотложн|крит|шошилинч|hayotga\s*xavf|"
    r"life[-\s]*threat|cardiac\s*arrest|septic|shock|reanimats|реаним)",
    re.IGNORECASE,
)


def reconcile_verdict(plan: GeneratedPlan) -> list[str]:
    """Mutates `plan` in-place. Returns a list of human-readable adjustment notes
    for the audit log."""

    notes: list[str] = []

    # Upgrade proceed → caution if LLM contradicted itself.
    if plan.verdict == Verdict.PROCEED:
        block_count = sum(1 for di in plan.drug_interactions if di.severity == "Block")
        if block_count >= 3:
            notes.append(
                f"verdict=proceed but {block_count} Block drug_interactions — upgraded to caution."
            )
            plan.verdict = Verdict.CAUTION

    if plan.verdict == Verdict.PROCEED:
        emergency_match = any(
            _EMERGENCY_PATTERNS.search(sa.alert + " " + sa.detail) for sa in plan.safety_alerts
        )
        if emergency_match:
            notes.append(
                "verdict=proceed but safety_alerts contain emergency language — upgraded to caution."
            )
            plan.verdict = Verdict.CAUTION

    # Sync the legacy 2-value decision field after any verdict change. An
    # abstaining verdict (insufficient_information / contraindicated /
    # evidence_unavailable) is not a valid `decision` value and must map to
    # "caution" so the review UI never reads it as an actionable "proceed".
    plan.decision = (  # type: ignore[assignment]
        "caution"
        if (plan.verdict == Verdict.CAUTION or plan.verdict.abstains)
        else plan.verdict.value
    )

    if notes:
        logger.warning(
            "verdict_reconciler adjusted final_verdict=%s notes=%s",
            plan.verdict.value,
            " | ".join(notes),
        )

    return notes


__all__ = ["reconcile_verdict"]
