"""HMAC-signed audit row for every generated plan.

Records plan integrity hash + signature, the external evidence (PMIDs) that
grounded the plan, a drug-safety fact summary, and LLM token meta. For a
patient-linked plan this row is MANDATORY: the worker fails the job closed
(LedgerWriteError) if it cannot be committed, rather than surface an unauditable
plan. Only the patient-less (case-history-only) flow legitimately skips it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.features.plan_generation.application.ports import GenerationOutcome
from app.features.plan_generation.domain import GeneratePlanCommand
from app.features.plan_generation.settings import get_plan_v2_settings
from app.models.evidence_ledger import EvidenceLedger

logger = logging.getLogger(__name__)


def build_grounding_summary(plan: Any) -> dict[str, Any]:
    """Summarise the evidence that GROUNDED this plan, for the ledger audit row.

    Two grounding kinds are recorded so the audit reflects REAL grounding:
      * PMID grounding — invented/registry-miss exercises, supplements, and
        medications cite retrieved PMIDs (PMID-gated + NCBI-verified before emit).
      * Guideline grounding — a registry-backed exercise is cited by its guideline
        source + GRADE (evidence_source/evidence_level), NOT a PMID. The ledger
        previously recorded ONLY PMIDs, so a plan built entirely from A/B-graded,
        guideline-cited registry exercises was ledgered as unique_pmid_count=0 /
        grounding_evidence=[] — falsely ungrounded. This records that provenance too.
    """
    grounding_evidence: list[dict[str, Any]] = []
    all_pmids: set[str] = set()
    guideline_grounded = 0
    for phase in plan.phases:
        for ex in phase.exercises:
            if ex.evidence_pmids:
                grounding_evidence.append(
                    {
                        "element": "exercise",
                        "ref": ex.exercise_id or ex.name,
                        "pmids": list(ex.evidence_pmids),
                    }
                )
                all_pmids.update(ex.evidence_pmids)
            elif getattr(ex, "evidence_source", ""):
                grounding_evidence.append(
                    {
                        "element": "exercise",
                        "ref": ex.exercise_id or ex.name,
                        "guideline": ex.evidence_source,
                        "grade": getattr(ex, "evidence_level", "") or None,
                    }
                )
                guideline_grounded += 1
    for sup in plan.diet_plan.supplements:
        if sup.evidence_pmids:
            grounding_evidence.append(
                {"element": "supplement", "ref": sup.name, "pmids": list(sup.evidence_pmids)}
            )
            all_pmids.update(sup.evidence_pmids)
    for med in plan.medication_schedule:
        if med.evidence_pmids:
            grounding_evidence.append(
                {"element": "medication", "ref": med.medication, "pmids": list(med.evidence_pmids)}
            )
            all_pmids.update(med.evidence_pmids)
    return {
        "grounding_evidence": grounding_evidence,
        "unique_pmid_count": len(all_pmids),
        "unique_pmids": sorted(all_pmids),
        "guideline_grounded_count": guideline_grounded,
    }


def _sign(plan_dict: dict[str, Any], key: str) -> tuple[str, str]:
    canonical = json.dumps(plan_dict, sort_keys=True, separators=(",", ":"), default=str)
    plan_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    signature = hmac.new(key.encode("utf-8"), plan_hash.encode("utf-8"), hashlib.sha256).hexdigest()
    return plan_hash, signature


def write_ledger_entry(
    db: Session,
    *,
    cmd: GeneratePlanCommand,
    outcome: GenerationOutcome,
) -> uuid.UUID | None:
    """Persist tamper-evident audit row. Returns plan_id (UUID) or None when skipped.

    Skipped when patient_id is None — the case-history-only flow never persists
    a patient row, so the FK target is missing.
    """
    if cmd.patient_id is None:
        return None

    settings = get_settings()
    v2 = get_plan_v2_settings()

    plan_dict = outcome.plan.to_api_dict()
    plan_hash, signature = _sign(plan_dict, settings.MYREHAB_SIGNING_KEY_V1)

    # Drug interactions are now LLM-detected (no deterministic oracle). Record the
    # interactions the plan itself surfaced so the audit ledger still captures the
    # drug-safety signal; provenance is the model + retrieved evidence.
    plan_dis = outcome.plan.drug_interactions
    drug_verification = {
        "drug_interactions": [
            {"title": di.title, "severity": di.severity, "source": di.source} for di in plan_dis
        ],
    }
    safety_rules_fired = sorted(
        {f"DRUG:{di.title}:{di.severity}" for di in plan_dis if di.severity in ("Block", "Caution")}
    )
    exercise_ids_prescribed = [
        ex.exercise_id for phase in outcome.plan.phases for ex in phase.exercises if ex.exercise_id
    ]

    pubmed_queries = build_grounding_summary(outcome.plan)

    # Policy: every generated plan goes through explicit doctor review.
    # No auto-approve path exists. Flag is fixed in the ledger row so legacy
    # consumers (frontend evidence panel, pipeline_health_service) keep working.
    entry = EvidenceLedger(
        plan_id=uuid.uuid4(),
        patient_id=cmd.patient_id,
        plan_version=1,
        guideline_pack_ids=[],
        safety_rules_fired=safety_rules_fired,
        drug_verification=drug_verification,
        exercise_ids_prescribed=exercise_ids_prescribed,
        llm_provider="anthropic",
        llm_model_version=outcome.llm_model or v2.PLAN_V2_MODEL,
        llm_tokens_used=outcome.llm_input_tokens + outcome.llm_output_tokens,
        llm_cost_usd=round(float(outcome.estimated_cost_usd), 4),
        is_fully_compliant=False,
        requires_doctor_review=True,
        plan_hash=plan_hash,
        ledger_signature=signature,
        signing_key_version=1,
        delivery_decision="REVIEW_REQUIRED",
        # Grounding evidence recorded durably in the audit row.
        pubmed_queries=pubmed_queries,
    )
    db.add(entry)
    db.commit()
    return entry.plan_id
