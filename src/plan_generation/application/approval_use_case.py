"""Approval / delivery use cases (Stage K).

Pure orchestration:

  ApprovePlanUseCase  — validate doctor + job + clinic, copy result_json
                        into rehab_plans tree, mark job approved
  RejectPlanUseCase   — validate, mark job rejected
  EditPlanUseCase     — validate edited result against schema, replace
                        result_json, bump edit_count
  DeliverPlanUseCase  — flip plan to 'delivered', push plan_delivered +
                        drug_interaction_warning (Block-severity) to patient

All raise DomainError; presentation layer maps to HTTP.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from app.features.plan_generation.application.ports import (
    JobRepo,
    PlanWriter,
)
from app.features.plan_generation.domain import (
    ApprovalDecision,
    ApprovalDecisionKind,
    CriticalAckRequired,
    DeliveryOutcome,
    DeliveryRequest,
    GeneratedPlan,
    JobNotApprovable,
    JobNotFound,
    PlanApproval,
    PlanEditOutcome,
    PlanNotDeliverable,
    PlanNotFound,
    PlanRejection,
    TenancyViolation,
)

logger = logging.getLogger(__name__)


def _result_has_block_ddi(result_json: dict | None) -> bool:
    """True if the generated plan carries a "Block"-severity drug interaction.

    Detection mirrors the legacy /plan-approval path (and SqlPlanWriter
    .block_drug_titles): a per-patient critical condition produced by the
    generator from the patient's actual drugs — never a hardcoded rule.
    """
    if not isinstance(result_json, dict):
        return False
    interactions = result_json.get("drug_interactions")
    if not isinstance(interactions, list):
        interactions = result_json.get("drug_exercise_interaction_matrix")
    if not isinstance(interactions, list):
        return False
    for di in interactions:
        if isinstance(di, dict) and str(di.get("severity", "")).strip().lower() == "block":
            return True
    return False


_ABSTAINING_VERDICTS = {
    "insufficient_information",
    "contraindicated",
    "evidence_unavailable",
}


def _result_abstains(result_json: dict | None) -> tuple[bool, str]:
    """True + reason if the generated result is a fail-closed abstention (no plan
    body was produced). Such a result has no phases to deliver and MUST NOT be
    approved — the doctor resolves the abstention_reason first."""
    if not isinstance(result_json, dict):
        return False, ""
    verdict = str(result_json.get("verdict", "")).strip().lower()
    if verdict in _ABSTAINING_VERDICTS:
        return True, str(result_json.get("abstention_reason", "")).strip()
    return False, ""


class ApprovePlanUseCase:
    def __init__(self, job_repo: JobRepo, plan_writer: PlanWriter):
        self._jobs = job_repo
        self._writer = plan_writer

    async def execute(self, decision: ApprovalDecision) -> PlanApproval:
        if decision.kind is not ApprovalDecisionKind.APPROVE:
            raise ValueError("ApprovePlanUseCase requires kind=APPROVE")

        job = await self._jobs.get(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
        )
        if job is None:
            raise JobNotFound(f"Job {decision.job_id} not found")
        if job.status != "succeeded":
            raise JobNotApprovable(
                f"Job is in '{job.status}' state — only 'succeeded' jobs can be approved"
            )
        if not job.result:
            raise JobNotApprovable("Job has no generated plan")

        # ── Fail-closed gate: an abstaining result has NO plan body ──
        # insufficient_information / contraindicated / evidence_unavailable mean the
        # generator deliberately withheld a plan. There is nothing to approve or
        # deliver; the doctor must resolve the abstention_reason (obtain the missing
        # data, get clearance) and re-generate. Blocking here prevents an empty
        # "plan" from ever reaching a patient.
        abstains, reason = _result_abstains(job.result)
        if abstains:
            raise JobNotApprovable(
                "This case was returned as fail-closed (no rehabilitation plan was "
                "generated) and cannot be approved. Resolve the reason and generate "
                "again. Reason: " + (reason or "not specified")
            )

        # ── Warn-Never-Block CRITICAL acknowledgment gate (mobile/web parity) ──
        # A "Block"-severity drug interaction in THIS patient's generated plan is
        # a CRITICAL per-patient condition (NOT a hardcoded rule). The doctor must
        # explicitly acknowledge it before approval. This mirrors the legacy
        # /plan-approval gate so both clients enforce the same approval-time
        # safety. It never permanently blocks — acknowledging proceeds.
        if _result_has_block_ddi(job.result) and not decision.critical_safety_acknowledged:
            raise CriticalAckRequired(
                "This plan contains a CRITICAL safety condition (an emergency-level "
                "drug interaction for this patient). The approving doctor must review "
                "and acknowledge it before approval. Approval proceeds once acknowledged."
            )

        # Atomically claim the job before writing. The status check above is a
        # fast-fail; this guarded UPDATE is the real concurrency gate. Without
        # it two simultaneous approvals both pass the check and each runs
        # write_approved_plan, leaving the patient with two is_current_version
        # plans plus duplicate daily_tasks / medication_schedules. Only the
        # caller that flips succeeded -> approving proceeds.
        claimed = await self._jobs.claim_for_approval(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
        )
        if not claimed:
            raise JobNotApprovable(
                "Job is already being approved or is no longer in 'succeeded' state"
            )

        try:
            write = await self._writer.write_approved_plan(
                job_id=decision.job_id,
                doctor_id=decision.doctor_id,
                clinic_id=decision.clinic_id,
                result_json=job.result,
                approved_by_id=decision.doctor_id,
            )
        except Exception:
            # Release the claim so the doctor can retry rather than being stuck
            # in the transient 'approving' state.
            await self._jobs.release_approval_claim(
                job_id=decision.job_id,
                doctor_id=decision.doctor_id,
                clinic_id=decision.clinic_id,
            )
            raise

        await self._jobs.mark_approved(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
            plan_id=write.plan_id,
        )
        approved_at = datetime.now(timezone.utc)
        logger.info(
            "[STAGE_K] approve job=%s plan=%s patient=%s phases=%d exercises=%d meds=%d daily=%d",
            decision.job_id,
            write.plan_id,
            write.patient_id,
            write.phase_count,
            write.exercise_count,
            write.medication_count,
            write.daily_task_count,
        )
        return PlanApproval(
            job_id=decision.job_id,
            plan_id=write.plan_id,
            patient_id=write.patient_id,
            status="approved",
            approved_at=approved_at,
            approved_by_id=decision.doctor_id,
        )


class RejectPlanUseCase:
    def __init__(self, job_repo: JobRepo):
        self._jobs = job_repo

    async def execute(self, decision: ApprovalDecision) -> PlanRejection:
        if decision.kind is not ApprovalDecisionKind.REJECT:
            raise ValueError("RejectPlanUseCase requires kind=REJECT")

        job = await self._jobs.get(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
        )
        if job is None:
            raise JobNotFound(f"Job {decision.job_id} not found")
        if job.status not in {"succeeded", "running"}:
            raise JobNotApprovable(f"Job is in '{job.status}' state — cannot reject")

        await self._jobs.mark_rejected(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
            rejection_reason=decision.note,
        )
        return PlanRejection(
            job_id=decision.job_id,
            status="rejected",
            rejected_at=datetime.now(timezone.utc),
            rejection_reason=decision.note,
        )


class EditPlanUseCase:
    def __init__(self, job_repo: JobRepo):
        self._jobs = job_repo

    async def execute(self, decision: ApprovalDecision) -> PlanEditOutcome:
        if decision.kind is not ApprovalDecisionKind.EDIT:
            raise ValueError("EditPlanUseCase requires kind=EDIT")
        if decision.edited_result is None:
            raise ValueError("Edit decision missing edited_result payload")

        job = await self._jobs.get(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
        )
        if job is None:
            raise JobNotFound(f"Job {decision.job_id} not found")
        if job.status != "succeeded":
            raise JobNotApprovable(f"Job is in '{job.status}' state — cannot edit")

        try:
            GeneratedPlan.model_validate(decision.edited_result)
        except ValidationError as exc:
            raise JobNotApprovable(
                f"Edited plan failed schema validation: {exc.errors()[0].get('msg')}"
            ) from exc

        edit_count = await self._jobs.update_result(
            job_id=decision.job_id,
            doctor_id=decision.doctor_id,
            clinic_id=decision.clinic_id,
            result=decision.edited_result,
        )
        if edit_count < 0:
            raise JobNotFound(f"Job {decision.job_id} not found on update")
        return PlanEditOutcome(
            job_id=decision.job_id,
            edit_count=edit_count,
            status="succeeded",
        )


class DeliverPlanUseCase:
    def __init__(self, plan_writer: PlanWriter):
        self._writer = plan_writer

    async def execute(self, request: DeliveryRequest) -> DeliveryOutcome:
        # Authz before publication: the plan must exist in the clinic and
        # belong to the requesting doctor. Owner mismatch -> 403 (not 404) so
        # the doctor cannot publish a colleague's draft; missing -> 404;
        # wrong state -> 409. Keeping the decision here keeps the use case the
        # single authority on the deliver gate.
        ownership = await self._writer.get_plan_ownership(
            plan_id=request.plan_id,
            clinic_id=request.clinic_id,
        )
        if ownership is None:
            raise PlanNotFound(f"Plan {request.plan_id} not found")
        owner_doctor_id, status = ownership
        if owner_doctor_id != request.doctor_id:
            raise TenancyViolation("Plan belongs to a different doctor — cannot deliver")
        if status != "approved":
            raise PlanNotDeliverable(
                f"Plan is in '{status}' state — only 'approved' plans can be delivered"
            )

        try:
            patient_id, is_revision = await self._writer.mark_delivered(
                plan_id=request.plan_id,
                clinic_id=request.clinic_id,
                doctor_id=request.doctor_id,
            )
        except ValueError as exc:
            raise PlanNotDeliverable(str(exc)) from exc

        try:
            block_drugs = await self._writer.block_drug_titles(
                plan_id=request.plan_id, clinic_id=request.clinic_id
            )
        except Exception:
            logger.exception(
                "[DeliverPlanUseCase] block_drug_titles lookup failed plan=%s",
                request.plan_id,
            )
            block_drugs = []

        # Notification dispatch is a presentation-layer concern (needs a DB
        # session); the route fires plan_delivered + drug_interaction_warning
        # from the outcome. Keeping it out of here preserves application-layer
        # purity (no app.db import).
        return DeliveryOutcome(
            plan_id=request.plan_id,
            patient_id=patient_id,
            delivered_at=datetime.now(timezone.utc),
            visibility_max_phase=1,
            block_drug_titles=tuple(block_drugs),
            is_revision=is_revision,
        )
