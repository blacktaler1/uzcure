"""Approval / delivery domain entities (Stage K).

Pure data + invariants. The doctor reviews the generated plan stored in
``plan_generation_jobs.result_json``, then issues an ``ApprovalDecision``
that the use case applies — copying the plan into the canonical
``rehab_plans`` table tree and triggering patient-side delivery.

Internal-only fields are stripped on the API boundary by ``to_api_dict``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApprovalDecisionKind(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"


class ApprovalDecision(BaseModel):
    """Doctor's decision on a generated plan.

    For ``EDIT``, ``edited_result`` is the updated plan JSON. The use case
    persists it back to ``plan_generation_jobs.result_json`` and bumps
    ``edit_count``. Approve still required afterwards.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1, max_length=36)
    doctor_id: int = Field(ge=1)
    clinic_id: int = Field(ge=1)
    kind: ApprovalDecisionKind
    note: str | None = Field(default=None, max_length=2000)
    edited_result: dict[str, Any] | None = None
    # Warn-Never-Block: when the plan carries a CRITICAL per-patient condition
    # (a "Block"-severity drug interaction), the doctor must set this True to
    # acknowledge before approval. Parity with the legacy /plan-approval gate.
    critical_safety_acknowledged: bool = False


class PlanApproval(BaseModel):
    """Outcome of an approve action — what the doctor sees in response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    plan_id: int
    patient_id: int
    status: Literal["approved"]
    approved_at: datetime
    approved_by_id: int


class PlanRejection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    status: Literal["rejected"]
    rejected_at: datetime
    rejection_reason: str | None


class PlanEditOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    edit_count: int
    status: Literal["succeeded"]


class DeliveryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: int = Field(ge=1)
    doctor_id: int = Field(ge=1)
    clinic_id: int = Field(ge=1)


class DeliveryOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: int
    patient_id: int
    delivered_at: datetime
    visibility_max_phase: int
    block_drug_titles: tuple[str, ...] = ()
    # True when this delivery superseded a prior current plan (a revision) —
    # the route emits plan_revised instead of plan_delivered.
    is_revision: bool = False
