"""Presentation-layer DTOs for plan_v2 endpoints.

Kept separate from domain entities so the public API contract can evolve
without touching the use case.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GeneratePlanRequestDTO(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "patient_id": 42,
                "patient_gender": "female",
                "case_history_text": (
                    "72-year-old female, right total knee arthroplasty 5 days ago. "
                    "Weight-bearing as tolerated. Pain 4/10 at rest. ROM 0-70 degrees."
                ),
                "output_language": "uzbek",
                "clinician_override": False,
            }
        },
    )

    patient_id: int | None = None
    patient_gender: Literal["male", "female"] | None = None
    # Minimal pregnancy gate (S2): explicit pregnancy status. True → escalates
    # (drug/exercise pregnancy-contraindications need doctor review). None for a
    # female patient surfaces an "unconfirmed" reminder. Full DB-persisted
    # intake lands in S4.
    is_pregnant: bool | None = None
    case_history_text: str = Field(min_length=20, max_length=50000)
    document_id: int | None = None
    output_language: (
        Literal[
            "english",
            "russian",
            "uzbek",
            "uzbek_cyrillic",
            "en",
            "ru",
            "uz",
            "uz-cyrl",
        ]
        | None
    ) = None
    clinician_override: bool = False


class EscalationReason(BaseModel):
    """Structured, PHI-free escalation reason. The frontend renders `code` via
    its own trilingual i18n table; `count`/`total` feed interpolation. `verdict`
    names the abstaining verdict for the `plan_withheld` reason (contraindicated /
    insufficient_information / evidence_unavailable)."""

    code: str
    count: int | None = None
    total: int | None = None
    verdict: str | None = None


class JobEnqueuedResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "1eff40b9fc644b64b3709c125ebc44ab",
                "status": "queued",
            }
        }
    )

    job_id: str
    status: Literal["queued"]


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "1eff40b9fc644b64b3709c125ebc44ab",
                "status": "succeeded",
                "progress_step": "done",
                "result": {
                    "plan_metadata": {"diagnosis_short": "Post-TKR rehab"},
                    "phases": [],
                },
                "error_code": None,
                "error_message": None,
                "created_at": "2026-05-05T12:00:00+00:00",
                "started_at": "2026-05-05T12:00:01+00:00",
                "completed_at": "2026-05-05T12:00:42+00:00",
            }
        }
    )

    job_id: str
    status: Literal[
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "approving",
        "approved",
        "rejected",
    ]
    progress_step: str
    escalation_level: str | None = None
    # Union tolerates legacy string rows stored before the structured contract
    # (pending-plans inbox can surface old jobs); new jobs are EscalationReason.
    escalation_reasons: list[EscalationReason | str] = Field(default_factory=list)
    # Full verdict vocabulary — an emergency fail-closed abstention
    # (contraindicated / insufficient_information / evidence_unavailable) must
    # reach the doctor's job status + inbox badge UNCHANGED. Narrowing this to
    # proceed|caution forced the route to collapse abstentions to "caution",
    # silently downgrading an emergency stop to a mild caution chip.
    verdict: (
        Literal[
            "proceed",
            "caution",
            "insufficient_information",
            "contraindicated",
            "evidence_unavailable",
        ]
        | None
    ) = None
    is_actionable: bool | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


# ──────────────────────────────────────────────────────────────────────────
# Approval / delivery DTOs (Stage K)
# ──────────────────────────────────────────────────────────────────────────


class ApproveRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str | None = Field(default=None, max_length=2000)
    # Doctor's explicit acknowledgment of a CRITICAL per-patient safety condition
    # (e.g. a "Block"-severity drug interaction). Required to approve such a plan;
    # never permanently blocks (acknowledging proceeds). Mobile/web parity.
    critical_safety_acknowledged: bool = Field(default=False)


class RejectRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=500)


class EditRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edited_result: dict[str, Any]


class ApproveResponse(BaseModel):
    job_id: str
    plan_id: int
    patient_id: int
    status: Literal["approved"]
    approved_at: str


class RejectResponse(BaseModel):
    job_id: str
    status: Literal["rejected"]
    rejected_at: str
    rejection_reason: str | None = None


class EditResponse(BaseModel):
    job_id: str
    edit_count: int
    status: Literal["succeeded"]


class DeliverResponse(BaseModel):
    plan_id: int
    patient_id: int
    delivered_at: str
    visibility_max_phase: int


class JobListItemDTO(BaseModel):
    """One row in the doctor's pending-plans inbox."""

    job_id: str
    patient_id: int | None = None
    status: str
    created_at: str
    verdict: str | None = None
    escalation_level: str | None = None
    plan_id: int | None = None


class JobListResponse(BaseModel):
    total: int
    items: list[JobListItemDTO]
