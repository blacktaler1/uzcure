"""Plan generation endpoints — job-queue pattern.

POST /api/plans/jobs        — enqueue, return 202 + job_id
GET  /api/plans/jobs/{id}   — poll status / result
DELETE /api/plans/jobs/{id} — cancel queued/running

Router stays thin: validate DTO, build command, persist job, schedule worker.
All clinical logic lives in the use case.

NOTE: deliberately no `from __future__ import annotations` — FastAPI relies
on resolved type hints for Body / Header inference, and stringified hints
trigger a Pydantic v2 ForwardRef rebuild error on the body parameter.
"""

import hashlib
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_clinic_id, require_verified_doctor
from app.core.db_thread import db_in_thread
from app.core.rate_limit import AI_GENERATE_PLAN_LIMIT, limiter
from app.core.uzbek_translit import cyrillic_to_latin, is_uzbek_cyrillic
from app.db.database import get_db
from app.features.plan_generation.api.schemas import (
    ApproveRequestDTO,
    ApproveResponse,
    DeliverResponse,
    EditRequestDTO,
    EditResponse,
    GeneratePlanRequestDTO,
    JobEnqueuedResponse,
    JobListItemDTO,
    JobListResponse,
    JobStatusResponse,
    RejectRequestDTO,
    RejectResponse,
)
from app.features.plan_generation.application.approval_use_case import (
    ApprovePlanUseCase,
    DeliverPlanUseCase,
    EditPlanUseCase,
    RejectPlanUseCase,
)
from app.features.plan_generation.domain import (
    ApprovalDecision,
    ApprovalDecisionKind,
    DeliveryRequest,
    PlanLanguage,
    Verdict,
)
from app.features.plan_generation.infrastructure.job_repo import SqlJobRepo
from app.features.plan_generation.infrastructure.plan_writer import SqlPlanWriter
from app.features.plan_generation.jobs.worker import schedule_job
from app.services.cache_service import get_cached, set_cached

logger = logging.getLogger(__name__)
router = APIRouter()

# Idempotency-Key cache TTL for approve / reject / edit / deliver actions.
# A safe value relative to typical client retry windows (30-60s) while short
# enough that stale state can't accidentally replay across user sessions.
_IDEM_ACTION_TTL_SECONDS = 300

# Every verdict the current domain vocabulary can emit, including the fail-closed
# abstentions. These pass through the job status response UNCHANGED.
_VALID_JOB_VERDICTS = frozenset(v.value for v in Verdict)


def _normalize_job_verdict(raw: str | None) -> str | None:
    """Map a stored verdict onto the job status response vocabulary.

    Every current verdict — including the fail-closed abstentions
    (``contraindicated`` / ``insufficient_information`` / ``evidence_unavailable``)
    — passes through UNCHANGED, so an emergency abstention is never downgraded to
    ``caution`` in the job status or the inbox verdict badge. Only genuinely
    unknown legacy strings (e.g. ``escalate`` / ``monitoring_only`` from
    pre-vocabulary rows) fall back to ``caution`` so the response model never
    500s when a doctor opens an old job from the inbox.
    """
    if raw is None or raw in _VALID_JOB_VERDICTS:
        return raw
    return "caution"


def _resolve_output_language(
    db: Session, requested: Optional[str], patient_id: Optional[int], clinic_id: int
) -> PlanLanguage:
    """Language the plan is generated in.

    The plan is written FOR the patient, so when the request doesn't pin a
    language we default to the patient's own `primary_language` — not a fixed
    house default and not the doctor's UI language. An explicit request value
    always wins (the doctor can still override per generation in the composer).
    Falls back to `from_str(None)` (UZ) only when there is no patient or the
    patient has no recorded language.
    """
    if requested is not None:
        return PlanLanguage.from_str(requested)
    if patient_id is not None:
        from app.models.patient import Patient

        patient = (
            db.query(Patient)
            .filter(Patient.id == patient_id, Patient.clinic_id == clinic_id)
            .first()
        )
        if patient is not None and getattr(patient, "primary_language", None):
            return PlanLanguage.from_str(patient.primary_language)
    return PlanLanguage.from_str(requested)


def _idem_action_cache_key(action: str, *, user_id: int, scope_id: object, idem_key: str) -> str:
    return f"idem:plan_action:{action}:{user_id}:{scope_id}:{idem_key}"


def _idem_replay(action: str, *, user_id: int, scope_id: object, idem_key: Optional[str]):
    if not idem_key:
        return None
    try:
        return get_cached(
            _idem_action_cache_key(action, user_id=user_id, scope_id=scope_id, idem_key=idem_key)
        )
    except Exception:
        logger.exception("idempotency_replay_lookup_failed action=%s", action)
        return None


def _idem_store(
    action: str, *, user_id: int, scope_id: object, idem_key: Optional[str], payload: dict
) -> None:
    if not idem_key:
        return
    try:
        set_cached(
            _idem_action_cache_key(action, user_id=user_id, scope_id=scope_id, idem_key=idem_key),
            payload,
            ttl=_IDEM_ACTION_TTL_SECONDS,
        )
    except Exception:
        logger.exception("idempotency_replay_store_failed action=%s", action)


@router.post(
    "/jobs",
    response_model=JobEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit(AI_GENERATE_PLAN_LIMIT)
async def enqueue_plan_job(
    request: Request,  # required by SlowAPI rate limiter
    body: GeneratePlanRequestDTO,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> JobEnqueuedResponse:
    lang = _resolve_output_language(db, body.output_language, body.patient_id, clinic_id)

    case_text = body.case_history_text
    if lang is PlanLanguage.UZ and is_uzbek_cyrillic(case_text):
        case_text = cyrillic_to_latin(case_text)

    payload = {
        "doctor_id": current_user.id,
        "clinic_id": clinic_id,
        "patient_id": body.patient_id,
        "patient_gender": body.patient_gender,
        "is_pregnant": body.is_pregnant,
        "case_history_text": case_text,
        "document_id": body.document_id,
        "output_language": lang.value,
        "clinician_override": body.clinician_override,
    }

    request_hash = _hash_payload(payload)
    if idempotency_key:
        # Bind the client-supplied key to the doctor + payload so a replay
        # from a different doctor (or different content) gets a fresh job.
        request_hash = hashlib.sha256(
            f"{idempotency_key}|{current_user.id}|{request_hash}".encode("utf-8")
        ).hexdigest()

    job_repo = SqlJobRepo(db)

    # Idempotency-Key replay: same key + same doctor + same payload returns
    # the existing job_id (queued|running|succeeded) instead of creating a new
    # one. Lookup is best-effort.
    if idempotency_key:
        existing = await _find_job_by_hash(
            db, doctor_id=current_user.id, clinic_id=clinic_id, request_hash=request_hash
        )
        if existing is not None:
            return JobEnqueuedResponse(job_id=existing, status="queued")

    # Server-enforced idempotency (UNCONDITIONAL — not gated on a client header):
    # if an IDENTICAL request is already queued|running for this doctor+clinic,
    # return it rather than starting — and billing — a second full generation.
    # The generate call-site sends no Idempotency-Key, so this is what actually
    # collapses a double click / the frontend's enqueue-retry. Terminal jobs are
    # excluded, so a deliberate re-generation after completion still creates one.
    existing = await _find_inflight_job_by_hash(
        db, doctor_id=current_user.id, clinic_id=clinic_id, request_hash=request_hash
    )
    if existing is not None:
        return JobEnqueuedResponse(job_id=existing, status="queued")

    try:
        record = await job_repo.create(
            doctor_id=current_user.id,
            clinic_id=clinic_id,
            request_payload=payload,
            request_hash=request_hash,
        )
    except IntegrityError:
        # Lost the create race to a concurrent identical POST: the partial
        # UNIQUE index (uq_plan_job_inflight_hash) rejected this insert. Roll
        # back and return the job the winner created — one generation, one bill.
        await db_in_thread(db.rollback)
        existing = await _find_inflight_job_by_hash(
            db, doctor_id=current_user.id, clinic_id=clinic_id, request_hash=request_hash
        )
        if existing is not None:
            return JobEnqueuedResponse(job_id=existing, status="queued")
        raise

    schedule_job(record.job_id, payload)

    return JobEnqueuedResponse(job_id=record.job_id, status="queued")


@router.get("/jobs", response_model=JobListResponse)
async def list_plan_jobs(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> JobListResponse:
    """Doctor's pending-plans inbox. Tenant + doctor scoped; optionally filtered
    by `?status=succeeded` so a generated-but-not-delivered draft can be resumed
    (lost-plan recovery)."""
    job_repo = SqlJobRepo(db)
    items, total = await job_repo.list_for_doctor(
        doctor_id=current_user.id,
        clinic_id=clinic_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    return JobListResponse(
        total=total,
        items=[
            JobListItemDTO(
                job_id=i.job_id,
                patient_id=i.patient_id,
                status=i.status,
                created_at=i.created_at,
                verdict=i.verdict,
                escalation_level=i.escalation_level,
                plan_id=i.plan_id,
            )
            for i in items
        ],
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_plan_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> JobStatusResponse:
    job_repo = SqlJobRepo(db)
    record = await job_repo.get(job_id=job_id, doctor_id=current_user.id, clinic_id=clinic_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Reap-on-read: an in-process job whose worker died between restarts (deploy
    # after --graceful-timeout, worker recycle, dropped task) stays non-terminal
    # with nothing to advance it, and the client polls it until its own ceiling.
    # If it is stuck past the hard ceiling (or long-'queued'), transition it to
    # failed here so the poll returns a real error + retry instead of hanging.
    # Atomic + status-guarded, so it can never race a live worker that just
    # finished (see SqlJobRepo.reap_if_stale).
    if record.status in ("queued", "running"):
        from app.features.plan_generation.settings import get_plan_v2_settings

        reaped = await job_repo.reap_if_stale(
            job_id=job_id,
            doctor_id=current_user.id,
            clinic_id=clinic_id,
            running_ceiling_s=get_plan_v2_settings().PLAN_V2_OUTCOME_TIMEOUT_SECONDS,
        )
        if reaped:
            record = await job_repo.get(
                job_id=job_id, doctor_id=current_user.id, clinic_id=clinic_id
            )
            if record is None:
                raise HTTPException(status_code=404, detail="Job not found")

    result_dict = record.result if isinstance(record.result, dict) else None
    verdict = _normalize_job_verdict(result_dict.get("verdict") if result_dict else None)
    is_actionable = result_dict.get("is_actionable") if result_dict else None
    escalation_level = result_dict.get("escalation_level") if result_dict else None
    escalation_reasons = result_dict.get("escalation_reasons") or [] if result_dict else []
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,  # type: ignore[arg-type]
        progress_step=record.progress_step or "",
        verdict=verdict,  # type: ignore[arg-type]
        is_actionable=is_actionable,
        escalation_level=escalation_level,
        escalation_reasons=escalation_reasons,
        result=result_dict,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


@router.delete("/jobs/{job_id}")
async def cancel_plan_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> JSONResponse:
    job_repo = SqlJobRepo(db)
    cancelled = await job_repo.cancel(job_id=job_id, doctor_id=current_user.id, clinic_id=clinic_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail="Job not found or no longer cancellable",
        )
    return JSONResponse({"job_id": job_id, "status": "cancelled"})


@router.post(
    "/jobs/{job_id}/approve",
    response_model=ApproveResponse,
    status_code=status.HTTP_201_CREATED,
)
async def approve_plan_job(
    job_id: str,
    body: ApproveRequestDTO,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> ApproveResponse:
    cached = _idem_replay(
        "approve", user_id=current_user.id, scope_id=job_id, idem_key=idempotency_key
    )
    if cached:
        return ApproveResponse(**cached)

    use_case = ApprovePlanUseCase(SqlJobRepo(db), SqlPlanWriter(db))
    decision = ApprovalDecision(
        job_id=job_id,
        doctor_id=current_user.id,
        clinic_id=clinic_id,
        kind=ApprovalDecisionKind.APPROVE,
        note=body.note,
        critical_safety_acknowledged=body.critical_safety_acknowledged,
    )
    outcome = await use_case.execute(decision)
    response = ApproveResponse(
        job_id=outcome.job_id,
        plan_id=outcome.plan_id,
        patient_id=outcome.patient_id,
        status="approved",
        approved_at=outcome.approved_at.isoformat(),
    )
    _idem_store(
        "approve",
        user_id=current_user.id,
        scope_id=job_id,
        idem_key=idempotency_key,
        payload=response.model_dump(),
    )
    return response


@router.post("/jobs/{job_id}/reject", response_model=RejectResponse)
async def reject_plan_job(
    job_id: str,
    body: RejectRequestDTO,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> RejectResponse:
    cached = _idem_replay(
        "reject", user_id=current_user.id, scope_id=job_id, idem_key=idempotency_key
    )
    if cached:
        return RejectResponse(**cached)

    use_case = RejectPlanUseCase(SqlJobRepo(db))
    decision = ApprovalDecision(
        job_id=job_id,
        doctor_id=current_user.id,
        clinic_id=clinic_id,
        kind=ApprovalDecisionKind.REJECT,
        note=body.reason,
    )
    outcome = await use_case.execute(decision)
    response = RejectResponse(
        job_id=outcome.job_id,
        status="rejected",
        rejected_at=outcome.rejected_at.isoformat(),
        rejection_reason=outcome.rejection_reason,
    )
    _idem_store(
        "reject",
        user_id=current_user.id,
        scope_id=job_id,
        idem_key=idempotency_key,
        payload=response.model_dump(),
    )
    return response


@router.post("/jobs/{job_id}/edit", response_model=EditResponse)
async def edit_plan_job(
    job_id: str,
    body: EditRequestDTO,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> EditResponse:
    cached = _idem_replay(
        "edit", user_id=current_user.id, scope_id=job_id, idem_key=idempotency_key
    )
    if cached:
        return EditResponse(**cached)

    use_case = EditPlanUseCase(SqlJobRepo(db))
    decision = ApprovalDecision(
        job_id=job_id,
        doctor_id=current_user.id,
        clinic_id=clinic_id,
        kind=ApprovalDecisionKind.EDIT,
        edited_result=body.edited_result,
    )
    outcome = await use_case.execute(decision)
    response = EditResponse(
        job_id=outcome.job_id,
        edit_count=outcome.edit_count,
        status="succeeded",
    )
    _idem_store(
        "edit",
        user_id=current_user.id,
        scope_id=job_id,
        idem_key=idempotency_key,
        payload=response.model_dump(),
    )
    return response


@router.post("/{plan_id}/deliver", response_model=DeliverResponse)
async def deliver_plan(
    plan_id: int,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    current_user=Depends(require_verified_doctor()),
    clinic_id: int = Depends(get_current_clinic_id),
) -> DeliverResponse:
    cached = _idem_replay(
        "deliver", user_id=current_user.id, scope_id=plan_id, idem_key=idempotency_key
    )
    if cached:
        return DeliverResponse(**cached)

    use_case = DeliverPlanUseCase(SqlPlanWriter(db))
    outcome = await use_case.execute(
        DeliveryRequest(
            plan_id=plan_id,
            doctor_id=current_user.id,
            clinic_id=clinic_id,
        )
    )
    response = DeliverResponse(
        plan_id=outcome.plan_id,
        patient_id=outcome.patient_id,
        delivered_at=outcome.delivered_at.isoformat(),
        visibility_max_phase=outcome.visibility_max_phase,
    )

    try:
        from app.services.notification_router import notify

        # First delivery → plan_delivered; superseding an existing current plan
        # → plan_revised. template_key localizes at read (was hardcoded Uzbek).
        _event = "plan_revised" if outcome.is_revision else "plan_delivered"
        notify(
            db,
            _event,
            patient_id=outcome.patient_id,
            template_key=_event,
            template_params={},
            reference_type="rehab_plan",
            reference_id=outcome.plan_id,
        )
        if outcome.block_drug_titles:
            notify(
                db,
                "drug_interaction_warning",
                patient_id=outcome.patient_id,
                template_key="drug_interaction_warning",
                template_params={"drugs": "; ".join(outcome.block_drug_titles[:5])},
                reference_type="rehab_plan",
                reference_id=outcome.plan_id,
            )
    except Exception:
        logger.exception("[deliver_plan] notify failed plan=%s", outcome.plan_id)

    _idem_store(
        "deliver",
        user_id=current_user.id,
        scope_id=plan_id,
        idem_key=idempotency_key,
        payload=response.model_dump(),
    )
    return response


def _hash_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def _find_job_by_hash(
    db: Session, *, doctor_id: int, clinic_id: int, request_hash: str
) -> Optional[str]:
    """Lookup an existing live job by (doctor + clinic + request_hash). Used by
    Idempotency-Key replay handling. Failed/cancelled jobs are excluded so the
    same key under a fresh attempt creates a new job instead of returning a
    dead one.
    """
    return await _find_job_by_hash_in_statuses(
        db,
        doctor_id=doctor_id,
        clinic_id=clinic_id,
        request_hash=request_hash,
        statuses=("queued", "running", "succeeded"),
        log_label="idempotency_key_lookup_failed",
    )


async def _find_inflight_job_by_hash(
    db: Session, *, doctor_id: int, clinic_id: int, request_hash: str
) -> Optional[str]:
    """Lookup an in-flight (queued|running) job with the same request hash.

    Backs the unconditional enqueue dedup: a concurrent identical submit returns
    this job instead of billing a second generation. Deliberately excludes
    terminal statuses (incl. `succeeded`) so a re-generation after completion is
    never blocked — that is the difference from `_find_job_by_hash`.
    """
    return await _find_job_by_hash_in_statuses(
        db,
        doctor_id=doctor_id,
        clinic_id=clinic_id,
        request_hash=request_hash,
        statuses=("queued", "running"),
        log_label="inflight_dedup_lookup_failed",
    )


async def _find_job_by_hash_in_statuses(
    db: Session,
    *,
    doctor_id: int,
    clinic_id: int,
    request_hash: str,
    statuses: tuple[str, ...],
    log_label: str,
) -> Optional[str]:
    from app.core.db_thread import db_in_thread
    from app.models.plan_generation_job import PlanGenerationJob

    def _query() -> Optional[str]:
        row = (
            db.query(PlanGenerationJob.id)
            .filter(
                PlanGenerationJob.doctor_id == doctor_id,
                PlanGenerationJob.clinic_id == clinic_id,
                PlanGenerationJob.request_hash == request_hash,
                PlanGenerationJob.status.in_(list(statuses)),
            )
            .order_by(PlanGenerationJob.created_at.desc())
            .first()
        )
        return row[0] if row else None

    try:
        return await db_in_thread(_query)
    except Exception:
        logger.exception(log_label)
        return None
