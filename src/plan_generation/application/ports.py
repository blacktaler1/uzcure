"""Application ports — interfaces the use case depends on.

Following Dependency Inversion: the use case talks to these Protocols, not to
SQLAlchemy / Anthropic SDK directly. Concrete implementations live in
`infrastructure/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.features.plan_generation.domain import (
    GeneratedPlan,
    PlanLanguage,
)


# ──────────────────────────────────────────────────────────────────────────
# Patient
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatientSnapshot:
    """De-identified patient context the use case needs. PHI decrypted at boundary."""

    patient_id: int
    display_name: str  # decrypted, used only inside generation, never logged
    primary_language: PlanLanguage | None
    gender: str | None
    age_years: int | None
    diagnosis: str | None
    allergies: list[str]
    comorbidities: list[str]
    known_medications: list[str]
    # Clinician-entered ICD-10 from the baseline assessment, when one was
    # recorded. Read by the evidence path (use_case.py:661) to activate
    # PubMedService's MeSH-qualified query instead of the free-text fallback.
    # None means "not recorded" — never a guess: assigning a code from free-text
    # diagnosis is clinical coding, and a wrong code silently redirects the whole
    # literature search.
    icd10_code: str | None = None


@runtime_checkable
class PatientRepo(Protocol):
    async def get_for_generation(
        self, *, patient_id: int, clinic_id: int
    ) -> PatientSnapshot | None: ...

    async def doctor_has_patient(
        self, *, doctor_id: int, patient_id: int, clinic_id: int
    ) -> bool: ...

    async def can_add_new_plan(self, *, patient_id: int) -> bool: ...

    async def has_ai_consent(self, *, patient_id: int) -> bool: ...


# ──────────────────────────────────────────────────────────────────────────
# Exercise registry
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegistryExercise:
    exercise_id: str
    canonical_name: str
    display_name_uz: str | None
    display_name_ru: str | None
    display_name_en: str | None
    specialty: str
    body_region: str | None
    category: str | None
    default_sets: str | None
    default_reps: str | None
    contraindications: list[str]
    has_verified_video: bool
    video_url: str | None
    # Envelope used by post-LLM safety check. Schema:
    #   {sets_min, sets_max, reps_min, reps_max, duration_min, duration_max}
    # Sentinel reps_min=reps_max=1 marks non-rep-based exercise (duration-based
    # when duration_max present, education/instructional otherwise).
    dosage_ranges: dict[str, Any] | None = None
    # Registry safety text, surfaced to the reviewing doctor when the model left
    # the delivered exercise's own field blank (see use_case._resolve_exercises).
    # The safety scan consumes `contraindications`; `precautions` is display-only.
    precautions: list[str] = field(default_factory=list)
    # Registry evidence provenance — the guideline/standard this exercise traces to
    # (evidence_source, e.g. "ESSKA 2016") and its GRADE (evidence_level A/B/C).
    # Every registry exercise carries these (100% populated); they are the plan's
    # "cited" grounding for registry-backed exercises (which have no PMIDs), and are
    # surfaced onto the delivered exercise + recorded in the evidence ledger.
    evidence_source: str | None = None
    evidence_level: str | None = None


@dataclass(frozen=True)
class AiSuggestedExerciseLog:
    """One AI-invented exercise queued for admin review.

    Batched so the post-LLM resolver can flush every invented exercise from a
    plan in a SINGLE write, instead of one awaited commit per exercise (which
    turned N invented exercises into N separate db round-trips)."""

    name: str
    condition: str
    doctor_id: int
    clinic_id: int
    plan_job_id: str


@runtime_checkable
class ExerciseRepo(Protocol):
    async def pool_for(
        self,
        *,
        specialties: list[str],
        condition_keywords: list[str],
        limit: int = 50,
    ) -> list[RegistryExercise]: ...

    async def get(self, *, exercise_id: str) -> RegistryExercise | None: ...

    async def log_ai_suggested(
        self,
        *,
        name: str,
        condition: str,
        doctor_id: int,
        clinic_id: int,
        plan_job_id: str,
    ) -> None:
        """Queue a single AI-invented exercise for admin review (compatibility
        wrapper over ``log_ai_suggested_many``; no automatic registry add)."""

    async def log_ai_suggested_many(
        self,
        *,
        logs: list[AiSuggestedExerciseLog],
    ) -> None:
        """Persist a batch of AI-invented exercises for admin review in ONE write.
        An empty list is a no-op."""


# ──────────────────────────────────────────────────────────────────────────
# Drug oracle
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# LLM provider
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMRequest:
    system: list[dict[str, Any]]  # cache-controlled blocks
    user_message: str
    tool_schema: dict[str, Any]
    tool_name: str
    max_output_tokens: int = 8192
    temperature: float = 0.0
    # When True, the provider MUST force the tool call (tool_choice=tool) and, for
    # Anthropic, disable extended thinking (the API rejects a forced tool while
    # thinking is on). Used by the use case to deterministically retry after a
    # thinking-mode response skipped the tool. No-op for providers that already
    # force the tool. See ToolCallMissingError.
    force_tool: bool = False
    # Per-request extended-thinking effort ("low" | "medium" | "high"), chosen by
    # the use case from case complexity so simple cases run fast and complex
    # multi-morbid cases keep reasoning depth. None → the provider's configured
    # default (budget-derived tier).
    thinking_effort: str | None = None
    # Per-request wall-clock deadline (seconds) for THIS provider call. None → the
    # provider's configured default (PLAN_V2_PROVIDER_TIMEOUT_SECONDS). The use
    # case sets a SHORT grace on the thinking/auto-tool call so a slow or stalled
    # extended-thinking generation is cut over early to a fast forced-tool call,
    # and the full provider timeout on the forced-tool recovery. Enforced by the
    # provider via asyncio.wait_for.
    deadline_s: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    tool_input: dict[str, Any]  # parsed tool_use input — the structured plan
    raw_text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    latency_ms: int
    model: str
    estimated_cost_usd: float


# ──────────────────────────────────────────────────────────────────────────
# Evidence retrieval (V4 hybrid layer)
#
# When the internal registry does not cover a condition, the use case retrieves
# real, graded evidence from external sources (PubMed, clinical-guideline vector
# store) to GROUND the generated plan. Every article carries a real PMID + URL;
# no fabricated citations. Sources are stored internally for audit and are NEVER
# surfaced to the doctor or patient — they are a quality/safety mechanism only.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvidenceArticle:
    """A single piece of retrieved external evidence. Decoupled from any concrete
    retriever (PubMed ArticleResult, vector chunk, ...) so the application layer
    never depends on infrastructure. `pmid` and `url` are guaranteed real by the
    retriever; they are validated again before the plan is saved (hard PMID gate)."""

    pmid: str
    title: str
    journal: str
    year: str
    url: str
    # Evidence quality signals used to rank/grade grounding strength.
    is_systematic_review: bool = False
    is_rct: bool = False
    is_meta_analysis: bool = False
    is_guideline: bool = False
    source: str = "pubmed"  # which retriever produced it (pubmed | guideline_vector | ...)


@dataclass(frozen=True)
class EvidenceQuery:
    """What to retrieve evidence for. `domain` lets one retriever serve every plan
    element — exercises now; drugs / interactions / supplements / diet next — each
    mapping to the right query construction inside the adapter."""

    condition: str
    domain: str = "exercise"  # exercise | drug_interaction | supplement | diet
    icd10_code: str | None = None
    max_results: int = 10


@runtime_checkable
class EvidenceRetriever(Protocol):
    async def retrieve(self, query: EvidenceQuery) -> list[EvidenceArticle]:
        """Return graded external evidence for the query, or an empty list if none
        found. MUST NOT fabricate: every returned article has a real PMID + URL.
        Implementations chain sources (PubMed -> guideline vector -> ...) and may
        raise on total source failure so the caller can fail closed."""


@runtime_checkable
class PmidVerifier(Protocol):
    async def verify(self, pmids: list[str]) -> set[str]:
        """Confirm which of the given PMIDs actually exist at NCBI right now.

        Returns the SUBSET of `pmids` that resolve to a real PubMed record. PMIDs
        not in the returned set do not exist (fabricated / dead) and must be
        stripped by the caller. An empty input returns an empty set.

        MUST raise (not return a partial/empty set) when NCBI itself is
        unreachable, so the caller can distinguish 'this PMID is not real' from
        'we could not check' and fail closed on the latter."""


@runtime_checkable
class LLMProvider(Protocol):
    async def call_with_tool(self, req: LLMRequest) -> LLMResponse: ...


# ──────────────────────────────────────────────────────────────────────────
# Job persistence
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    status: str
    progress_step: str
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class JobListItem:
    """Light row for the doctor's pending-plans inbox (lost-plan recovery)."""

    job_id: str
    patient_id: int | None
    status: str
    created_at: str
    verdict: str | None
    escalation_level: str | None
    plan_id: int | None


@runtime_checkable
class JobRepo(Protocol):
    async def create(
        self,
        *,
        doctor_id: int,
        clinic_id: int,
        request_payload: dict[str, Any],
        request_hash: str,
    ) -> JobRecord: ...

    async def get(self, *, job_id: str, doctor_id: int, clinic_id: int) -> JobRecord | None: ...

    async def list_for_doctor(
        self,
        *,
        doctor_id: int,
        clinic_id: int,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobListItem], int]: ...

    async def mark_running(self, *, job_id: str, step: str = "started") -> None: ...

    async def update_progress(self, *, job_id: str, step: str) -> None: ...

    async def mark_succeeded(self, *, job_id: str, result: dict[str, Any]) -> None: ...

    async def mark_failed(self, *, job_id: str, code: str, message: str) -> None: ...

    async def cancel(self, *, job_id: str, doctor_id: int, clinic_id: int) -> bool: ...

    async def claim_for_approval(self, *, job_id: str, doctor_id: int, clinic_id: int) -> bool:
        """Atomically transition succeeded -> approving. Returns True only for
        the single caller that won the claim; concurrent approvers get False."""
        ...

    async def release_approval_claim(self, *, job_id: str, doctor_id: int, clinic_id: int) -> None:
        """Revert approving -> succeeded when the post-claim write fails, so the
        doctor can retry instead of being stuck in a transient state."""
        ...

    async def mark_approved(
        self,
        *,
        job_id: str,
        doctor_id: int,
        clinic_id: int,
        plan_id: int,
    ) -> None: ...

    async def mark_rejected(
        self,
        *,
        job_id: str,
        doctor_id: int,
        clinic_id: int,
        rejection_reason: str | None,
    ) -> None: ...

    async def update_result(
        self,
        *,
        job_id: str,
        doctor_id: int,
        clinic_id: int,
        result: dict[str, Any],
    ) -> int:
        """Replace result_json with edited content. Returns new edit_count."""


# ──────────────────────────────────────────────────────────────────────────
# Cooldown guard (prevent retry storm after failures)
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class CooldownGuard(Protocol):
    async def is_cooling_down(self, *, key: str) -> bool: ...

    async def remaining_seconds(self, *, key: str) -> int: ...

    async def record_failure(self, *, key: str) -> None: ...


# ──────────────────────────────────────────────────────────────────────────
# Use case result
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationOutcome:
    plan: GeneratedPlan
    llm_input_tokens: int
    llm_output_tokens: int
    llm_cache_read_tokens: int
    llm_latency_ms: int
    estimated_cost_usd: float
    ai_suggested_exercise_count: int
    safety_meta: dict[str, Any] | None = None
    llm_model: str = ""
    # ── Verdict + escalation (read by the worker for status + the Tier-1 push) ──
    # `verdict`/`is_actionable` mirror the plan's triage tag.
    # `requires_doctor_review` is ALWAYS True — every AI plan is doctor-reviewed
    # before a patient sees it (invariant #1). It is NEVER an auto-deliver flag.
    # `escalation_level` is the content-driven signal ("routine" | "escalated")
    # that drives the critical "review now" push; `escalation_reasons` names why.
    verdict: str = "proceed"
    is_actionable: bool = True
    requires_doctor_review: bool = True
    escalation_level: str = "routine"
    # Structured, PHI-free, language-agnostic: [{"code": str, "count": int|None,
    # "total": int|None}]. The frontend renders each code via its own i18n table
    # (trilingual); the worker push maps codes to a short label. Never prose.
    escalation_reasons: list[dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Plan writer (Stage K — approve copies job result_json into rehab_plans tree)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanWriteResult:
    plan_id: int
    patient_id: int
    phase_count: int
    exercise_count: int
    medication_count: int
    daily_task_count: int


@runtime_checkable
class PlanWriter(Protocol):
    async def write_approved_plan(
        self,
        *,
        job_id: str,
        doctor_id: int,
        clinic_id: int,
        result_json: dict[str, Any],
        approved_by_id: int,
    ) -> PlanWriteResult:
        """Atomic copy: result_json -> rehab_plans + rehab_phases +
        plan_exercises + medication_schedules + plan_visibility_rules +
        daily_tasks (phase 1)."""

    async def mark_delivered(
        self,
        *,
        plan_id: int,
        clinic_id: int,
        doctor_id: int,
    ) -> tuple[int, bool]:
        """Publish gate (Design 1). Supersede the prior current plan, activate
        this plan's meds, generate phase-1 daily_tasks, set status='delivered'
        + is_current_version=true + delivered_at, write the delivery
        confirmation + an APPROVE DoctorSignoff. Returns (patient_id,
        is_revision) where is_revision is True when a prior current plan was
        superseded (vs a first delivery)."""

    async def get_plan_ownership(
        self,
        *,
        plan_id: int,
        clinic_id: int,
    ) -> tuple[int | None, str] | None:
        """Return (doctor_id, status) for an in-clinic plan, or None if the plan
        does not exist in the clinic. Used by the use case to separate authz
        (owner mismatch) from deliverability (status)."""

    async def block_drug_titles(
        self,
        *,
        plan_id: int,
        clinic_id: int,
    ) -> list[str]:
        """Return Block-severity drug-interaction titles from the delivered
        plan body (rehab_plans.ai_raw_response.drug_interactions). Empty list
        when none. Used to push a patient-facing safety warning on delivery."""


__all__ = [
    "CooldownGuard",
    "ExerciseRepo",
    "GenerationOutcome",
    "JobListItem",
    "JobRecord",
    "JobRepo",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "PatientRepo",
    "PatientSnapshot",
    "PlanWriteResult",
    "PlanWriter",
    "RegistryExercise",
]
