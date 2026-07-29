from __future__ import annotations


class DomainError(Exception):
    """Base for domain-layer exceptions. Presentation layer maps these to HTTP."""

    code: str = "DOMAIN_ERROR"

    def __init__(self, message: str = "", *, code: str | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class TenancyViolation(DomainError):
    code = "TENANCY_VIOLATION"


class PatientNotFound(DomainError):
    code = "PATIENT_NOT_FOUND"


class ConsentMissingError(DomainError):
    code = "AI_CONSENT_MISSING"


class PlanLimitExceeded(DomainError):
    code = "PLAN_LIMIT_EXCEEDED"


class SafetyVerdictBlocked(DomainError):
    """Hard safety block — patient cannot receive an active rehab plan."""

    code = "SAFETY_BLOCKED"


class LateralityMismatch(DomainError):
    """Fail-closed: the plan's stated diagnosis names the OPPOSITE operated side
    from the case — a never-event (contralateral procedure statement).

    Unlike the codebase's default Warn-Never-Block posture, a wrong-side plan is
    blocked, not escalated: it must never reach a doctor's approve screen as a mere
    'doctor review' flag, because a rubber-stamped contralateral plan is a
    catastrophic patient-safety failure. Deterministic detector
    (``check_diagnosis_laterality_conflict``); the message is PHI-free."""

    code = "LATERALITY_MISMATCH"


class SchemaValidationError(DomainError):
    """LLM output failed strict schema validation after allowed retries."""

    code = "SCHEMA_VALIDATION_FAILED"


class LanguageComplianceError(DomainError):
    """LLM output emitted in wrong language after allowed retries."""

    code = "LANGUAGE_MISMATCH"


class LLMUpstreamError(DomainError):
    code = "LLM_UPSTREAM_ERROR"


class ToolCallMissingError(DomainError):
    """The LLM returned a response with no tool_use block for the required tool.

    This is the extended-thinking failure mode: when thinking is enabled the
    Anthropic API forbids forcing tool_choice (`tool`/`any` are rejected), so the
    provider must use `tool_choice: auto` — which lets the model answer in prose
    and skip the `submit_plan` tool entirely. That yields no structured plan.

    It is raised as a distinct error (not a generic upstream failure) so the use
    case can recover deterministically: retry once with thinking disabled and the
    tool forced (`LLMRequest.force_tool=True`), which the API permits and which
    guarantees the tool is called. Non-Anthropic providers already force the tool
    and never raise this.
    """

    code = "TOOL_CALL_MISSING"


class EvidenceVerificationFailed(DomainError):
    """Retrieved-evidence PMID verification failed — fail-closed by policy (V4).

    Raised when a registry-miss plan was grounded in external evidence but that
    evidence cannot be confirmed real against live NCBI before save: either NCBI
    is unreachable (cannot confirm), or an exercise that claimed evidence is left
    with no PMID that resolves at NCBI. A plan must never save citing evidence
    that is not verifiably real, so generation fails rather than emit it."""

    code = "EVIDENCE_VERIFICATION_FAILED"


class LedgerWriteError(DomainError):
    """The tamper-evident evidence-ledger row could not be durably written for a
    patient-linked plan — fail-closed by policy.

    Every patient-linked generated plan MUST have a signed, append-only audit row
    (integrity hash + grounding evidence + drug-safety + LLM provenance) before it
    can be marked `succeeded` and offered for delivery. If that row cannot be
    committed, the job fails rather than surface an unauditable plan; the doctor
    regenerates. (Case-history-only runs with no patient row legitimately skip the
    ledger and are unaffected.)"""

    code = "EVIDENCE_LEDGER_WRITE_FAILED"


class GenerationCooldownActive(DomainError):
    code = "GENERATION_COOLDOWN"

    def __init__(self, message: str, retry_after_seconds: int):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class JobNotFound(DomainError):
    code = "JOB_NOT_FOUND"


class JobNotApprovable(DomainError):
    """Job is not in 'succeeded' state or has already been approved/rejected."""

    code = "JOB_NOT_APPROVABLE"


class CriticalAckRequired(DomainError):
    """Plan has a CRITICAL per-patient safety condition (a "Block"-severity drug
    interaction) that the approving doctor must explicitly acknowledge before
    approval. Warn-Never-Block: acknowledging proceeds — never a permanent block.
    Parity with the legacy /plan-approval path's critical-acknowledgment gate."""

    code = "CRITICAL_ACK_REQUIRED"


class PlanNotFound(DomainError):
    code = "PLAN_NOT_FOUND"


class PlanNotDeliverable(DomainError):
    """Plan is not approved or already delivered."""

    code = "PLAN_NOT_DELIVERABLE"
