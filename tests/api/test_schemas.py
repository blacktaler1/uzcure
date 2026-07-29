"""Plan v2 API DTO tests (Step 81)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.features.plan_generation.api.schemas import (
    GeneratePlanRequestDTO,
    JobEnqueuedResponse,
    JobStatusResponse,
)


# ──────────────────────────────────────────────────────────────────────────
# GeneratePlanRequestDTO
# ──────────────────────────────────────────────────────────────────────────


def test_minimal_request():
    dto = GeneratePlanRequestDTO(case_history_text="x" * 30)
    assert dto.patient_id is None
    assert dto.clinician_override is False


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        GeneratePlanRequestDTO(
            case_history_text="x" * 30, foo="bar"  # type: ignore[call-arg]
        )


def test_case_history_too_short():
    with pytest.raises(ValidationError):
        GeneratePlanRequestDTO(case_history_text="short")


def test_case_history_too_long():
    with pytest.raises(ValidationError):
        GeneratePlanRequestDTO(case_history_text="x" * 50_001)


@pytest.mark.parametrize(
    "lang",
    ["english", "russian", "uzbek", "en", "ru", "uz"],
)
def test_output_language_accepted_aliases(lang):
    dto = GeneratePlanRequestDTO(case_history_text="x" * 30, output_language=lang)
    assert dto.output_language == lang


def test_invalid_output_language_rejected():
    with pytest.raises(ValidationError):
        GeneratePlanRequestDTO(case_history_text="x" * 30, output_language="french")


def test_invalid_gender_rejected():
    with pytest.raises(ValidationError):
        GeneratePlanRequestDTO(
            case_history_text="x" * 30, patient_gender="other"  # type: ignore[arg-type]
        )


# ──────────────────────────────────────────────────────────────────────────
# Response DTOs
# ──────────────────────────────────────────────────────────────────────────


def test_job_enqueued_response_status_locked():
    res = JobEnqueuedResponse(job_id="abc", status="queued")
    assert res.status == "queued"
    with pytest.raises(ValidationError):
        JobEnqueuedResponse(job_id="abc", status="weird")  # type: ignore[arg-type]


def test_job_status_response_full():
    res = JobStatusResponse(
        job_id="abc",
        status="succeeded",
        progress_step="done",
        result={"phases": []},
        error_code=None,
        error_message=None,
        created_at="2026-05-05T00:00:00",
        started_at="2026-05-05T00:00:01",
        completed_at="2026-05-05T00:00:42",
    )
    assert res.status == "succeeded"
    assert res.result == {"phases": []}


def test_job_status_response_status_locked():
    with pytest.raises(ValidationError):
        JobStatusResponse(
            job_id="abc",
            status="weird",  # type: ignore[arg-type]
            progress_step="x",
            created_at="2026-05-05",
        )


@pytest.mark.parametrize(
    "verdict",
    ["proceed", "caution", "contraindicated", "insufficient_information", "evidence_unavailable"],
)
def test_job_status_response_carries_every_verdict_including_abstentions(verdict):
    """Root fix: the job status response MUST carry the fail-closed abstentions
    (contraindicated / insufficient_information / evidence_unavailable) verbatim.
    The prior proceed|caution Literal forced the route to collapse an emergency
    abstention to 'caution' — downgrading a stop to a mild caution in the doctor's
    inbox badge and status."""
    res = JobStatusResponse(
        job_id="abc",
        status="succeeded",
        progress_step="done",
        verdict=verdict,  # type: ignore[arg-type]
        created_at="2026-05-05T00:00:00",
    )
    assert res.verdict == verdict
