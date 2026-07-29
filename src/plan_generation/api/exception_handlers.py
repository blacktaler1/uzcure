"""Domain → HTTP exception mapping for plan_v2.

Registered on the FastAPI app at startup so routes only see clean DTOs and the
use case can raise pure DomainErrors without touching FastAPI types.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.features.plan_generation.domain.errors import (
    ConsentMissingError,
    DomainError,
    GenerationCooldownActive,
    CriticalAckRequired,
    JobNotApprovable,
    JobNotFound,
    LanguageComplianceError,
    LLMUpstreamError,
    PatientNotFound,
    PlanLimitExceeded,
    PlanNotDeliverable,
    PlanNotFound,
    SafetyVerdictBlocked,
    SchemaValidationError,
    TenancyViolation,
)


_STATUS_FOR_CODE: dict[type[DomainError], int] = {
    PatientNotFound: 404,
    TenancyViolation: 403,
    ConsentMissingError: 403,
    PlanLimitExceeded: 409,
    SafetyVerdictBlocked: 422,
    SchemaValidationError: 502,
    LanguageComplianceError: 502,
    LLMUpstreamError: 502,
    GenerationCooldownActive: 429,
    JobNotFound: 404,
    JobNotApprovable: 409,
    CriticalAckRequired: 409,
    PlanNotFound: 404,
    PlanNotDeliverable: 409,
}


def register(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain_error_handler(_: Request, exc: DomainError) -> JSONResponse:
        status_code = _STATUS_FOR_CODE.get(type(exc), 500)
        body: dict[str, object] = {"code": exc.code, "message": exc.message}
        if isinstance(exc, GenerationCooldownActive):
            body["retry_after_seconds"] = exc.retry_after_seconds
        return JSONResponse(status_code=status_code, content=body)
