"""Domain → HTTP exception handler tests (Step 86)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.features.plan_generation.api.exception_handlers import register
from app.features.plan_generation.domain import (
    ConsentMissingError,
    LanguageComplianceError,
    LLMUpstreamError,
    PatientNotFound,
    PlanLimitExceeded,
    SafetyVerdictBlocked,
    SchemaValidationError,
    TenancyViolation,
)
from app.features.plan_generation.domain.errors import GenerationCooldownActive


@pytest.fixture
def app_with_handler():
    """Minimal app that raises a chosen DomainError on /raise."""
    app = FastAPI()
    register(app)
    return app


@pytest.fixture
def client(app_with_handler):
    return TestClient(app_with_handler)


def _add_endpoint(app, exc):
    @app.get("/raise")
    async def _raise():
        raise exc


@pytest.mark.parametrize(
    "exc,expected_status,expected_code",
    [
        (PatientNotFound("missing"), 404, "PATIENT_NOT_FOUND"),
        (TenancyViolation("not assigned"), 403, "TENANCY_VIOLATION"),
        (ConsentMissingError("no consent"), 403, "AI_CONSENT_MISSING"),
        (PlanLimitExceeded("over limit"), 409, "PLAN_LIMIT_EXCEEDED"),
        (SafetyVerdictBlocked("blocked"), 422, "SAFETY_BLOCKED"),
        (SchemaValidationError("schema bad"), 502, "SCHEMA_VALIDATION_FAILED"),
        (LanguageComplianceError("wrong lang"), 502, "LANGUAGE_MISMATCH"),
        (LLMUpstreamError("anthropic 500"), 502, "LLM_UPSTREAM_ERROR"),
    ],
)
def test_domain_error_maps_to_http(app_with_handler, exc, expected_status, expected_code):
    _add_endpoint(app_with_handler, exc)
    client = TestClient(app_with_handler)
    res = client.get("/raise")
    assert res.status_code == expected_status
    body = res.json()
    assert body["code"] == expected_code
    assert body["message"]


def test_cooldown_includes_retry_after(app_with_handler):
    _add_endpoint(
        app_with_handler,
        GenerationCooldownActive("wait 42s", retry_after_seconds=42),
    )
    client = TestClient(app_with_handler)
    res = client.get("/raise")
    assert res.status_code == 429
    body = res.json()
    assert body["code"] == "GENERATION_COOLDOWN"
    assert body["retry_after_seconds"] == 42
