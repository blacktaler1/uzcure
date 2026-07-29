"""Plan v2 router integration tests (Steps 82–87).

POST 202 enqueue + GET poll + DELETE cancel paths. Worker is patched to
no-op so we control job state via repo writes.
"""

from __future__ import annotations

import os

# Disable rate limiting BEFORE FastAPI app import. The limiter reads
# DISABLE_RATE_LIMIT at module import time.
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "sqlite" in os.environ.get("DATABASE_URL", "sqlite:///").lower(),
        reason="API integration tests require Postgres",
    ),
]


_TEST_DOCTOR_ID = 7
_TEST_CLINIC_ID = 1
_CASE_TEXT = (
    "65-year-old female, post-TKR rehab, 5 days post-op, ROM 0-70 deg. Pain 4/10."
)


class _FakeUser:
    id = _TEST_DOCTOR_ID
    role = "doctor"
    is_verified = True


@pytest.fixture(autouse=True)
def _disable_rate_limit(monkeypatch):
    """Disable rate limiting (SlowAPI limiter + middleware bypass) so test
    cases don't trip 429 after a handful of requests on the shared TestClient
    client IP."""
    from app.core import rate_limit, rate_middleware
    monkeypatch.setattr(rate_limit.limiter, "enabled", False)
    monkeypatch.setattr(
        rate_middleware, "_rate_limit_disabled_for_tests", lambda: True
    )


@pytest.fixture
def client(monkeypatch):
    # Patch worker so jobs stay 'queued' and we can drive state directly.
    monkeypatch.setattr(
        "app.features.plan_generation.api.routes.schedule_job",
        lambda job_id, payload: None,
    )
    from app.main import app
    from app.api.deps import (
        get_current_clinic_id,
        get_current_user,
        get_token_from_request,
        require_verified_doctor,
    )

    # Override auth chain so tests don't go through JWT decode + DB user
    # lookup (which decrypts encrypted PHI columns and may fail on dev DB
    # rows with mismatched ENCRYPTION_KEY_V1).
    fake_user = _FakeUser()
    app.dependency_overrides[get_token_from_request] = lambda: "fake-token"
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_current_clinic_id] = lambda: _TEST_CLINIC_ID
    require_factory = require_verified_doctor()
    app.dependency_overrides[require_factory] = lambda: fake_user

    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    # Auth deps overridden; headers retained for header passthrough only.
    return {
        "X-Clinic-ID": str(_TEST_CLINIC_ID),
        "X-Clinic-Slug": str(_TEST_CLINIC_ID),
    }


@pytest.fixture
def unauthenticated_client(monkeypatch):
    """A separate client WITHOUT auth overrides, for negative auth tests."""
    monkeypatch.setattr(
        "app.features.plan_generation.api.routes.schedule_job",
        lambda job_id, payload: None,
    )
    from app.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup():
    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        db.execute(
            text(
                "DELETE FROM plan_generation_jobs "
                "WHERE doctor_id=:did AND clinic_id=:cid"
            ),
            {"did": _TEST_DOCTOR_ID, "cid": _TEST_CLINIC_ID},
        )
        db.commit()
        yield
        db.execute(
            text(
                "DELETE FROM plan_generation_jobs "
                "WHERE doctor_id=:did AND clinic_id=:cid"
            ),
            {"did": _TEST_DOCTOR_ID, "cid": _TEST_CLINIC_ID},
        )
        db.commit()
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────
# Step 83 — POST enqueue
# ──────────────────────────────────────────────────────────────────────────


def test_post_returns_202_with_job_id(client, auth_headers):
    res = client.post(
        "/api/plans/jobs",
        headers=auth_headers,
        json={"case_history_text": _CASE_TEXT, "output_language": "uzbek"},
    )
    assert res.status_code == 202
    data = res.json()
    assert data["status"] == "queued"
    assert data["job_id"]
    assert len(data["job_id"]) == 32  # UUID4 hex


def test_post_rejects_short_case_history(client, auth_headers):
    res = client.post(
        "/api/plans/jobs",
        headers=auth_headers,
        json={"case_history_text": "short"},
    )
    assert res.status_code == 422


def test_post_requires_auth(unauthenticated_client):
    res = unauthenticated_client.post(
        "/api/plans/jobs",
        json={"case_history_text": _CASE_TEXT},
    )
    assert res.status_code in (401, 403)


# ──────────────────────────────────────────────────────────────────────────
# Step 84 — GET poll
# ──────────────────────────────────────────────────────────────────────────


def test_get_returns_queued_status(client, auth_headers):
    post = client.post(
        "/api/plans/jobs",
        headers=auth_headers,
        json={"case_history_text": _CASE_TEXT, "output_language": "uzbek"},
    )
    job_id = post.json()["job_id"]

    res = client.get(f"/api/plans/jobs/{job_id}", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"
    assert data["progress_step"] == "queued"


def test_get_unknown_job_returns_404(client, auth_headers):
    res = client.get(
        "/api/plans/jobs/00000000000000000000000000000000",
        headers=auth_headers,
    )
    assert res.status_code == 404


def test_get_requires_auth(unauthenticated_client):
    res = unauthenticated_client.get("/api/plans/jobs/abc")
    assert res.status_code in (401, 403)


# ──────────────────────────────────────────────────────────────────────────
# Step 85 — DELETE cancel
# ──────────────────────────────────────────────────────────────────────────


def test_delete_cancels_queued_job(client, auth_headers):
    post = client.post(
        "/api/plans/jobs",
        headers=auth_headers,
        json={"case_history_text": _CASE_TEXT},
    )
    job_id = post.json()["job_id"]

    res = client.delete(f"/api/plans/jobs/{job_id}", headers=auth_headers)
    assert res.status_code == 200
    assert res.json() == {"job_id": job_id, "status": "cancelled"}

    # GET shows cancelled
    follow = client.get(f"/api/plans/jobs/{job_id}", headers=auth_headers)
    assert follow.json()["status"] == "cancelled"


def test_delete_unknown_returns_409(client, auth_headers):
    res = client.delete(
        "/api/plans/jobs/00000000000000000000000000000000",
        headers=auth_headers,
    )
    assert res.status_code == 409


# ──────────────────────────────────────────────────────────────────────────
# Step 87 — Router wired in main.py
# ──────────────────────────────────────────────────────────────────────────


def test_openapi_includes_v2_endpoints(unauthenticated_client):
    res = unauthenticated_client.get("/openapi.json")
    assert res.status_code == 200
    paths = res.json()["paths"]
    assert "/api/plans/jobs" in paths
    assert "/api/plans/jobs/{job_id}" in paths
