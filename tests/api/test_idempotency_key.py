"""Idempotency-Key header test (Step 89)."""

from __future__ import annotations

import os

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
_CASE_TEXT = "65yo female, post-TKR rehab, 5 days post-op, ROM 0-70 deg."


class _FakeUser:
    id = _TEST_DOCTOR_ID
    role = "doctor"
    is_verified = True


@pytest.fixture(autouse=True)
def _disable_rate_limit(monkeypatch):
    from app.core import rate_limit, rate_middleware
    monkeypatch.setattr(rate_limit.limiter, "enabled", False)
    monkeypatch.setattr(
        rate_middleware, "_rate_limit_disabled_for_tests", lambda: True
    )


@pytest.fixture
def client(monkeypatch):
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

    fake = _FakeUser()
    app.dependency_overrides[get_token_from_request] = lambda: "fake-token"
    app.dependency_overrides[get_current_user] = lambda: fake
    app.dependency_overrides[get_current_clinic_id] = lambda: _TEST_CLINIC_ID
    require_factory = require_verified_doctor()
    app.dependency_overrides[require_factory] = lambda: fake

    yield TestClient(app)
    app.dependency_overrides.clear()


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


def _post(client, body, idempotency_key=None):
    headers = {
        "X-Clinic-ID": str(_TEST_CLINIC_ID),
        "X-Clinic-Slug": str(_TEST_CLINIC_ID),
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return client.post("/api/plans/jobs", headers=headers, json=body)


def test_same_key_same_payload_returns_same_job(client):
    body = {"case_history_text": _CASE_TEXT, "output_language": "uzbek"}
    res1 = _post(client, body, idempotency_key="key-1")
    res2 = _post(client, body, idempotency_key="key-1")
    assert res1.status_code == 202
    assert res2.status_code == 202
    assert res1.json()["job_id"] == res2.json()["job_id"]


def test_different_keys_create_distinct_jobs(client):
    body = {"case_history_text": _CASE_TEXT}
    res1 = _post(client, body, idempotency_key="key-A")
    res2 = _post(client, body, idempotency_key="key-B")
    assert res1.json()["job_id"] != res2.json()["job_id"]


def test_no_key_identical_inflight_collapses_to_one_job(client):
    """Server-enforced idempotency: with NO Idempotency-Key (the real generate
    call-site), two identical in-flight submits collapse to ONE job instead of
    billing a second full generation (audit E3 — the 0.9s double-submit)."""
    body = {"case_history_text": _CASE_TEXT}
    res1 = _post(client, body)
    res2 = _post(client, body)
    assert res1.status_code == 202
    assert res2.status_code == 202
    assert res1.json()["job_id"] == res2.json()["job_id"]


def test_no_key_reroll_after_terminal_creates_new_job(client):
    """The in-flight dedup must NEVER block a deliberate re-generation once the
    first attempt has finished — only queued|running jobs collapse."""
    from app.db.database import SessionLocal

    body = {"case_history_text": _CASE_TEXT}
    res1 = _post(client, body)
    job1 = res1.json()["job_id"]

    # Drive the first job to a terminal state (schedule_job is mocked to no-op,
    # so it stays 'queued' otherwise).
    db = SessionLocal()
    try:
        db.execute(
            text("UPDATE plan_generation_jobs SET status='succeeded' WHERE id=:id"),
            {"id": job1},
        )
        db.commit()
    finally:
        db.close()

    res2 = _post(client, body)
    assert res2.json()["job_id"] != job1


def test_same_key_different_payload_creates_distinct_jobs(client):
    res1 = _post(
        client,
        {"case_history_text": _CASE_TEXT, "output_language": "uzbek"},
        idempotency_key="key-X",
    )
    res2 = _post(
        client,
        {"case_history_text": _CASE_TEXT + " updated.", "output_language": "uzbek"},
        idempotency_key="key-X",
    )
    assert res1.json()["job_id"] != res2.json()["job_id"]
