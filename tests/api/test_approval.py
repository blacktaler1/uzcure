"""Stage K — approval / delivery route tests.

Drives a job into 'succeeded' state with a synthetic plan in result_json,
then exercises approve / reject / edit / deliver lifecycle. No LLM call —
the worker is patched out and we write directly into plan_generation_jobs.
"""

from __future__ import annotations

import os
import uuid
from datetime import date

os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "sqlite" in os.environ.get("DATABASE_URL", "sqlite:///").lower(),
        reason="approval tests require Postgres",
    ),
]


_TEST_DOCTOR_ID = 7
_TEST_CLINIC_ID = 1
_TEST_PATIENT_ID = 1


class _FakeUser:
    id = _TEST_DOCTOR_ID
    role = "doctor"
    is_verified = True


def _synthetic_plan() -> dict:
    return {
        "plan_metadata": {
            "patient_name": "Test Patient",
            "patient_id": str(_TEST_PATIENT_ID),
            "diagnosis_short": "Stage K test diagnosis",
            "start_date": date.today().isoformat(),
            "total_duration_weeks": 6,
            "total_duration_days": 42,
            "reward_day": "Sunday",
            "post_discharge_cap_weeks": 2,
            "post_discharge_note": "Continue at home",
        },
        "phases": [
            {
                "phase_number": 1,
                "phase_name": "Acute",
                "duration": "Week 1-2",
                "goal": "Pain control + ROM 0-90",
                "exercises": [
                    {
                        "name": "Quad sets",
                        "sets": 3,
                        "reps": 10,
                        "duration_minutes": 5,
                        "frequency": "3x/day",
                        "notes": "Hold 5 sec",
                        "hold_seconds": 5,
                        "intensity": "RPE 11-13",
                        "has_video": False,
                        # F2/F4 end-to-end: registry safety context + guideline citation
                        # server-filled in resolve; must survive persistence to the plan view.
                        "contraindications": "Extensor lag greater than 10 deg",
                        "evidence_source": "ESSKA Meniscus Consensus 2016",
                        "evidence_level": "B",
                    }
                ],
            },
            {
                "phase_number": 2,
                "phase_name": "Subacute",
                "duration": "Week 3-4",
                "goal": "Walk 200m unaided",
                "exercises": [
                    {
                        "name": "Heel slides",
                        "sets": 2,
                        "reps": 15,
                        "duration_minutes": 10,
                        "frequency": "2x/day",
                        "notes": "",
                        "has_video": False,
                    }
                ],
            },
        ],
        "medication_schedule": [
            {
                "time": "Morning",
                "medication": "Paracetamol",
                "dose": "500mg",
                "instructions": "After breakfast",
                "duration": "Until pain < 3/10",
            }
        ],
        "drug_interactions": [],
        "diet_plan": {
            "tailored_for": [],
            "recommended_foods": [],
            "foods_to_avoid": [],
            "supplements": [],
            "additional_recommendations": [],
        },
        "safety_alerts": [],
        "emergency_stop_criteria": ["Severe pain >7/10"],
        "emergency_action": "Call doctor",
        "actions": {
            "approve_button": "Approve",
            "edit_button": "Edit",
            "reject_button": "Reject",
        },
        # Public phases carry no exercise_id (to_api_dict strips it); the registry
        # provenance survives here in __meta and the plan writer reads it to persist
        # the real exercise_id (regression guard for the "" provenance-loss bug).
        "__meta": {
            "exercise_provenance": [
                {
                    "phase_number": 1,
                    "index": 0,
                    "exercise_id": "ORTHO_KNEE_QUADSET_001",
                    "origin": "registry",
                    "ai_suggested": False,
                    "evidence_pmids": [],
                },
                {
                    "phase_number": 2,
                    "index": 0,
                    "exercise_id": "ORTHO_KNEE_HEELSLIDE_002",
                    "origin": "registry",
                    "ai_suggested": False,
                    "evidence_pmids": [],
                },
            ]
        },
    }


@pytest.fixture(autouse=True)
def _disable_rate_limit(monkeypatch):
    from app.core import rate_limit, rate_middleware

    monkeypatch.setattr(rate_limit.limiter, "enabled", False)
    monkeypatch.setattr(rate_middleware, "_rate_limit_disabled_for_tests", lambda: True)


@pytest.fixture(autouse=True)
def _seed_fk_targets():
    """Ensure the FK targets (clinic / doctor user / patient) referenced by the
    seeded jobs exist. The Postgres harness TRUNCATEs between tests, so this
    re-seeds per test. PHI columns are nullable (encrypted) and left empty."""
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        db.execute(
            text(
                """
                INSERT INTO clinics
                    (id, name, slug, country, max_patients, max_doctors,
                     is_active, plan_tier)
                VALUES (:cid, 'Test Clinic', 'default', 'UZ', 1000, 100, true,
                        'standard')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"cid": _TEST_CLINIC_ID},
        )
        db.execute(
            text(
                """
                INSERT INTO users
                    (id, phone_verified, role, is_active, created_at, updated_at)
                VALUES (:uid, true, 'doctor', true, NOW(), NOW())
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"uid": _TEST_DOCTOR_ID},
        )
        db.execute(
            text(
                """
                INSERT INTO doctors
                    (user_id, specialty, is_verified, is_public_profile,
                     review_count)
                VALUES (:uid, 'rehab', true, false, 0)
                ON CONFLICT (user_id) DO NOTHING
                """
            ),
            {"uid": _TEST_DOCTOR_ID},
        )
        db.execute(
            text(
                """
                INSERT INTO patients (id, clinic_id, user_id, enrollment_status)
                VALUES (:pid, :cid, :uid, 'active')
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"pid": _TEST_PATIENT_ID, "cid": _TEST_CLINIC_ID, "uid": _TEST_DOCTOR_ID},
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    yield


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
    app.dependency_overrides[get_token_from_request] = lambda: "fake"
    app.dependency_overrides[get_current_user] = lambda: fake
    app.dependency_overrides[get_current_clinic_id] = lambda: _TEST_CLINIC_ID
    require_factory = require_verified_doctor()
    app.dependency_overrides[require_factory] = lambda: fake

    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed_succeeded_job(db, *, status: str = "succeeded") -> str:
    # Insert through the ORM so request_payload / result_json (EncryptedJSON,
    # Fernet at the DB layer) are written as proper ciphertext — a raw-SQL
    # plaintext insert fails to decrypt on read after the PHI encryption batch.
    from datetime import datetime, timezone

    from app.models.plan_generation_job import PlanGenerationJob

    job_id = uuid.uuid4().hex
    plan = _synthetic_plan()
    job = PlanGenerationJob(
        id=job_id,
        doctor_id=_TEST_DOCTOR_ID,
        clinic_id=_TEST_CLINIC_ID,
        patient_id=_TEST_PATIENT_ID,
        request_payload={"output_language": "uz"},
        request_hash=uuid.uuid4().hex,
        status=status,
        progress_step="done",
        result_json=plan,
        edit_count=0,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    return job_id


@pytest.fixture
def db_session():
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _cleanup_job(db, job_id: str) -> None:
    # Tests that assert an expected error (e.g. a 403 TENANCY_VIOLATION on
    # deliver) leave the session's transaction in an aborted state. Roll back
    # first so teardown DELETEs run on a clean transaction rather than hitting
    # "current transaction is aborted".
    db.rollback()
    db.execute(
        text(
            "DELETE FROM daily_tasks WHERE plan_id IN "
            "(SELECT plan_id FROM plan_generation_jobs WHERE id=:jid)"
        ),
        {"jid": job_id},
    )
    db.execute(
        text(
            "DELETE FROM patient_medications WHERE patient_id = :pid "
            "AND clinic_id = :cid AND entered_by = :did"
        ),
        {"pid": _TEST_PATIENT_ID, "cid": _TEST_CLINIC_ID, "did": _TEST_DOCTOR_ID},
    )
    db.execute(
        text(
            "DELETE FROM plan_visibility_rules WHERE plan_id IN "
            "(SELECT plan_id FROM plan_generation_jobs WHERE id=:jid)"
        ),
        {"jid": job_id},
    )
    db.execute(
        text(
            "DELETE FROM plan_exercises WHERE plan_id IN "
            "(SELECT plan_id FROM plan_generation_jobs WHERE id=:jid)"
        ),
        {"jid": job_id},
    )
    db.execute(
        text(
            "DELETE FROM rehab_phases WHERE plan_id IN "
            "(SELECT plan_id FROM plan_generation_jobs WHERE id=:jid)"
        ),
        {"jid": job_id},
    )
    db.execute(
        text(
            "DELETE FROM plan_delivery_confirmation WHERE plan_id IN "
            "(SELECT plan_id FROM plan_generation_jobs WHERE id=:jid)"
        ),
        {"jid": job_id},
    )
    db.execute(
        text(
            "DELETE FROM rehab_plans WHERE id IN "
            "(SELECT plan_id FROM plan_generation_jobs WHERE id=:jid)"
        ),
        {"jid": job_id},
    )
    db.execute(text("DELETE FROM plan_generation_jobs WHERE id=:jid"), {"jid": job_id})
    db.commit()


def test_approve_writes_invisible_plan(client, db_session):
    """Design 1: approve fully writes the plan but it is NOT patient-visible —
    is_current_version=false, status='approved', meds inactive, no daily_tasks."""
    job_id = _seed_succeeded_job(db_session)
    try:
        res = client.post(f"/api/plans/jobs/{job_id}/approve", json={})
        assert res.status_code == 201, res.text
        body = res.json()
        assert body["status"] == "approved"
        assert body["plan_id"]
        assert body["patient_id"] == _TEST_PATIENT_ID
        assert body["job_id"] == job_id

        plan_id = body["plan_id"]
        row = db_session.execute(
            text(
                "SELECT is_approved, status, is_current_version, total_phases, "
                "output_language FROM rehab_plans WHERE id=:pid"
            ),
            {"pid": plan_id},
        ).first()
        assert row.is_approved is True
        assert row.status == "approved"
        # The cutover invariant: an approved plan is INVISIBLE until deliver.
        assert row.is_current_version is False
        assert row.total_phases == 2
        assert row.output_language == "uz"

        phase_count = db_session.execute(
            text("SELECT COUNT(*) FROM rehab_phases WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).scalar()
        assert phase_count == 2

        ex_count = db_session.execute(
            text("SELECT COUNT(*) FROM plan_exercises WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).scalar()
        assert ex_count == 2

        # Registry provenance SURVIVES to persistence: the real exercise_id from
        # __meta.exercise_provenance is written, not "" (the provenance-loss bug).
        persisted_ids = {
            r[0]
            for r in db_session.execute(
                text("SELECT exercise_id FROM plan_exercises WHERE plan_id=:pid"),
                {"pid": plan_id},
            ).fetchall()
        }
        assert persisted_ids == {"ORTHO_KNEE_QUADSET_001", "ORTHO_KNEE_HEELSLIDE_002"}

        # P1 structured slot: the AI's hold_seconds is persisted to the normalized
        # plan_exercises column the patient task view renders (the full structured
        # slot set also survives in rehab_plans.ai_raw_response).
        holds = {
            r[0]
            for r in db_session.execute(
                text("SELECT hold_seconds FROM plan_exercises WHERE plan_id=:pid"),
                {"pid": plan_id},
            ).fetchall()
        }
        assert 5 in holds

        # F2/F4 end-to-end: the registry guideline citation (evidence_source +
        # evidence_level) and contraindications must persist to plan_exercises so the
        # DELIVERED plan view carries them — not only the generation result. Regression
        # guard for the "citations dropped at persistence" bug.
        cite = db_session.execute(
            text(
                "SELECT evidence_source, evidence_level, contraindications "
                "FROM plan_exercises WHERE plan_id=:pid AND exercise_id=:eid"
            ),
            {"pid": plan_id, "eid": "ORTHO_KNEE_QUADSET_001"},
        ).fetchone()
        assert cite is not None
        assert cite[0] == "ESSKA Meniscus Consensus 2016"
        assert cite[1] == "B"
        assert cite[2] == "Extensor lag greater than 10 deg"

        # The plan's medications are NOT written at approve. They become the
        # patient's record (patient_medications) only at DELIVER, pending
        # reconciliation. The legacy medication_schedules table was removed.
        vis_count = db_session.execute(
            text("SELECT COUNT(*) FROM plan_visibility_rules WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).scalar()
        assert vis_count == 1

        # No daily_tasks until deliver.
        daily_count = db_session.execute(
            text("SELECT COUNT(*) FROM daily_tasks WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).scalar()
        assert daily_count == 0
    finally:
        _cleanup_job(db_session, job_id)


def test_approve_sets_per_phase_sort_order(client, db_session):
    """#4: plan_writer MUST persist plan_exercises.sort_order per phase
    ((phase_number*100)+index, mirroring the manual builder plan_save.py:374), so
    task_scheduler's sort_order phase bands don't collapse. With every AI-plan
    exercise at the default sort_order=0, `_get_phase_exercises` matches nothing for
    phase-2+ dates and falls back to 'all active exercises' — scheduling phase-3
    high-load from ~day 15 (once past the materialized phase-1 window). The synthetic
    plan has one exercise in phase 1 (idx 0 -> 100) and one in phase 2 (idx 0 -> 200)."""
    job_id = _seed_succeeded_job(db_session)
    try:
        res = client.post(f"/api/plans/jobs/{job_id}/approve", json={})
        assert res.status_code == 201, res.text
        plan_id = res.json()["plan_id"]
        rows = db_session.execute(
            text(
                "SELECT phase_number, sort_order FROM plan_exercises "
                "WHERE plan_id = :pid ORDER BY phase_number"
            ),
            {"pid": plan_id},
        ).fetchall()
        by_phase = {r.phase_number: r.sort_order for r in rows}
        assert by_phase.get(1) == 100, f"phase-1 sort_order should be 100, got {by_phase}"
        assert by_phase.get(2) == 200, f"phase-2 sort_order should be 200, got {by_phase}"
        # The core desync guard: the two phases MUST NOT share a sort_order (the bug
        # persisted every exercise at 0, so all phases matched phase 1's band / fell
        # back to all-exercises for later phases).
        assert by_phase[1] != by_phase[2]
    finally:
        _cleanup_job(db_session, job_id)


def test_approve_running_job_returns_409(client, db_session):
    job_id = _seed_succeeded_job(db_session, status="running")
    try:
        res = client.post(f"/api/plans/jobs/{job_id}/approve", json={})
        assert res.status_code == 409
        assert res.json()["code"] == "JOB_NOT_APPROVABLE"
    finally:
        _cleanup_job(db_session, job_id)


def test_approve_unknown_job_returns_404(client):
    res = client.post(f"/api/plans/jobs/{uuid.uuid4().hex}/approve", json={})
    assert res.status_code == 404


def test_reject_marks_job(client, db_session):
    job_id = _seed_succeeded_job(db_session)
    try:
        res = client.post(
            f"/api/plans/jobs/{job_id}/reject",
            json={"reason": "Doctor rejected for clinical concern"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "rejected"
        assert body["rejection_reason"] == "Doctor rejected for clinical concern"

        row = db_session.execute(
            text(
                "SELECT status, rejection_reason, rejected_at "
                "FROM plan_generation_jobs WHERE id=:jid"
            ),
            {"jid": job_id},
        ).first()
        assert row.status == "rejected"
        assert row.rejected_at is not None
    finally:
        _cleanup_job(db_session, job_id)


def test_edit_replaces_result(client, db_session):
    job_id = _seed_succeeded_job(db_session)
    try:
        edited = _synthetic_plan()
        edited["plan_metadata"]["diagnosis_short"] = "Edited diagnosis"

        res = client.post(
            f"/api/plans/jobs/{job_id}/edit",
            json={"edited_result": edited},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["edit_count"] == 1

        # Second edit increments
        res2 = client.post(
            f"/api/plans/jobs/{job_id}/edit",
            json={"edited_result": edited},
        )
        assert res2.json()["edit_count"] == 2
    finally:
        _cleanup_job(db_session, job_id)


def test_edit_rejects_invalid_schema(client, db_session):
    job_id = _seed_succeeded_job(db_session)
    try:
        res = client.post(
            f"/api/plans/jobs/{job_id}/edit",
            json={"edited_result": {"phases": []}},  # missing required fields
        )
        assert res.status_code == 409
        assert res.json()["code"] == "JOB_NOT_APPROVABLE"
    finally:
        _cleanup_job(db_session, job_id)


def _seed_ledger_entry(db, *, patient_id: int = _TEST_PATIENT_ID):
    """Seed an evidence_ledger row so the deliver signoff (FK -> ledger) can be
    written. The worker normally writes this at generation; the tests seed the
    job directly, so we add it here."""
    ledger_uuid = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO evidence_ledger
                (plan_id, patient_id, plan_version, guideline_pack_ids,
                 safety_rules_fired, drug_verification, is_fully_compliant,
                 requires_doctor_review, plan_hash, ledger_signature,
                 signing_key_version, generated_at)
            VALUES
                (:pid, :patient_id, 1, '{}', '{}', '{}'::json, false, true,
                 :phash, :sig, 1, NOW())
            """
        ),
        {
            "pid": ledger_uuid,
            "patient_id": patient_id,
            "phash": uuid.uuid4().hex,
            "sig": uuid.uuid4().hex,
        },
    )
    db.commit()
    return ledger_uuid


def _cleanup_ledger(db, ledger_uuid: str) -> None:
    db.execute(
        text("DELETE FROM doctor_signoffs WHERE plan_id=:lid"),
        {"lid": ledger_uuid},
    )
    db.execute(
        text("DELETE FROM evidence_ledger WHERE plan_id=:lid"),
        {"lid": ledger_uuid},
    )
    db.commit()


def test_deliver_makes_plan_visible_and_signs_off(client, db_session):
    """Deliver is the publish gate: it flips the plan to delivered + current,
    generates phase-1 daily_tasks, activates meds, and writes a DoctorSignoff."""
    ledger_uuid = _seed_ledger_entry(db_session)
    job_id = _seed_succeeded_job(db_session)
    try:
        approve = client.post(f"/api/plans/jobs/{job_id}/approve", json={})
        plan_id = approve.json()["plan_id"]

        # Pre-deliver: invisible (no daily_tasks).
        pre_daily = db_session.execute(
            text("SELECT COUNT(*) FROM daily_tasks WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).scalar()
        assert pre_daily == 0

        deliver = client.post(f"/api/plans/{plan_id}/deliver")
        assert deliver.status_code == 200, deliver.text
        body = deliver.json()
        assert body["plan_id"] == plan_id
        assert body["visibility_max_phase"] == 1

        row = db_session.execute(
            text("SELECT status, is_current_version, delivered_at FROM rehab_plans WHERE id=:pid"),
            {"pid": plan_id},
        ).first()
        assert row.status == "delivered"
        assert row.is_current_version is True
        assert row.delivered_at is not None

        # daily_tasks generated at deliver (14 days x 1 phase-1 exercise).
        daily_count = db_session.execute(
            text("SELECT COUNT(*) FROM daily_tasks WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).scalar()
        assert daily_count == 14

        # The plan's medication becomes the patient's record at deliver, as a
        # pending reconciliation row (patient_medications) — the legacy
        # medication_schedules table is gone.
        pending_med_count = db_session.execute(
            text(
                "SELECT COUNT(*) FROM patient_medications "
                "WHERE patient_id=:pid AND source='ai_plan' "
                "AND reconciliation_status='pending'"
            ),
            {"pid": _TEST_PATIENT_ID},
        ).scalar()
        assert pending_med_count == 1

        conf = db_session.execute(
            text("SELECT delivered_at FROM plan_delivery_confirmation WHERE plan_id=:pid"),
            {"pid": plan_id},
        ).first()
        assert conf is not None

        # DoctorSignoff row written at delivery (action=APPROVE).
        signoff = db_session.execute(
            text("SELECT action, doctor_user_id FROM doctor_signoffs WHERE plan_id=:lid"),
            {"lid": ledger_uuid},
        ).first()
        assert signoff is not None
        assert signoff.action == "APPROVE"
        assert signoff.doctor_user_id == _TEST_DOCTOR_ID
    finally:
        _cleanup_job(db_session, job_id)
        _cleanup_ledger(db_session, ledger_uuid)


def _assign_doctor_patient(db):
    """Seed the DoctorPatient assignment the patient-medications endpoints
    require for manage access (doctor role → must be assigned)."""
    db.execute(
        text(
            """
            INSERT INTO doctor_patient
                (clinic_id, patient_id, doctor_id, role, status,
                 created_at, assigned_at)
            VALUES (:cid, :pid, :did, 'primary', 'active', NOW(), NOW())
            ON CONFLICT DO NOTHING
            """
        ),
        {"cid": _TEST_CLINIC_ID, "pid": _TEST_PATIENT_ID, "did": _TEST_DOCTOR_ID},
    )
    db.commit()


def test_deliver_ai_medications_land_pending_reconciliation(client, db_session):
    """Medication reconciliation gate (audit P0 #9): an AI plan's medication is
    NOT activated into the patient's record at deliver. It lands source='ai_plan',
    reconciliation_status='pending', is_active=False so it neither fires reminders
    nor feeds `known_medications` into the next plan's safety reasoning until a
    doctor reconciles it."""
    from app.models.patient_medication import PatientMedication

    ledger_uuid = _seed_ledger_entry(db_session)
    job_id = _seed_succeeded_job(db_session)
    try:
        plan_id = client.post(f"/api/plans/jobs/{job_id}/approve", json={}).json()["plan_id"]
        assert client.post(f"/api/plans/{plan_id}/deliver").status_code == 200

        meds = (
            db_session.query(PatientMedication)
            .filter(
                PatientMedication.patient_id == _TEST_PATIENT_ID,
                PatientMedication.clinic_id == _TEST_CLINIC_ID,
            )
            .all()
        )
        para = [m for m in meds if (m.drug_name or "").casefold() == "paracetamol"]
        assert len(para) == 1, "AI medication must be proposed exactly once"
        m = para[0]
        assert m.source == "ai_plan"
        assert m.reconciliation_status == "pending"
        assert m.is_active is False, "AI med must NOT be active before reconciliation"

        # Excluded from the safety input: known_medications reads active rows.
        active = [x for x in meds if x.is_active]
        assert active == [], "no AI medication may be active pre-reconciliation"
    finally:
        _cleanup_job(db_session, job_id)
        _cleanup_ledger(db_session, ledger_uuid)


def test_reconcile_confirm_activates_and_reject_discards(client, db_session):
    """The doctor reconcile action is the only path that turns an AI-proposed
    medication into an active, safety-bearing fact — or discards it."""
    from app.models.patient_medication import PatientMedication

    ledger_uuid = _seed_ledger_entry(db_session)
    job_id = _seed_succeeded_job(db_session)
    try:
        plan_id = client.post(f"/api/plans/jobs/{job_id}/approve", json={}).json()["plan_id"]
        assert client.post(f"/api/plans/{plan_id}/deliver").status_code == 200
        _assign_doctor_patient(db_session)

        # Pending list surfaces the one AI proposal.
        pending = client.get(f"/api/patient-medications/{_TEST_PATIENT_ID}/pending-reconciliation")
        assert pending.status_code == 200, pending.text
        body = pending.json()
        assert body["total"] == 1
        med_id = body["medications"][0]["id"]

        # Confirm → active + reconciled, with a doctor dose edit applied.
        r = client.post(
            f"/api/patient-medications/{_TEST_PATIENT_ID}/{med_id}/reconcile",
            json={"action": "confirm", "dose": "650mg"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["is_active"] is True
        assert r.json()["reconciliation_status"] == "reconciled"
        assert r.json()["dose"] == "650mg"

        db_session.expire_all()
        m = db_session.get(PatientMedication, med_id)
        assert m.is_active is True and m.reconciliation_status == "reconciled"
        assert m.entered_by == _TEST_DOCTOR_ID

        # Re-reconciling a non-pending row is a 409, not a silent no-op.
        again = client.post(
            f"/api/patient-medications/{_TEST_PATIENT_ID}/{med_id}/reconcile",
            json={"action": "confirm"},
        )
        assert again.status_code == 409

        # Reject path: seed a second pending AI med and reject it.
        pend2 = PatientMedication(
            patient_id=_TEST_PATIENT_ID,
            clinic_id=_TEST_CLINIC_ID,
            entered_by=_TEST_DOCTOR_ID,
            drug_name="Ibuprofen",
            medication_name="Ibuprofen",
            dose="200mg",
            is_active=False,
            source="ai_plan",
            reconciliation_status="pending",
        )
        db_session.add(pend2)
        db_session.commit()
        rej = client.post(
            f"/api/patient-medications/{_TEST_PATIENT_ID}/{pend2.id}/reconcile",
            json={"action": "reject"},
        )
        assert rej.status_code == 200, rej.text
        assert rej.json()["is_active"] is False
        assert rej.json()["reconciliation_status"] == "rejected"
    finally:
        _cleanup_job(db_session, job_id)
        _cleanup_ledger(db_session, ledger_uuid)


def test_reconcile_pending_bulk_activates_all_reviewed_meds(client, db_session):
    """Approve & send — the bulk reconciliation step. A single doctor action
    (POST /{patient_id}/reconcile-pending) activates every AI-proposed med the
    doctor reviewed drug-by-drug in the composer, flipping each pending row to
    active + reconciled, attributed to the doctor, with the delivered dose_times
    preserved. This is what lets the deterministic missed-dose / reminder jobs
    (which scan is_active=True rows by dose_times slot) fire. Idempotent."""
    from app.models.patient_medication import PatientMedication

    ledger_uuid = _seed_ledger_entry(db_session)
    job_id = _seed_succeeded_job(db_session)
    try:
        plan_id = client.post(f"/api/plans/jobs/{job_id}/approve", json={}).json()["plan_id"]
        assert client.post(f"/api/plans/{plan_id}/deliver").status_code == 200
        _assign_doctor_patient(db_session)

        # Pre-state: the AI med is pending + inactive (the delivered gate).
        pend = client.get(
            f"/api/patient-medications/{_TEST_PATIENT_ID}/pending-reconciliation"
        ).json()
        assert pend["total"] == 1

        # Bulk reconcile — the "send to patient" activation.
        r = client.post(f"/api/patient-medications/{_TEST_PATIENT_ID}/reconcile-pending")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["reconciled"] == 1
        assert body["medications"][0]["is_active"] is True
        assert body["medications"][0]["reconciliation_status"] == "reconciled"

        db_session.expire_all()
        meds = (
            db_session.query(PatientMedication)
            .filter(
                PatientMedication.patient_id == _TEST_PATIENT_ID,
                PatientMedication.clinic_id == _TEST_CLINIC_ID,
            )
            .all()
        )
        assert len(meds) == 1
        m = meds[0]
        assert m.is_active is True
        assert m.reconciliation_status == "reconciled"
        assert m.entered_by == _TEST_DOCTOR_ID
        # dose_times carried through delivery so the missed-dose job has slots.
        assert isinstance(m.dose_times, list) and len(m.dose_times) >= 1

        # No pending rows remain; a second call is an idempotent no-op.
        pend2 = client.get(
            f"/api/patient-medications/{_TEST_PATIENT_ID}/pending-reconciliation"
        ).json()
        assert pend2["total"] == 0
        again = client.post(f"/api/patient-medications/{_TEST_PATIENT_ID}/reconcile-pending")
        assert again.status_code == 200
        assert again.json()["reconciled"] == 0
    finally:
        _cleanup_job(db_session, job_id)
        _cleanup_ledger(db_session, ledger_uuid)


def test_e2e_patient_receives_delivered_plan(client, db_session):
    """End-to-end regression guard for the silent-failure class fixed in #159.

    Drives the doctor's real confirm flow (approve -> deliver -> activate the
    reviewed meds) and then reads the PATIENT-facing endpoints, asserting the
    patient actually receives the plan, today's exercises, and their active
    medication. Before #159 the composer only approved: the plan stayed out of
    PATIENT_VISIBLE_STATUSES with no daily_tasks and no active meds, so every one
    of these patient reads came back empty. This test fails closed if that
    regresses."""
    from types import SimpleNamespace

    from app.api.endpoints.patient_exercises import get_patient_exercises
    from app.api.endpoints.patient_medications import get_my_medications
    from app.api.endpoints.patient_plan_detail import get_patient_plan_detail

    ledger_uuid = _seed_ledger_entry(db_session)
    job_id = _seed_succeeded_job(db_session)
    try:
        # ── Doctor: the composer "approve & send" sequence ──
        appr = client.post(f"/api/plans/jobs/{job_id}/approve", json={})
        assert appr.status_code == 201, appr.text
        plan_id = appr.json()["plan_id"]
        assert client.post(f"/api/plans/{plan_id}/deliver").status_code == 200
        _assign_doctor_patient(db_session)
        assert (
            client.post(
                f"/api/patient-medications/{_TEST_PATIENT_ID}/reconcile-pending"
            ).status_code
            == 200
        )

        db_session.expire_all()
        # The seeded patient (id=1) is linked to user_id=_TEST_DOCTOR_ID; a
        # patient-role identity with that id resolves to that Patient row.
        patient = SimpleNamespace(id=_TEST_DOCTOR_ID, role="patient")

        # ── Patient sees today's exercises (materialized daily_tasks) ──
        ex = get_patient_exercises(token_data=patient, db=db_session, clinic_id=_TEST_CLINIC_ID)
        assert ex.today_schedule.total >= 1, "patient must see today's exercises"
        assert len(ex.exercises) >= 1
        assert ex.plan_status == "delivered"

        # ── Patient sees the delivered plan itself ──
        plan = get_patient_plan_detail(token_data=patient, db=db_session, clinic_id=_TEST_CLINIC_ID)
        assert plan.plan_status.status in ("delivered", "active")
        assert plan.plan_status.is_active is True
        assert plan.exercise_count.total >= 1

        # ── Patient sees their now-active medication (reminder-bearing) ──
        meds = get_my_medications(
            active_only=True, current_user=patient, db=db_session, clinic_id=_TEST_CLINIC_ID
        )
        assert meds["active_count"] >= 1
        assert len(meds["medications"]) >= 1
        first = meds["medications"][0]
        first_active = first["is_active"] if isinstance(first, dict) else first.is_active
        assert first_active is True
    finally:
        _cleanup_job(db_session, job_id)
        _cleanup_ledger(db_session, ledger_uuid)


def test_deliver_does_not_duplicate_already_active_medication(client, db_session):
    """Dedup: a drug the patient already has as an active (doctor-reconciled)
    medication is not re-proposed as a pending row when a new plan is delivered."""
    from app.models.patient_medication import PatientMedication

    ledger_uuid = _seed_ledger_entry(db_session)
    job_id = _seed_succeeded_job(db_session)
    try:
        # Doctor already has Paracetamol as an active reconciled med.
        existing = PatientMedication(
            patient_id=_TEST_PATIENT_ID,
            clinic_id=_TEST_CLINIC_ID,
            entered_by=_TEST_DOCTOR_ID,
            drug_name="paracetamol",  # different case on purpose
            medication_name="paracetamol",
            dose="500mg",
            is_active=True,
            source="doctor",
            reconciliation_status="reconciled",
        )
        db_session.add(existing)
        db_session.commit()

        plan_id = client.post(f"/api/plans/jobs/{job_id}/approve", json={}).json()["plan_id"]
        assert client.post(f"/api/plans/{plan_id}/deliver").status_code == 200

        db_session.expire_all()
        para = (
            db_session.query(PatientMedication)
            .filter(
                PatientMedication.patient_id == _TEST_PATIENT_ID,
                PatientMedication.clinic_id == _TEST_CLINIC_ID,
            )
            .all()
        )
        names = [(m.drug_name or "").casefold() for m in para]
        assert names.count("paracetamol") == 1, "must not duplicate an already-active drug"
        # The single row is the doctor's active one, untouched.
        assert para[0].is_active is True and para[0].reconciliation_status == "reconciled"
    finally:
        _cleanup_job(db_session, job_id)
        _cleanup_ledger(db_session, ledger_uuid)


def test_deliver_by_other_doctor_is_rejected(client, db_session):
    """Authz: a doctor who does not own the plan cannot deliver it (403)."""
    job_id = _seed_succeeded_job(db_session)
    try:
        approve = client.post(f"/api/plans/jobs/{job_id}/approve", json={})
        plan_id = approve.json()["plan_id"]

        # Reassign the plan to a different doctor, then attempt deliver as the
        # logged-in doctor (id=7) — ownership mismatch must 403.
        other_doctor_id = _TEST_DOCTOR_ID + 999
        # Seed the other doctor as a real user first — rehab_plans.doctor_id has
        # an FK to users.id, so the reassignment needs a valid target row.
        db_session.execute(
            text(
                "INSERT INTO users (id, phone_verified, role, is_active, "
                "created_at, updated_at) VALUES (:uid, true, 'doctor', true, "
                "NOW(), NOW()) ON CONFLICT (id) DO NOTHING"
            ),
            {"uid": other_doctor_id},
        )
        db_session.execute(
            text("UPDATE rehab_plans SET doctor_id=:did WHERE id=:pid"),
            {"did": other_doctor_id, "pid": plan_id},
        )
        db_session.commit()

        deliver = client.post(f"/api/plans/{plan_id}/deliver")
        assert deliver.status_code == 403, deliver.text
        assert deliver.json()["code"] == "TENANCY_VIOLATION"

        # Plan stays unpublished.
        row = db_session.execute(
            text("SELECT status, is_current_version FROM rehab_plans WHERE id=:pid"),
            {"pid": plan_id},
        ).first()
        assert row.status == "approved"
        assert row.is_current_version is False
    finally:
        _cleanup_job(db_session, job_id)


def test_deliver_unapproved_plan_returns_404(client, db_session):
    """A plan id that does not exist in the clinic returns 404 (not found)."""
    res = client.post("/api/plans/999999999/deliver")
    assert res.status_code == 404
    assert res.json()["code"] == "PLAN_NOT_FOUND"


def test_cancelled_job_worker_mark_succeeded_is_noop(db_session):
    """Guarded terminal write: a job cancelled mid-flight is NOT clobbered by
    the still-running worker's mark_succeeded (only 'running' -> 'succeeded')."""
    import asyncio

    from app.features.plan_generation.infrastructure.job_repo import SqlJobRepo

    # Seed a job already flipped to 'cancelled' (simulating a cancel that won
    # the race while the worker was still generating).
    job_id = _seed_succeeded_job(db_session, status="cancelled")
    try:
        repo = SqlJobRepo(db_session)
        asyncio.run(repo.mark_succeeded(job_id=job_id, result=_synthetic_plan()))
        row = db_session.execute(
            text("SELECT status FROM plan_generation_jobs WHERE id=:jid"),
            {"jid": job_id},
        ).first()
        assert row.status == "cancelled"

        # mark_failed is equally a no-op against a cancelled job.
        asyncio.run(repo.mark_failed(job_id=job_id, code="X", message="y"))
        row2 = db_session.execute(
            text("SELECT status FROM plan_generation_jobs WHERE id=:jid"),
            {"jid": job_id},
        ).first()
        assert row2.status == "cancelled"
    finally:
        _cleanup_job(db_session, job_id)
