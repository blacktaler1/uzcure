"""Application ports — type + Protocol shape tests (Steps 36–42).

Confirms every port is a runtime-checkable Protocol with the correct method
surface. Lets us fail-fast at test time when a port shape diverges from the
plan spec, instead of waiting for runtime AttributeError.
"""

from __future__ import annotations

import inspect

from app.features.plan_generation.application.ports import (
    CooldownGuard,
    ExerciseRepo,
    GenerationOutcome,
    JobRecord,
    JobRepo,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    PatientRepo,
    PatientSnapshot,
    RegistryExercise,
)


# ──────────────────────────────────────────────────────────────────────────
# Value objects must be frozen dataclasses
# ──────────────────────────────────────────────────────────────────────────


def test_value_objects_frozen():
    import dataclasses

    for cls in (
        PatientSnapshot,
        RegistryExercise,
        LLMRequest,
        LLMResponse,
        JobRecord,
        GenerationOutcome,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


# ──────────────────────────────────────────────────────────────────────────
# PatientRepo (Step 37)
# ──────────────────────────────────────────────────────────────────────────


def test_patient_repo_protocol_shape():
    expected = {
        "get_for_generation",
        "doctor_has_patient",
        "can_add_new_plan",
        "has_ai_consent",
    }
    actual = {
        n
        for n, _ in inspect.getmembers(PatientRepo, predicate=inspect.isfunction)
        if not n.startswith("_")
    }
    assert expected.issubset(actual), f"PatientRepo missing: {expected - actual}"


# ──────────────────────────────────────────────────────────────────────────
# ExerciseRepo (Step 39)
# ──────────────────────────────────────────────────────────────────────────


def test_exercise_repo_protocol_shape():
    expected = {"pool_for", "get", "log_ai_suggested"}
    actual = {
        n
        for n, _ in inspect.getmembers(ExerciseRepo, predicate=inspect.isfunction)
        if not n.startswith("_")
    }
    assert expected.issubset(actual), f"ExerciseRepo missing: {expected - actual}"


# ──────────────────────────────────────────────────────────────────────────
# LLMProvider (Step 41)
# ──────────────────────────────────────────────────────────────────────────


def test_llm_provider_protocol_shape():
    actual = {
        n
        for n, _ in inspect.getmembers(LLMProvider, predicate=inspect.isfunction)
        if not n.startswith("_")
    }
    assert "call_with_tool" in actual


def test_llm_request_fields():
    req = LLMRequest(
        system=[{"type": "text", "text": "hi"}],
        user_message="case",
        tool_schema={"type": "object"},
        tool_name="submit_plan",
    )
    assert req.tool_name == "submit_plan"
    assert req.max_output_tokens > 0


def test_llm_response_fields():
    res = LLMResponse(
        tool_input={"phases": []},
        raw_text="",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        latency_ms=100,
        model="claude-opus-4-7",
        estimated_cost_usd=0.01,
    )
    assert res.tool_input == {"phases": []}
    assert res.estimated_cost_usd == 0.01


# ──────────────────────────────────────────────────────────────────────────
# JobRepo (Step 42)
# ──────────────────────────────────────────────────────────────────────────


def test_job_repo_protocol_shape():
    expected = {
        "create",
        "get",
        "mark_running",
        "update_progress",
        "mark_succeeded",
        "mark_failed",
        "cancel",
    }
    actual = {
        n
        for n, _ in inspect.getmembers(JobRepo, predicate=inspect.isfunction)
        if not n.startswith("_")
    }
    assert expected.issubset(actual), f"JobRepo missing: {expected - actual}"


def test_job_record_full_shape():
    rec = JobRecord(
        job_id="abc",
        status="queued",
        progress_step="queued",
        result=None,
        error_code=None,
        error_message=None,
        created_at="2026-05-05T00:00:00",
        started_at=None,
        completed_at=None,
    )
    assert rec.job_id == "abc"
    assert rec.status == "queued"


# ──────────────────────────────────────────────────────────────────────────
# CooldownGuard (Step 43 — same plan section)
# ──────────────────────────────────────────────────────────────────────────


def test_cooldown_guard_protocol_shape():
    expected = {"is_cooling_down", "remaining_seconds", "record_failure"}
    actual = {
        n
        for n, _ in inspect.getmembers(CooldownGuard, predicate=inspect.isfunction)
        if not n.startswith("_")
    }
    assert expected.issubset(actual)


# ──────────────────────────────────────────────────────────────────────────
# GenerationOutcome (Step 44)
# ──────────────────────────────────────────────────────────────────────────


def test_generation_outcome_fields():
    from app.features.plan_generation.domain import (
        Actions,
        DietPlan,
        GeneratedPlan,
        Phase,
        PlanMetadata,
        Exercise,
    )

    plan = GeneratedPlan(
        plan_metadata=PlanMetadata(
            patient_name="X",
            patient_id="0",
            diagnosis_short="t",
            start_date="2026-05-05",
            total_duration_weeks=1,
            total_duration_days=7,
            reward_day="Mon",
            post_discharge_cap_weeks=1,
            post_discharge_note="x",
        ),
        phases=[
            Phase(
                phase_number=1,
                phase_name="x",
                duration="1w",
                goal="g",
                exercises=[Exercise(name="A")],
            )
        ],
        diet_plan=DietPlan(),
        emergency_action="x",
        actions=Actions(approve_button="A", edit_button="E", reject_button="R"),
    )
    outcome = GenerationOutcome(
        plan=plan,
        llm_input_tokens=100,
        llm_output_tokens=200,
        llm_cache_read_tokens=0,
        llm_latency_ms=42,
        estimated_cost_usd=0.05,
        ai_suggested_exercise_count=0,
    )
    assert outcome.plan is plan
